# AGIself · D1 edu 部署主脚本
"""读状态 → 拼 45 维观测 → ONNX 推理 → 换算目标角 → 下发。整条链路就这一个文件。

与官方 lowlevel_demo.py 的关系：软启动两段式、阻尼停机、Ctrl+C 捕获三件都照搬它的做法，
只把「按斜坡算目标角」换成「按策略算目标角」，另加三样官方 demo 不需要而我们必须有的
东西 —— 硬限位夹取、非有限值拒绝、逐帧留痕。

2026-08-29 续10 相对上一版的改动，每条都对着一个具体失败：
  1. 读回限位检查加余量        吊起来 ABAD 实测 +0.490 > 写死的 0.489，上一版首跑必然误报退出
  2. 墙钟节拍代替 sleep(dt)    sleep(dt) 是「每圈多睡 dt」，实际频率必然低于 50 Hz 且会累积漂移
  3. 循环频率实测并打印        50 Hz 是训练时的拍子，实际跑多少必须有数（D29）
  4. --abad-sign              ABAD 符号待 AGIself_probe_abad.py 定，留一个开关而不是改代码
  5. 每帧校验重力投影模长      IMU 数据坏掉时模长会离开 1，这是唯一不需要外部参照的自检
  6. 逐帧留痕存 npz            「不抖」要有数字，不能只靠眼睛
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time

import numpy as np

# ---- 关节顺序 -------------------------------------------------------------
# Isaac : 分块 ABAD×4 | HIP×4 | KNEE×4，层内腿序 FL, FR, RL, RR   （D06 实测）
# SDK   : q_*_abad[4] / q_*_hip[4] / q_*_knee[4]，层内腿序 FR, FL, RR, RL
#         （出处：include/zsl-1/lowlevel.h 里 motorState 上方注释 "关节顺序: FR,FL,RR,RL"）
# 分块方式相同，只差层内腿序 ⇒ 两两交换。
# 这个排列是对合（连用两次回到原处），所以读和写共用它，不存在方向写反的问题。
LEG_PERM = np.array([1, 0, 3, 2])

# ---- 关节限位 --------------------------------------------------------------
# 这里有两套范围，管的是两件不同的事，不能混用一个数组：
#
#   URDF_*  机械止档，管【编码器读数能到哪】。读回检查用它。
#           URDF edu.urdf 实读：ABAD ±0.4887 | HIP -1.1519~+2.967 | KNEE -2.723~-0.602
#
#   CMD_*   指令允许范围，管【p_des 能填什么】。发送夹取用它。
#           取 URDF 与固件校验范围的【交集】，两者都能拦下我们。
#           固件对 KNEE 校验 -2.9~-0.65（2026-08-29 探针停机实测报错发现，文档没写），
#           上端比 URDF 窄 0.048 rad ⇒ 落在 -0.65~-0.602 的指令 URDF 放行、固件整帧拒收。
#           ABAD 与 HIP 的固件范围未知（没触发过），暂用 URDF 值；真被拒会由
#           sendMotorCmd 返回值统计当场暴露，不会静默。
#
# 为什么必须夹：p_des = default + 0.25*action，ABAD 只有 ±0.489，
# action 超过 1.96 就越界，而策略输出是无上界的高斯采样 ⇒ 限幅不是可选项。
URDF_LOWER = np.array([-0.4887] * 4 + [-1.1519] * 4 + [-2.723] * 4)
URDF_UPPER = np.array([+0.4887] * 4 + [+2.967] * 4 + [-0.602] * 4)

CMD_LOWER = URDF_LOWER.copy()
CMD_UPPER = np.minimum(URDF_UPPER, np.array([1e9] * 8 + [-0.65] * 4))

# 读回侧留的余量。理由：上面那对数是 URDF 的标称值，而编码器读数受机械止档公差与
# 零位偏置影响，可以合法地略微越过它 —— 续9 吊起来实测 ABAD 停在 +0.490 / +0.494，
# 就在标称 0.4887 外面一点点。用同一个数去卡读回值，首跑必然误报。
READBACK_MARGIN = 0.05

# 重力投影模长的容许偏差。SDK 返回单位四元数（D28 实测模长 0.9999），
# 离开 1 太远说明 IMU 数据本身坏了 —— 这是运行期唯一不依赖外部参照的自检。
GRAVITY_NORM_TOL = 0.05


def load_sdk(root):
    arch = platform.machine().replace("amd64", "x86_64").replace("arm64", "aarch64")
    sys.path.insert(0, os.path.join(os.path.expanduser(root), "lib", "zsl-1", arch))
    import mc_sdk_zsl_1_py as sdk

    return sdk


def make_converters(abad_sign):
    """造一对 SDK ↔ Isaac 的换算函数。

    abad_sign 是 ABAD 那一块的符号：+1 表示实机编码器正方向与 URDF 一致，
    -1 表示整体反向。待 AGIself_probe_abad.py 实测确定（D14）。
    符号只作用在 ABAD 块，且读写各乘一次 —— 与 LEG_PERM 一样，自己是自己的逆。
    """
    sgn = np.array([abad_sign] * 4 + [1.0] * 4 + [1.0] * 4)

    def isaac_from_sdk(a, h, k):
        v = np.concatenate([np.asarray(a)[LEG_PERM],
                            np.asarray(h)[LEG_PERM],
                            np.asarray(k)[LEG_PERM]])
        return v * sgn

    def sdk_from_isaac(v):
        v = np.asarray(v) * sgn
        return v[0:4][LEG_PERM], v[4:8][LEG_PERM], v[8:12][LEG_PERM]

    return isaac_from_sdk, sdk_from_isaac


def projected_gravity(q_wxyz):
    """世界系重力 [0,0,-1] 旋进机体系 —— 观测的第二块。

    与 get_status.py 里的闭式 gx=2(wy-xz) / gy=-2(wx+yz) / gz=1-2(w^2+z^2)
    离线比过 5000 个随机单位四元数，最大差 8.9e-16，两者可互为对照。
    约定核对（D28 实测）：水平静止 (0,0,-1)；抬头 20.7° 得 (-0.354, 0, -0.935)。
    """
    w, x, y, z = (float(t) for t in q_wxyz)
    g = np.array([0.0, 0.0, -1.0])
    qv = np.array([-x, -y, -z])          # 单位四元数的逆 = 共轭
    t = 2.0 * np.cross(qv, g)
    return g + w * t + np.cross(qv, t)


def fill(sdk, sdk_from_isaac, p_des, kp, kd):
    cmd = sdk.MotorCommand()
    pa, ph, pk = sdk_from_isaac(p_des)
    for i in range(4):
        cmd.q_des_abad[i], cmd.q_des_hip[i], cmd.q_des_knee[i] = pa[i], ph[i], pk[i]
        cmd.kp_abad[i] = cmd.kp_hip[i] = cmd.kp_knee[i] = kp
        cmd.kd_abad[i] = cmd.kd_hip[i] = cmd.kd_knee[i] = kd
    return cmd
    # qd_des_* 与 tau_*_ff 不填：motorCmd 结构体里它们的默认值就是 0.0
    # （lowlevel.h 第 10-27 行），而 Isaac 隐式执行器算的是 tau = kp(q_des-q) + kd(0-qd)，
    # 目标速度同样是 0。两边默认对上了，填了反而多一处可能写错的地方。


class Pacer:
    """按墙钟卡节拍，顺带记录真实周期。

    为什么不能用 time.sleep(dt)：那是「干完活再睡 dt」，一圈的实际耗时是
    干活时间 + dt，必然大于 dt，而且误差每圈累积。官方 demo 用 sleep(0.002)
    配 progress += 0.002 推斜坡，正是踩在这个假设上 —— 它的「2 秒」比 2 秒长。
    """

    def __init__(self, dt):
        self.dt = dt
        self.next_t = None
        self.periods = []
        self.overruns = 0

    def start(self):
        self.next_t = time.perf_counter()
        self._last = self.next_t

    def wait(self):
        now = time.perf_counter()
        self.periods.append(now - self._last)
        self._last = now
        self.next_t += self.dt
        remain = self.next_t - now
        if remain > 0:
            time.sleep(remain)
        else:
            # 这一圈的活干超了一个周期。不追补（追补会连着几圈发得过密），
            # 直接把基准挪到现在，只记一笔。
            self.overruns += 1
            self.next_t = now

    def report(self, label):
        p = np.array(self.periods[1:]) if len(self.periods) > 1 else np.array([])
        if p.size == 0:
            return
        print(f"[节拍] {label}: 目标 {1/self.dt:.1f} Hz，实测均值 {1/p.mean():.1f} Hz"
              f"（周期 {p.mean()*1e3:.2f} ms，p99 {np.percentile(p,99)*1e3:.2f} ms，"
              f"超时 {self.overruns}/{len(self.periods)} 圈）")


class Sender:
    """包一层 sendMotorCmd，把返回值真的看一眼。

    不看会怎样：固件拒收一帧时返回负数、终端上刷一行 "send cmd error"，
    但循环照跑、策略照算 —— 狗保持着上一个目标角不动，而我们以为自己在控制它。
    连续拒收超过 3 秒就触发看门狗自动趴下，诊断信息只有那行刷屏。
    2026-08-29 探针停机时整整 3 秒的阻尼指令全被拒收，就是这个形状。
    """

    def __init__(self, low, max_consecutive=5):
        self.low = low
        self.max_consecutive = max_consecutive
        self.total = 0
        self.failed = 0
        self.consecutive = 0
        self.worst = 0

    def __call__(self, cmd):
        self.total += 1
        if self.low.sendMotorCmd(cmd) < 0:
            self.failed += 1
            self.consecutive += 1
            self.worst = max(self.worst, self.consecutive)
            if self.consecutive >= self.max_consecutive:
                raise RuntimeError(
                    f"连续 {self.consecutive} 帧指令被固件拒收 —— 目标角落在允许范围外。"
                    f"终端上方的 invalid ... cmd 那行写了是哪个关节、允许范围是多少。"
                )
        else:
            self.consecutive = 0

    def report(self, label):
        if self.failed:
            print(f"[警告] {label}: {self.failed}/{self.total} 帧被拒收"
                  f"（最长连续 {self.worst} 帧）—— 这些帧狗保持着上一个目标角没动")
        else:
            print(f"[OK] {label}: {self.total} 帧全部下发成功")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local-ip", default="192.168.168.100")
    ap.add_argument("--dog-ip", default="192.168.168.168")
    ap.add_argument("--port", type=int, default=43988)
    ap.add_argument("--sdk-root", default="~/self-AGI/agibot_D1_Edu-Ultra")
    ap.add_argument("--bundle", default=None, help="缺省用包内 bundle/")
    ap.add_argument("--vx", type=float, default=0.0)
    ap.add_argument("--vy", type=float, default=0.0)
    ap.add_argument("--wz", type=float, default=0.0)
    ap.add_argument("--ramp", type=float, default=3.0, help="软启动秒数")
    ap.add_argument("--seconds", type=float, default=10.0, help="策略运行秒数")
    ap.add_argument("--stance-kp", type=float, default=None, help="软启动增益，缺省用标称值")
    ap.add_argument("--abad-sign", type=float, default=1.0, choices=[1.0, -1.0],
                    help="实机 ABAD 编码器正方向：1=与 URDF 一致，-1=反向（D14，由探针实测定）")
    ap.add_argument("--max-joint-rate", type=float, default=4.0,
                    help="目标角变化率上限 rad/s")
    ap.add_argument("--trace", default=None, help="逐帧留痕存到这个 npz 路径")
    ap.add_argument("--dry-run", action="store_true", help="只读不发，段 2 用")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    bundle = args.bundle or os.path.join(here, "..", "agiself_d1_deploy", "bundle")
    bundle = os.path.abspath(bundle)

    obs_c = json.load(open(os.path.join(bundle, "d1_edu_obs_contract.json"), encoding="utf-8"))
    default_q = np.array([j["default_pos"] for j in obs_c["joints_isaac_order"]])
    scale = float(obs_c["action"]["scale"])
    dt = float(obs_c["timing"]["control_dt"])
    kp = float(obs_c["actuator_gains"]["nominal_stiffness"])
    kd = float(obs_c["actuator_gains"]["nominal_damping"])
    kp_stance = args.stance_kp if args.stance_kp is not None else kp

    print(f"[契约] 策略 {1/dt:.1f} Hz  动作 p_des = default + {scale}*action  "
          f"标称 kp/kd = {kp}/{kd}")
    if args.abad_sign < 0:
        print("[契约] ABAD 符号取反（--abad-sign -1）")

    isaac_from_sdk, sdk_from_isaac = make_converters(args.abad_sign)

    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1      # 控制循环里不要线程池，它只带来抖动
    opts.inter_op_num_threads = 1
    sess = ort.InferenceSession(os.path.join(bundle, "policy.onnx"),
                                sess_options=opts, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    for _ in range(20):                # 预热：首帧要做内存分配与图优化，比稳态慢一个量级
        sess.run(None, {in_name: np.zeros((1, 45), dtype=np.float32)})

    sdk = load_sdk(args.sdk_root)
    low = sdk.LowLevel()
    low.initRobot(args.local_ip, args.port, args.dog_ip)

    t0 = time.time()
    while not low.haveMotorData():
        if time.time() - t0 > 10:
            sys.exit("[FAIL] 10 秒收不到电机数据 —— 多半是狗上 sdk_config.yaml 的 "
                     "target_ip 不是本机。数据流是「狗 → target_ip」，改完要重启狗。")
        time.sleep(0.01)
    print(f"[OK] 通信正常，{time.time()-t0:.2f}s")

    st = low.getMotorState()
    q0 = isaac_from_sdk(st.q_abad, st.q_hip, st.q_knee)
    print("当前关节角 (Isaac 顺序，层内 FL FR RL RR):")
    for lo, hi, nm in ((0, 4, "ABAD"), (4, 8, "HIP "), (8, 12, "KNEE")):
        print(f"  {nm} " + " ".join(f"{v:+7.3f}" for v in q0[lo:hi]))
    print("  默认角 " + " ".join(f"{v:+7.3f}" for v in default_q[[0, 4, 8]]) + "  (ABAD/HIP/KNEE)")

    # 读回检查留余量：URDF 标称限位是我们对电机的承诺，不是编码器读数的保证。
    # 吊起来腿软垂时 ABAD 会停在机械止档上，实测 +0.490 略微越过标称 0.4887。
    bad = np.where((q0 < URDF_LOWER - READBACK_MARGIN) | (q0 > URDF_UPPER + READBACK_MARGIN))[0]
    if bad.size:
        sys.exit(f"[FAIL] 关节 {bad.tolist()} 读数超出 URDF 限位 {READBACK_MARGIN} rad 以上 —— "
                 f"实机零位与 URDF 不一致（D14），或腿序映射错了（D26）。不查清不许上电。")
    edge = np.where((q0 < URDF_LOWER) | (q0 > URDF_UPPER))[0]
    if edge.size:
        print(f"[提示] 关节 {edge.tolist()} 略微超出 URDF 标称限位但在余量内 —— "
              f"腿软垂时顶住机械止档就是这个样子，正常。")
    print("[OK] 十二个关节读数都在限位内（含余量）")

    g0 = projected_gravity(low.getQuaternion())
    print(f"[OK] 重力投影 {np.round(g0,4)}  模长 {np.linalg.norm(g0):.4f}（应≈1）")

    if args.dry_run:
        gyro = np.asarray(low.getBodyGyro())
        print(f"     机体角速度 {np.round(gyro,4)} rad/s")
        print("[dry-run] 未发送任何指令。")
        return

    last_action = np.zeros(12, dtype=np.float32)
    last_cmd = q0.copy()
    max_step = args.max_joint_rate * dt
    n_ramp = int(args.ramp / dt)
    n_run = int(args.seconds / dt)
    trace = {k: [] for k in ("t", "q", "qd", "gyro", "grav", "action", "p_des")}

    send = Sender(low)
    pacer = Pacer(dt)
    try:
        # ---- 软启动：从实测姿态平滑到站姿 ----
        # 平滑步 3a^2-2a^3 而非线性：线性斜坡在起止两端目标角速度是阶跃变化，
        # 对电机是两次冲击；平滑步两端导数为零。
        pacer.start()
        for i in range(n_ramp):
            a = (i + 1) / n_ramp
            a = a * a * (3 - 2 * a)
            p = (1 - a) * q0 + a * default_q
            send(fill(sdk, sdk_from_isaac, p, kp_stance, kd))
            last_cmd = p
            pacer.wait()
        pacer.report("软启动")
        send.report("软启动")
        print("[OK] 软启动完成，进入策略控制")

        # ---- 策略循环 ----
        pacer = Pacer(dt)
        pacer.start()
        t_start = time.perf_counter()
        for _ in range(n_run):
            st = low.getMotorState()
            q = isaac_from_sdk(st.q_abad, st.q_hip, st.q_knee)
            qd = isaac_from_sdk(st.qd_abad, st.qd_hip, st.qd_knee)
            gyro = np.asarray(low.getBodyGyro(), dtype=np.float64)
            grav = projected_gravity(low.getQuaternion())

            # 输入侧自检。观测拼错、IMU 掉数、编码器读到 NaN —— 这几种错网络照样
            # 吃得下、照样输出 12 个数、机器人照样动，只是动得不对。必须在进网络前拦。
            gn = np.linalg.norm(grav)
            if abs(gn - 1.0) > GRAVITY_NORM_TOL:
                raise RuntimeError(f"重力投影模长 {gn:.4f} 偏离 1 —— IMU 数据异常")
            for nm, v in (("q", q), ("qd", qd), ("gyro", gyro), ("grav", grav)):
                if not np.all(np.isfinite(v)):
                    raise RuntimeError(f"观测输入 {nm} 含 NaN/inf: {v}")

            obs = np.concatenate([
                gyro,                                    # [0:3]   机体角速度 rad/s
                grav,                                    # [3:6]   重力投影
                [args.vx, args.vy, args.wz],             # [6:9]   速度指令
                q - default_q,                           # [9:21]  关节角（相对默认角）
                qd,                                      # [21:33] 关节角速度
                last_action,                             # [33:45] 上一步网络原始输出
            ]).astype(np.float32)

            action = sess.run(None, {in_name: obs[None, :]})[0].reshape(-1)
            if not np.all(np.isfinite(action)):
                raise RuntimeError("策略输出含 NaN/inf")
            last_action = action.astype(np.float32)

            p = default_q + scale * action
            p = np.clip(p, CMD_LOWER, CMD_UPPER)                        # 硬限位（URDF ∩ 固件）
            p = last_cmd + np.clip(p - last_cmd, -max_step, max_step)   # 速率限制
            last_cmd = p
            send(fill(sdk, sdk_from_isaac, p, kp, kd))

            for k, v in (("t", time.perf_counter() - t_start), ("q", q), ("qd", qd),
                         ("gyro", gyro), ("grav", grav), ("action", action), ("p_des", p)):
                trace[k].append(np.copy(v))
            pacer.wait()

    finally:
        # ---- 阻尼停机：kp 归零、kd 加大，缓缓趴下（照官方 demo 的 stop()）----
        # 不做这一步的话，Ctrl+C 之后狗会保持最后一帧目标角硬撑着。
        print("\n[..] 阻尼停机 …")
        # last_cmd 一定在 CMD_* 范围内（上面每帧都夹过），所以这里不会被固件拒收。
        # 但仍然统计：这是最后一道路径，它静默失效的话，狗会硬撑到看门狗超时。
        # 注意这里不复用 Sender —— Sender 连续失败会抛错，而停机路径必须跑完。
        stop_fail = 0
        for _ in range(150):
            if low.sendMotorCmd(fill(sdk, sdk_from_isaac, last_cmd, 0.0, 4.0)) < 0:
                stop_fail += 1
            time.sleep(0.02)
        if stop_fail:
            print(f"[警告] 阻尼停机有 {stop_fail}/150 帧被拒收 —— 急停路径不可靠，先别上地面")
        else:
            print("[OK] 已停机（150 帧全部下发成功）")

        pacer.report("策略循环")
        send.report("策略循环")
        if trace["t"]:
            summarize(trace, dt, max_step)
            if args.trace:
                # ★ 把判读要用的参数一并存进去。分析端不许再自己写一份 —— 本次
                # max_joint_rate 改成 2.0 后，分析脚本里写死的 4.0 让它报 0.00%，
                # 而主脚本报 5.32%，同一份数据两个答案。数跟着数据走就不会有这种事。
                meta = dict(dt=dt, scale=scale, max_step=max_step,
                            max_joint_rate=args.max_joint_rate,
                            kp=kp, kd=kd, kp_stance=kp_stance,
                            abad_sign=args.abad_sign,
                            cmd_lower=CMD_LOWER, cmd_upper=CMD_UPPER,
                            urdf_lower=URDF_LOWER, urdf_upper=URDF_UPPER,
                            default_q=default_q,
                            command=np.array([args.vx, args.vy, args.wz]))
                np.savez_compressed(args.trace,
                                    **{k: np.array(v) for k, v in trace.items()}, **meta)
                print(f"[OK] 逐帧留痕已存 {args.trace}（{len(trace['t'])} 帧）")


def summarize(trace, dt, max_step):
    """把「抖不抖」变成几个可对比的数，而不是留给眼睛判断。

    吊着 + 零指令时策略的期望行为是保持站姿不动 —— 45 维观测里没有接触也没有高度，
    策略看不见自己被吊着，它以为自己站在地上、指令为零。所以此时任何抖动都是真抖动。
    """
    p = np.array(trace["p_des"])
    a = np.array(trace["action"])
    qd = np.array(trace["qd"])
    dp = np.abs(np.diff(p, axis=0))
    print("\n---- 本次运行的抖动指标 ----")
    print(f"  目标角逐帧变化 |Δp_des|   均值 {dp.mean():.4f}  p99 {np.percentile(dp,99):.4f}  "
          f"最大 {dp.max():.4f} rad")
    print(f"    （速率限幅上限 {max_step:.4f} rad/帧，"
          f"触顶 {100*np.mean(dp >= max_step*0.999):.2f}% 的关节-帧）")
    print(f"  网络输出 action           |均值| {np.abs(a.mean(axis=0)).max():.4f}  "
          f"逐关节标准差最大 {a.std(axis=0).max():.4f}")
    print(f"  实测关节角速度 |qd|       均值 {np.abs(qd).mean():.4f}  "
          f"p99 {np.percentile(np.abs(qd),99):.4f}  最大 {np.abs(qd).max():.4f} rad/s")
    print("  判读：吊着零指令时上面三行都应该很小。速率限幅频繁触顶 = 策略在硬拽，"
          "那就是红线要拦的抖动。")


if __name__ == "__main__":
    main()
