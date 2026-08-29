"""
ABAD 探针 —— 一次实测同时收 D26(下标→腿) 与 D14(符号方向)。

做法：挂起状态下，先按官方 demo 的两段式软启动把狗摆成站姿，
      再【只把一个下标】的 ABAD 推到 +0.35，其余三条保持 0。

判读：
  1. 动的是哪条腿  -> 该下标对应的物理腿  (D26)
  2. 那只脚往狗自己的【左】摆 -> 编码器符号与 URDF 一致 (D14 收)
     往【右】摆                 -> 全局反向，部署时 ABAD 要整体取负

约定：左右一律以【狗自己朝前】为准，不是站在狗对面看的左右。

前置：狗必须吊起来，四脚离地。人手放在 Ctrl+C 上。

用法：
    python AGIself_probe_abad.py            # 默认探下标 0
    python AGIself_probe_abad.py 1          # 探下标 1
"""

import sys
import time

sys.path.insert(0, "/home/lqm/self-AGI/agibot_D1_Edu-Ultra/lib/zsl-1/x86_64")
import mc_sdk_zsl_1_py

# ---- 参数 ----------------------------------------------------------------
LOCAL_IP, LOCAL_PORT, DOG_IP = "192.168.168.100", 43988, "192.168.168.168"

KP, KD = 80.0, 1.0            # 与官方 demo 一致，不引入新变量
DT = 0.002                    # 发包间隔，与官方 demo 一致

CROUCH = (0.0, 1.4, -2.4)     # demo 第一段目标 (abad, hip, knee)
STAND = (0.0, 0.8, -1.5)      # demo 第二段目标
PROBE_ANGLE = 0.35            # 探针幅度，硬限位 ±0.4887，留 0.14 余量
ABAD_LIMIT = 0.4887

T_CROUCH, T_STAND = 2.0, 2.0  # 两段软启动时长
T_PROBE, T_HOLD = 2.0, 6.0    # 探针斜坡 / 保持观察
T_BACK = 2.0                  # 探针归零


def smooth(a):
    """平滑步 3a^2-2a^3：两端导数为零，避免斜坡起止对电机的两次冲击。"""
    a = 0.0 if a < 0.0 else (1.0 if a > 1.0 else a)
    return a * a * (3.0 - 2.0 * a)


class Probe:
    def __init__(self, index):
        assert 0 <= index <= 3, "下标必须是 0..3"
        self.index = index
        self.dog = mc_sdk_zsl_1_py.LowLevel()
        self.dog.initRobot(LOCAL_IP, LOCAL_PORT, DOG_IP)
        self.periods = []

    def wait_data(self, timeout=10.0):
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            if self.dog.haveMotorData():
                return True
            time.sleep(0.05)
        return False

    def send(self, abad, hip, knee):
        """abad/hip/knee 各是长度 4 的列表，下标即 SDK 下标。"""
        cmd = mc_sdk_zsl_1_py.MotorCommand()
        for i in range(4):
            a = abad[i]
            if a > ABAD_LIMIT:
                a = ABAD_LIMIT
            elif a < -ABAD_LIMIT:
                a = -ABAD_LIMIT
            cmd.q_des_abad[i] = a
            cmd.q_des_hip[i] = hip[i]
            cmd.q_des_knee[i] = knee[i]
            cmd.kp_abad[i] = KP
            cmd.kp_hip[i] = KP
            cmd.kp_knee[i] = KP
            cmd.kd_abad[i] = KD
            cmd.kd_hip[i] = KD
            cmd.kd_knee[i] = KD
        if self.dog.sendMotorCmd(cmd) < 0:
            print("send cmd error")

    def ramp(self, duration, start, target, label):
        """按【墙钟】做斜坡，不按循环计数——循环真实周期未必是 DT。"""
        print(f"  {label} ...")
        t0 = time.monotonic()
        last = t0
        while True:
            now = time.monotonic()
            a = (now - t0) / duration
            if a >= 1.0:
                a = 1.0
            r = smooth(a)
            abad = [(1 - r) * start[0][i] + r * target[0][i] for i in range(4)]
            hip = [(1 - r) * start[1][i] + r * target[1][i] for i in range(4)]
            knee = [(1 - r) * start[2][i] + r * target[2][i] for i in range(4)]
            self.send(abad, hip, knee)
            self.periods.append(now - last)
            last = now
            if a >= 1.0:
                return
            time.sleep(DT)

    def hold(self, duration, pose, label):
        print(f"  {label} ...")
        t0 = time.monotonic()
        while time.monotonic() - t0 < duration:
            self.send(*pose)
            time.sleep(DT)

    def run(self):
        if not self.wait_data():
            print("没有收到电机数据，检查网络与运控是否起来")
            return

        s = self.dog.getMotorState()
        q0 = ([s.q_abad[i] for i in range(4)],
              [s.q_hip[i] for i in range(4)],
              [s.q_knee[i] for i in range(4)])
        print("起始实测角:")
        print("  ABAD", [round(v, 3) for v in q0[0]])
        print("  HIP ", [round(v, 3) for v in q0[1]])
        print("  KNEE", [round(v, 3) for v in q0[2]])

        crouch = tuple([CROUCH[k]] * 4 for k in range(3))
        stand = tuple([STAND[k]] * 4 for k in range(3))

        self.ramp(T_CROUCH, q0, crouch, "第一段 软启动 -> 蹲姿")
        self.ramp(T_STAND, crouch, stand, "第二段 软启动 -> 站姿")
        self.hold(1.0, stand, "站姿稳定")

        probe = ([STAND[0]] * 4, [STAND[1]] * 4, [STAND[2]] * 4)
        probe[0][self.index] = PROBE_ANGLE

        print()
        print(f"===== 现在只推下标 {self.index} 的 ABAD 到 +{PROBE_ANGLE} =====")
        print("请看：动的是哪条腿？那只脚往【狗自己的】左边还是右边摆？")
        print()
        self.ramp(T_PROBE, stand, probe, "探针斜坡")
        self.hold(T_HOLD, probe, f"保持 {T_HOLD} 秒，观察")

        s = self.dog.getMotorState()
        print()
        print("保持时读回 ABAD:", [round(s.q_abad[i], 3) for i in range(4)])
        print(f"  下标 {self.index} 应接近 +{PROBE_ANGLE}，其余三个应接近 0.0")

        self.ramp(T_BACK, probe, stand, "探针归零")
        self.hold(0.5, stand, "归零稳定")

    def report(self):
        p = sorted(self.periods[1:])
        if not p:
            return
        n = len(p)
        mean = sum(p) / n
        print()
        print("---- 循环周期实测 ----")
        print(f"  样本 {n}   均值 {mean * 1000:.3f} ms   等效 {1 / mean:.1f} Hz")
        print(f"  p50 {p[n // 2] * 1000:.3f} ms   p99 {p[int(n * 0.99)] * 1000:.3f} ms")
        print(f"  （官方 demo 假定每圈恰好 {DT * 1000:.0f} ms，实测差多少直接看均值）")

    def stop(self):
        """照搬官方 demo 的阻尼停机：kp=0，只留 kd，连发 3 秒。

        ⚠ q_des_knee 必须显式填一个合法值，不能靠 MotorCommand() 的默认 0.0。
        固件对 KNEE 指令做范围校验（实测报 "invalid knee cmd, expect -2.9~-0.65 rad"），
        0.0 在范围外 ⇒ 整帧被拒、sendMotorCmd 返回负数、阻尼停机一帧都发不出去。
        官方 demo 的 stop() 里写 q_des_knee = -1.5 正是为这个。
        kp=0 时 q_des 不参与力矩计算，填什么都不影响行为，只是要过校验。
        """
        print("阻尼停机中，约 3 秒 ...")
        cmd = mc_sdk_zsl_1_py.MotorCommand()
        for i in range(4):
            cmd.q_des_abad[i] = 0.0
            cmd.q_des_hip[i] = 0.0
            cmd.q_des_knee[i] = -1.5
            cmd.kd_abad[i] = 4.0
            cmd.kd_hip[i] = 4.0
            cmd.kd_knee[i] = 4.0
        t0 = time.monotonic()
        fails = 0
        while time.monotonic() - t0 < 3.0:
            if self.dog.sendMotorCmd(cmd) < 0:
                fails += 1
            time.sleep(DT)
        print(f"已停。（发送失败 {fails} 帧，应为 0）")


if __name__ == "__main__":
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    p = Probe(idx)
    try:
        p.run()
    except KeyboardInterrupt:
        print("\n收到 Ctrl+C")
    finally:
        p.report()
        p.stop()
