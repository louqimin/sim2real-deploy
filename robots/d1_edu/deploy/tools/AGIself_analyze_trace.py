"""读 AGIself_run_policy.py 存的留痕 npz，把「抖动」拆到每个关节。

用法：
    python AGIself_analyze_trace.py /tmp/hang_run.npz

关注三件事：
  1. 哪个关节的 action 均值大 —— 均值大 = 策略持续想把它推到离默认角很远的地方，
     不是噪声。若换算出的 p_des 越过硬限位，那个关节就是每一帧都在被夹，
     等于策略和限幅在互相顶。
  2. 哪个关节在触速率限幅 —— 那是策略想让它动得比 4 rad/s 还快。
  3. 抖动是什么性质 —— 用 Δp_des 的变号率区分，注意基线是 0.5 不是 0：
       ~0.0  平滑的单向移动（策略在慢慢调整姿态）
       ~0.5  白噪声式抖动 —— 随机游走的基线，相邻两步同向反向各一半
       ~1.0  几乎每帧反向 —— 奈奎斯特频率上的干净振荡，这才是真颤振
     0.55~0.65 只是「略偏向反向的宽带噪声」，和 1.0 是两回事，别混。
     光看变号率不够，还要看幅度：同一个变号率下，0.5° 的抖和 5° 的抖不是一个问题。
"""

import sys

import numpy as np

JOINT_NAMES = ([f"ABAD_{s}" for s in ("FL", "FR", "RL", "RR")]
               + [f"HIP_{s}" for s in ("FL", "FR", "RL", "RR")]
               + [f"KNEE_{s}" for s in ("FL", "FR", "RL", "RR")])

# ⚠ 这些参数一律从 npz 里读，不在这里写死。
# 教训：本文件曾把 MAX_STEP 硬编码成 4.0*0.02=0.08，而那一跑用的是 --max-joint-rate 2.0
# （真实上限 0.04）⇒ 本脚本报「触速率 0.00%」，主脚本报「5.32%」，同一份数据两个答案。
# 兜底值只在读老留痕（没存参数那版）时用，且会大声警告。
FALLBACK = dict(scale=0.25, dt=0.02, max_step=0.08,
                default_q=np.array([0.0] * 4 + [0.8] * 4 + [-1.5] * 4),
                cmd_lower=np.array([-0.4887] * 4 + [-1.1519] * 4 + [-2.723] * 4),
                cmd_upper=np.array([+0.4887] * 4 + [+2.967] * 4 + [-0.65] * 4))


def main(path):
    d = np.load(path)
    a, p, q, qd = d["action"], d["p_des"], d["q"], d["qd"]

    missing = [k for k in FALLBACK if k not in d]
    if missing:
        print(f"⚠ 留痕里没有 {missing} —— 是旧版主脚本存的。以下用兜底值，"
              f"若当时改过 --max-joint-rate，「触速率%」那一列会是错的。\n")
    par = {k: (d[k] if k in d else FALLBACK[k]) for k in FALLBACK}
    SCALE = float(par["scale"]); DT = float(par["dt"])
    MAX_STEP = float(par["max_step"])
    DEFAULT_Q = np.asarray(par["default_q"])
    CMD_LOWER = np.asarray(par["cmd_lower"]); CMD_UPPER = np.asarray(par["cmd_upper"])

    n = len(a)
    dp = np.diff(p, axis=0)
    print(f"{path}   {n} 帧 / {n*DT:.1f} 秒   "
          f"速率上限 {MAX_STEP/DT:.1f} rad/s（{MAX_STEP:.4f} rad/帧）\n")

    hdr = (f"{'关节':10}{'act均值':>9}{'p_des均值':>10}{'抖动°':>8}"
           f"{'夹限位%':>9}{'触速率%':>9}{'Δ变号率':>9}{'|qd|p99':>9}")
    print(hdr)
    print("-" * 78)

    rows = []
    for j in range(12):
        p_raw = DEFAULT_Q[j] + SCALE * a[:, j]
        clamped = 100 * np.mean((p_raw < CMD_LOWER[j] - 1e-9) | (p_raw > CMD_UPPER[j] + 1e-9))
        rate = 100 * np.mean(np.abs(dp[:, j]) >= MAX_STEP * 0.999)
        s = np.sign(dp[:, j])
        s = s[s != 0]
        flip = np.mean(s[1:] != s[:-1]) if s.size > 1 else 0.0
        rows.append((j, clamped, rate, flip))
        # 抖动幅度换成度：action 的标准差经 scale 变成目标角的标准差，再转度。
        # 这是「策略每帧把目标角晃多少」的直接物理量，比 action 标准差好判断。
        dither_deg = np.degrees(a[:, j].std() * SCALE)
        print(f"{JOINT_NAMES[j]:10}{a[:,j].mean():+9.3f}{p[:,j].mean():+10.3f}"
              f"{dither_deg:8.2f}{clamped:9.1f}{rate:9.2f}{flip:9.2f}"
              f"{np.percentile(np.abs(qd[:,j]),99):9.2f}")

    print()
    worst_clamp = max(rows, key=lambda r: r[1])
    worst_rate = max(rows, key=lambda r: r[2])
    print(f"夹限位最多 : {JOINT_NAMES[worst_clamp[0]]}  {worst_clamp[1]:.1f}% 的帧")
    print(f"触速率最多 : {JOINT_NAMES[worst_rate[0]]}  {worst_rate[2]:.2f}% 的帧")
    # 阈值 0.75：明显高于随机基线 0.5、朝着 1.0 那一端走，才叫颤振。
    hi_flip = [JOINT_NAMES[r[0]] for r in rows if r[3] > 0.75]
    print(f"疑似颤振   : {hi_flip if hi_flip else '无（变号率都在随机基线 0.5 附近，属宽带噪声不是振荡）'}")
    dither = np.degrees(a.std(axis=0) * SCALE)
    k = int(np.argmax(dither))
    print(f"抖动最大   : {JOINT_NAMES[k]}  目标角标准差 {dither[k]:.2f}°"
          f"（十二关节中位数 {np.median(dither):.2f}°）")

    print("\n---- 姿态与跟踪 ----")
    print("默认角与策略稳态目标角之差（正=策略想推过头，D20 那件事）：")
    for lo, hi, nm in ((0, 4, "ABAD"), (4, 8, "HIP "), (8, 12, "KNEE")):
        d_ = p[:, lo:hi].mean(axis=0) - DEFAULT_Q[lo:hi]
        print(f"  {nm} " + " ".join(f"{v:+7.3f}" for v in d_) + "   (FL FR RL RR)")
    print("目标角与实测角之差（PD 稳态误差，吊着不承重应该很小）：")
    err = (p - q).mean(axis=0)
    for lo, hi, nm in ((0, 4, "ABAD"), (4, 8, "HIP "), (8, 12, "KNEE")):
        print(f"  {nm} " + " ".join(f"{v:+7.3f}" for v in err[lo:hi]))

    if "gyro" in d:
        report_segments(a, d["gyro"], DT, SCALE)


def segment_by_disturbance(gyro, dt, dilate_s=0.3):
    """按机身角速度把帧切成「被外力推的」与「安静的」两段。

    动机：人推狗时策略会做大幅调节，那是【响应】不是【抖动】，混在一起统计会
    把抖动指标顶高，让人误判。而两者有一个干净的区分量 —— 推会让机身转起来，
    策略自发的小幅关节抖动不会。所以用陀螺仪幅度切。

    阈值用中位数 + 3×MAD 而不是均值 + 3σ：如果推的次数多，均值和标准差本身
    就被推的那些帧抬高了，阈值会跟着飘、反而漏掉扰动。中位数和 MAD 对少数
    极端值不敏感，这正是这里需要的性质。

    切出来的点再向两边各扩 dilate_s：一次推的影响会延续到机身停下之后，
    只标记角速度超阈的那几帧会把响应的尾巴留在「安静」段里。
    """
    mag = np.linalg.norm(gyro, axis=1)
    med = np.median(mag)
    mad = np.median(np.abs(mag - med))
    thresh = med + 3.0 * 1.4826 * mad
    hot = mag > thresh
    # 去抖：孤立的单帧尖峰不算扰动。安静段的噪声偶尔会越过阈值，若不去抖，
    # 每个尖峰都会被计成一次「推」——离线自测里 3 次推被数成 5 段就是这个。
    # 要求 3 帧窗口内至少 2 帧超阈，既滤掉单帧尖峰，又容忍一次掉帧。
    hot = np.convolve(hot.astype(float), np.ones(3), mode="same") >= 2
    k = max(1, int(dilate_s / dt))
    disturbed = np.convolve(hot.astype(float), np.ones(2 * k + 1), mode="same") > 0
    # 数一下有几段连续的扰动，对应「推了几下」
    edges = np.diff(np.concatenate([[0], disturbed.astype(int), [0]]))
    n_events = int((edges == 1).sum())
    return disturbed, thresh, med, n_events, mag


def report_segments(a, gyro, dt, scale):
    disturbed, thresh, med, n_events, mag = segment_by_disturbance(gyro, dt)
    quiet = ~disturbed
    print("\n---- 按外力扰动切段 ----")
    print(f"  机身角速度 中位 {med:.3f} rad/s，阈值 {thresh:.3f} rad/s，"
          f"峰值 {mag.max():.3f} rad/s")
    print(f"  识别出 {n_events} 段扰动，占 {100*disturbed.mean():.1f}% 的帧"
          f"（{disturbed.sum()*dt:.1f} 秒 / 共 {len(mag)*dt:.1f} 秒）")
    if quiet.sum() < 20 or disturbed.sum() < 20:
        print("  某一段样本太少，不分开统计。")
        return
    for name, mask in (("安静段（真·基线抖动）", quiet), ("扰动段（推它时的响应）", disturbed)):
        dither = np.degrees(a[mask].std(axis=0) * scale)
        print(f"  {name}: 目标角抖动 中位 {np.median(dither):5.2f}°  "
              f"最大 {dither.max():5.2f}°（{JOINT_NAMES[int(np.argmax(dither))]}）")
    print("  判读：安静段那个数才是能和空载基线比的抖动；扰动段大是好事，"
          "说明策略在响应外力而不是无动于衷。")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/hang_run.npz")
