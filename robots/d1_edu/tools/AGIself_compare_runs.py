#!/usr/bin/env python3
# AGIself · 训练轮次并排对比
"""把若干个 rsl_rl 训练目录的标量指标并排打出来，用于单变量对比。

动因：终端 scrollback 会滚掉，8/27 基线那两轮的输出早已不在。
数据本身一直躺在各目录的 events.out.tfevents.* 里，本脚本只是把它读回来。

用法：
    python AGIself_compare_runs.py <基线目录> <对比目录> [更多目录...]

第一个目录当基线，后面每个都与它逐项作差。
只读，不写任何文件。

依赖 tensorboard（env_all_pro 里训练本身就在用它写日志，必有）。
"""

from __future__ import annotations

import os
import sys

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except ImportError:
    raise SystemExit(
        "导入 tensorboard 失败。这个脚本要在训练环境里跑（env_all_pro），\n"
        "不是部署环境（d1-deploy-py310，那边刻意不装训练侧依赖）。"
    )


# 关心的指标排在前面，其余的按字母序跟在后面。
# 顺序即阅读顺序：先看终止率（策略活没活下来），再看跟踪（任务完成得如何），
# 最后看代价项（为了完成任务付出了什么）。
PRIORITY = [
    "Episode_Termination/base_contact",
    "Episode_Termination/time_out",
    "Metrics/base_velocity/error_vel_xy",
    "Metrics/base_velocity/error_vel_yaw",
    "Episode_Reward/track_lin_vel_xy_exp",
    "Episode_Reward/track_ang_vel_z_exp",
    "Episode_Reward/feet_air_time",
    "Episode_Reward/undesired_contacts",
    "Episode_Reward/flat_orientation_l2",
    "Episode_Reward/dof_torques_l2",
    "Episode_Reward/dof_acc_l2",
    "Episode_Reward/dof_pos_limits",
    "Episode_Reward/action_rate_l2",
    "Episode_Reward/lin_vel_z_l2",
    "Episode_Reward/ang_vel_xy_l2",
    "Train/mean_reward",
    "Train/mean_episode_length",
]


# 取最后多少个记录点求平均。
#
# 为什么不能取单个末值（2026-08-29 踩过）：Episode_Reward/* 只在「有回合结束」的那一轮
# 才更新，记的是刚结算的那一批回合的平均。回合约 993 步、每轮只推进 24 步，
# 所以任一轮里只有约 2.4% 的环境在结算 —— 单个末值是从一个很小的随机子集里抽的。
# 那次两轮对比中，基线的末值恰好有代表性、新轮的没有，导致同一张表里
# 「各项之和 × 20」与 Train/mean_reward 在一轮里吻合、另一轮差 7 倍。
TAIL = 20


def read_run(path: str, tail: int = TAIL) -> tuple[dict[str, float], dict[str, float], int]:
    """读一个训练目录。

    返回三样：{标量名: 末 tail 个点的均值}、{标量名: 单个末值}、最后的 step。
    均值用于判读，末值只用来提示波动有多大。

    size_guidance={"scalars": 0} 的 0 是「不限量」，不是「不要」——
    默认值会为省内存对时间轴做抽稀，而我们要的恰好是完整的尾段。
    """
    if not os.path.isdir(path):
        raise SystemExit(f"不是目录：{path}")

    acc = EventAccumulator(path, size_guidance={"scalars": 0})
    acc.Reload()

    tags = acc.Tags().get("scalars", [])
    if not tags:
        raise SystemExit(f"这个目录里没有标量数据，确认是训练目录而非别的：{path}")

    avg: dict[str, float] = {}
    last: dict[str, float] = {}
    last_step = 0
    for t in tags:
        events = acc.Scalars(t)
        if not events:
            continue
        vals = [float(e.value) for e in events[-tail:]]
        avg[t] = sum(vals) / len(vals)
        last[t] = float(events[-1].value)
        last_step = max(last_step, int(events[-1].step))
    return avg, last, last_step


def self_check(name: str, vals: dict[str, float], episode_length_s: float = 20.0) -> None:
    """自检：各奖励项之和 × 回合时长，应当等于 Train/mean_reward。

    这两个量由不同代码路径写出，对得上才说明这一列可以拿来判读。
    2026-08-29 正是这条对账拦下了一次错误结论 —— 当时新轮差了 7 倍，
    而表面上每个数字都长得像正常数字。
    """
    terms = {k: v for k, v in vals.items() if k.startswith("Episode_Reward/")}
    mean_reward = vals.get("Train/mean_reward")
    if not terms or mean_reward is None:
        print(f"  {name}：缺少 Episode_Reward/* 或 Train/mean_reward，跳过自检")
        return

    predicted = sum(terms.values()) * episode_length_s
    if abs(mean_reward) < 1e-9:
        rel = float("inf")
    else:
        rel = abs(predicted - mean_reward) / abs(mean_reward)

    verdict = "✓ 一致" if rel < 0.15 else "✗ 不一致，这一列不可用于判读"
    print(f"  {name}：各项之和×{episode_length_s:g} = {predicted:8.2f}   "
          f"mean_reward = {mean_reward:8.2f}   偏差 {rel * 100:5.1f}%   {verdict}")


def fmt(v) -> str:
    """定宽格式化。None 表示这个指标在该轮里根本不存在。"""
    if v is None:
        return "      --"
    return f"{v:>8.4f}"


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2

    paths = sys.argv[1:]
    runs = []
    for p in paths:
        avg, last, step = read_run(p)
        runs.append((os.path.basename(os.path.normpath(p)), avg, step, last))

    # ---- 先报 step，不一致就意味着两轮长度不同、逐项对比无意义 ----
    print("=" * 78)
    for name, _, step, _ in runs:
        print(f"  {name}    最后 step = {step}")
    steps = {step for _, _, step, _ in runs}
    if len(steps) > 1:
        print("\n  ⚠ 各轮最后 step 不一致 —— 训练长度不同，下面的逐项对比不具可比性。")

    # ---- 自检：这一列的数字自洽吗 ----
    print(f"\n  自检（取末 {TAIL} 个记录点的均值）：")
    for name, avg, _, _ in runs:
        self_check(name, avg)
    print("=" * 78)
    print()

    # ---- 收集所有出现过的标量名：优先项按 PRIORITY 排，其余按字母序 ----
    seen = set()
    for _, vals, _, _ in runs:
        seen.update(vals.keys())
    ordered = [t for t in PRIORITY if t in seen] + sorted(seen - set(PRIORITY))

    base_vals = runs[0][1]

    header = f"{'指标':<44}" + "".join(f"{n[-8:]:>10}" for n, _, _, _ in runs)
    if len(runs) == 2:
        header += f"{'相对基线':>12}" + f"{'末值抖动':>12}"
    print(header)
    print("-" * len(header))

    for tag in ordered:
        row = f"{tag:<44}"
        for _, vals, _, _ in runs:
            row += fmt(vals.get(tag)).rjust(10)

        # 只在恰好两轮时给差值列，三轮以上并排看原值更清楚
        if len(runs) == 2:
            b = base_vals.get(tag)
            c = runs[1][1].get(tag)
            if b is None or c is None:
                row += f"{'--':>12}"
            elif abs(b) < 1e-9:
                # 基线为 0 时百分比没有意义，改报绝对差
                row += f"{c - b:>+12.4f}"
            else:
                row += f"{(c - b) / abs(b) * 100:>+11.1f}%"

            # 末值相对均值偏了多少 —— 这一项的逐轮波动有多大。
            # 偏得厉害说明它本来就是稀疏更新的噪声量，别在单轮数字上做文章。
            c_last = runs[1][3].get(tag)
            if c is None or c_last is None or abs(c) < 1e-9:
                row += f"{'--':>12}"
            else:
                row += f"{(c_last - c) / abs(c) * 100:>+11.0f}%"
        print(row)

    print()
    print("读法提醒：")
    print("  · 数字取的是末 %d 个记录点的均值，不是单个末值" % TAIL)
    print("  · 「末值抖动」= 单个末值偏离该均值多少。偏得大的项是稀疏更新的噪声量，")
    print("    单轮数字不可信 —— 2026-08-29 就是在这上面差点判错")
    print("  · Episode_Reward/* 是「权重乘过之后」的贡献，负值项本来就该是负的")
    print("  · 上面自检打 ✗ 的那一列，整列都不能用于判读")
    print("  · 就算全部一致也不代表步态正确 —— 2026-08-27 那轮跪着走时终止率 0.0000、")
    print("    跟踪奖励拿到理论上限 96%，只有录像能发现。数字用来定位问题，不用来放行")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
