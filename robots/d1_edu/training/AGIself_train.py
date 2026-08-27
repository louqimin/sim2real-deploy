#!/usr/bin/env python3
# AGIself · 训练入口（薄壳）
"""调用 Isaac Lab 自带的 rsl_rl 训练脚本，但先把我们的 gym 环境注册进去。

为什么需要这层壳：Isaac Lab 的 scripts/reinforcement_learning/rsl_rl/train.py 只会
import isaaclab_tasks，因此它认识的 --task 只有官方那些。我们的环境在仓库外，
必须有人在它跑起来之前先执行一次 gym.register。这个壳干的就是这一件事。

用法与官方 train.py 完全一致，参数原样透传：
    python AGIself_train.py --task AGIself-D1-Edu-Flat-v0 --headless --num_envs 4096
"""

import os
import runpy
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))

# 把 robots/d1_edu 放进搜索路径，使 training 成为可 import 的顶层包
sys.path.insert(0, os.path.dirname(_HERE))

import training  # noqa: E402,F401  仅执行 gym.register，不拉起 pxr

##
# 定位官方训练脚本
##

_CANDIDATES = [
    os.path.expanduser("~/IsaacLab/scripts/reinforcement_learning/rsl_rl/train.py"),
    os.path.expanduser("~/IsaacLab/source/standalone/workflows/rsl_rl/train.py"),
]

_STOCK_TRAIN = next((p for p in _CANDIDATES if os.path.isfile(p)), None)
if _STOCK_TRAIN is None:
    raise SystemExit(
        "找不到 Isaac Lab 自带的 rsl_rl train.py，试过：\n  " + "\n  ".join(_CANDIDATES) + "\n"
        "请用 find ~/IsaacLab -path '*rsl_rl/train.py' 定位后，把路径加进 _CANDIDATES。"
    )

# 官方 train.py 里有 `import cli_args`，那是它同目录下的兄弟模块。
# 直接 python train.py 时解释器会自动把脚本所在目录放进 sys.path，
# 但 runpy.run_path 不会，所以这里得手动补上，否则必然 ModuleNotFoundError。
sys.path.insert(0, os.path.dirname(_STOCK_TRAIN))

print("[AGIself] 已注册环境：AGIself-D1-Edu-Flat-v0 / AGIself-D1-Edu-Flat-Play-v0", flush=True)
print(f"[AGIself] 转交官方脚本：{_STOCK_TRAIN}", flush=True)

runpy.run_path(_STOCK_TRAIN, run_name="__main__")
