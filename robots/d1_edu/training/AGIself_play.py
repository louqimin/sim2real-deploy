#!/usr/bin/env python3
# AGIself · 回放 / 导出入口（薄壳）
"""调用 Isaac Lab 自带的 rsl_rl play.py，先把我们的 gym 环境注册进去。

play.py 在加载 checkpoint 之后会把策略导出成 TorchScript 与 ONNX 两份文件，
落在 <日志目录>/exported/ 下。那两份文件就是交付条「可脱离 Isaac Lab 加载」的载体 ——
它们不含任何 isaaclab 符号，部署环境里只需要 torch（或 onnxruntime）。

用法：
    python AGIself_play.py --task AGIself-D1-Edu-Flat-Play-v0 --headless --num_envs 16
"""

import os
import runpy
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

import training  # noqa: E402,F401  仅执行 gym.register

_CANDIDATES = [
    os.path.expanduser("~/IsaacLab/scripts/reinforcement_learning/rsl_rl/play.py"),
    os.path.expanduser("~/IsaacLab/source/standalone/workflows/rsl_rl/play.py"),
]

_STOCK_PLAY = next((p for p in _CANDIDATES if os.path.isfile(p)), None)
if _STOCK_PLAY is None:
    raise SystemExit(
        "找不到 Isaac Lab 自带的 rsl_rl play.py，试过：\n  " + "\n  ".join(_CANDIDATES) + "\n"
        "请用 find ~/IsaacLab -path '*rsl_rl/play.py' 定位后，把路径加进 _CANDIDATES。"
    )

# 与 AGIself_train.py 同理：play.py 也 import 了同目录的 cli_args，
# runpy.run_path 不会自动把脚本所在目录加进搜索路径
sys.path.insert(0, os.path.dirname(_STOCK_PLAY))

print("[AGIself] 已注册环境：AGIself-D1-Edu-Flat-v0 / AGIself-D1-Edu-Flat-Play-v0", flush=True)
print(f"[AGIself] 转交官方脚本：{_STOCK_PLAY}", flush=True)

runpy.run_path(_STOCK_PLAY, run_name="__main__")
