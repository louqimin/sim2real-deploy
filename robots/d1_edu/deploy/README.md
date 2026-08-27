# D1 edu · 部署侧运行时

把训练出来的 PPO locomotion policy 跑在真实机器人上所需的最小闭环。

## 这个目录的硬约束

**运行时不 import isaaclab，也不 import torch。** 依赖只有 `numpy` 与 `onnxruntime`，
写在 `pyproject.toml` 里——写错了连装都装不上。这是交付条 D2「可脱离 Isaac Lab 加载」
的结构性保证，不是风格偏好。

`policy.pt` 留在 bundle 里，但只在训练机上用作 ONNX 的对账标准答案，板子上永远不加载。

## 装

```bash
conda create -n d1_deploy python=3.10 -y
conda activate d1_deploy
cd robots/d1_edu/deploy
pip install -e .
```

## 打包与自检

```bash
# 1. 从某一次训练产物打 bundle（--run-dir 必填，刻意不支持「取最新」）
python tools/AGIself_make_bundle.py \
    --run-dir ~/sim2real-deploy/logs/rsl_rl/agiself_d1_edu_flat/2026-08-27_16-27-07

# 2. 干净环境自检：45 维进、12 维出
python -m agiself_d1_deploy.selftest
```

## 模块

| 文件 | 作用 |
|---|---|
| `contract.py` | 读 bundle 里的契约 JSON，校验 sha256，暴露冻结常量。**一个数都不手抄** |
| `observation.py` | 45 维观测拼装（D05） |
| `policy.py` | ONNX 推理封装 |
| `runner.py` | 观测→推理→关节目标角，并持有 `last_action` 状态 |
| `selftest.py` | 干净环境自检，D2 的可判读判据 |

## 接口契约要点

从 `docs/contracts/` 读出，此处只作速查——**以 JSON 为准，不以本表为准**。

- 观测 45 维：`[0:3]` 机体角速度 ｜ `[3:6]` 重力投影 ｜ `[6:9]` 速度指令 `[vx,vy,wz]` ｜ `[9:21]` 关节角（减默认角）｜ `[21:33]` 关节角速度 ｜ `[33:45]` 上一步网络原始输出
- 块内顺序 = Isaac 关节顺序：四个 ABAD → 四个 HIP → 四个 KNEE，层内 `FL, FR, RL, RR`
- 缩放系数全部 1.0，无 clip，无经验归一化
- 动作：`p_des = 默认角 + 0.25 × 网络输出`
- 频率：50 Hz
- SDK 增益填**标称值** kp=25 / kd=0.6，不要填契约里 `sampled_*` 的任何数

## 尚未实现（等硬件）

| 钉子 | 内容 |
|---|---|
| D09 | 软启动 + 动作限幅 + 急停三件套。**没写完不许上电** |
| D10 | ≥200 Hz 写入线程。策略只有 50 Hz，直接写会被电机守护进程反复清指令 |
| D17 | SDK 结构体填充三条：每条腿 `flags` 置 1 ／ `foot` 槽显式清零 ／ `spline_cmd_data_t` 必须是 344 字节 |
| D14 | 实机编码器正方向是否与 URDF 一致。**仿真里绝对看不出来**，只能上电实测 |
