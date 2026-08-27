# AGIself · D1 edu 部署侧运行时
"""把训练出来的 policy 跑在真实机器人上所需的最小闭环。

**本包运行时不 import isaaclab，也不 import torch。** 这不是风格偏好，是交付条 D2
（可脱离 Isaac Lab 加载）的结构性保证 —— 依赖写错了连装都装不上，不会等到板子上才发现。

模块分工：
    contract.py     读 bundle/ 里的契约 JSON，校验指纹，暴露冻结常量。**一个数都不手抄**
    observation.py  45 维观测拼装（D05 落在这里）
    policy.py       ONNX 推理封装
    runner.py       观测 → 推理 → 关节目标角，并持有 last_action 状态
    selftest.py     干净环境自检：45 维进、12 维出，D2 的可判读判据

尚未实现（Stage B，等明天硬件到位）：
    SDK 结构体填充（D17）／软启动·限幅·急停（D09）／200 Hz 写入线程（D10）
"""

__version__ = "0.1.0"

__all__ = [
    "ContractError",
    "get_contract",
    "ObservationBuilder",
    "OnnxPolicy",
    "PolicyRunner",
]


def __getattr__(name):
    """惰性转发。

    这样 `import agiself_d1_deploy` 本身不会拖起 numpy / onnxruntime，
    也不会在 bundle 缺失时炸在 import 阶段 —— 想拿契约的人自己去调 get_contract()，
    那时候再响亮地失败。
    """
    if name in ("ContractError", "get_contract"):
        from . import contract

        return getattr(contract, name)
    if name == "ObservationBuilder":
        from .observation import ObservationBuilder

        return ObservationBuilder
    if name == "OnnxPolicy":
        from .policy import OnnxPolicy

        return OnnxPolicy
    if name == "PolicyRunner":
        from .runner import PolicyRunner

        return PolicyRunner
    raise AttributeError(name)
