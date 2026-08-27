# AGIself · D1 edu 部署侧策略循环
"""观测拼装 + 推理 + 动作换算，合成部署侧真正会调用的那一个对象。

单独拎出这一层，主要是为了**把 last_action 这个状态关起来**。

`last_action` 是观测的第六块，内容是上一步网络的原始输出。它是整条链路上唯一的
内部状态，也是最容易出错的地方：忘了更新、更新成了 p_des 而不是原始输出、复位时
没清零 —— 三种错都让机器人照常跑，只是行为不对。放进类里由 step() 自己维护，
调用方就没有写错的机会。

依赖：numpy + onnxruntime。仍然不碰 torch、不碰 isaaclab。
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .contract import Contract, get_contract
from .observation import ObservationBuilder
from .policy import OnnxPolicy


class PolicyRunner:
    """一次 step = 一帧观测进、一组关节目标角出。"""

    def __init__(
        self,
        contract: Optional[Contract] = None,
        policy: Optional[OnnxPolicy] = None,
        builder: Optional[ObservationBuilder] = None,
    ):
        self.c = contract if contract is not None else get_contract()
        self.builder = builder if builder is not None else ObservationBuilder(self.c)
        self.policy = policy if policy is not None else OnnxPolicy(contract=self.c)
        self._last_action = np.zeros(self.c.action_dim, dtype=np.float32)
        self.n_steps = 0

    def reset(self) -> None:
        """复位内部状态。

        清成全零，是因为训练侧 Isaac 的 ActionManager 在 episode 复位时也把
        last_action 清成全零。首帧两边必须从同一个起点出发，否则逐帧对账从第一帧就开始偏。
        """
        self._last_action[:] = 0.0
        self.n_steps = 0

    @property
    def last_action(self) -> np.ndarray:
        return self._last_action.copy()

    def step(
        self,
        base_ang_vel,
        projected_gravity,
        velocity_commands,
        joint_pos,
        joint_vel,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """跑一帧。

        入参含义见 ObservationBuilder.build()，一律是绝对量、Isaac 关节顺序。
        注意这里**没有** last_action 参数 —— 那是本对象自己管的状态。

        返回三样：
            p_des  (12,) 关节目标角 rad，填进 SDK 的 joint_control_t.p_des
            action (12,) 网络原始输出，留给调用方记录/调试
            obs    (45,) 这一帧实际喂进网络的观测，逐帧对账要用
        """
        obs = self.builder.build(
            base_ang_vel=base_ang_vel,
            projected_gravity=projected_gravity,
            velocity_commands=velocity_commands,
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            last_action=self._last_action,
        )
        action = self.policy(obs)
        p_des = self.builder.action_to_joint_targets(action)

        # 存进去的是网络原始输出，不是 p_des。契约原文：
        # 「actions 观测缓存的是网络原始输出，不是 p_des」
        self._last_action = action.copy()
        self.n_steps += 1
        return p_des, action, obs
