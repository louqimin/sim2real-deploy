# AGIself · D1 edu 部署侧观测拼装
"""把机器人身上读到的原始量，拼成策略认识的那个 45 维向量。

这是 D05 落进代码的地方，也是整条链路上最容易出**静默错误**的一段 ——
拼错一块、少减一次默认角、块内顺序错位，网络照样吃得下、照样输出 12 个数、
机器人照样动。仿真里不报错，实机上就是乱蹬。

所以这里的写法刻意笨：
  1. 每一块按契约声明的 slice 往一个全零向量里**填**，而不是按调用顺序 concat。
     顺序错位这种错，用 concat 写法查不出来，用 slice 写法根本不会发生。
  2. 输入一律要求「绝对量」（编码器读数原样），减默认角由本模块内部做。
     调用方不需要知道哪一块是相对量 —— 那是契约的知识，不是调用方的知识。
  3. 每个入参都查长度，长度不对当场抛错。

依赖：numpy。不 import torch，不 import isaaclab。
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .contract import Contract, ContractError, get_contract


class ObservationBuilder:
    """45 维观测拼装器。一台机器人一个实例，可反复调用 build()。"""

    def __init__(self, contract: Optional[Contract] = None):
        self.c = contract if contract is not None else get_contract()
        n = self.c.action_dim

        self._default_pos = np.asarray(self.c.default_joint_pos, dtype=np.float32)
        self._default_vel = np.asarray(self.c.default_joint_vel, dtype=np.float32)
        self._n_joints = n

        # 每块观测该减哪个默认值。契约里 relative_to_default=False 的块减零向量。
        # 注意 joint_vel：契约标了 relative_to_default=True，但默认关节速度恒为 0，
        # 所以「相对」在数值上是恒等的。这里照样走减法，是为了让代码与契约逐字对应 ——
        # 哪天训练侧把默认速度改成非零，这里自动跟上，不需要有人记得来改。
        self._offsets = {}
        for b in self.c.obs_blocks:
            if not b.relative_to_default:
                continue
            if b.name == "joint_pos":
                self._offsets[b.name] = self._default_pos
            elif b.name == "joint_vel":
                self._offsets[b.name] = self._default_vel
            else:
                raise ContractError(
                    f"观测项 {b.name!r} 标了 relative_to_default，但部署侧不知道该减什么默认值"
                )

    # ---------- 内部工具 ----------

    def _check(self, name: str, arr, dim: int) -> np.ndarray:
        v = np.asarray(arr, dtype=np.float32).reshape(-1)
        if v.size != dim:
            raise ValueError(f"{name} 期望 {dim} 个数，实际收到 {v.size} 个")
        if not np.all(np.isfinite(v)):
            raise ValueError(f"{name} 里出现 NaN 或 inf：{v}")
        return v

    # ---------- 主接口 ----------

    def build(
        self,
        base_ang_vel,
        projected_gravity,
        velocity_commands,
        joint_pos,
        joint_vel,
        last_action,
    ) -> np.ndarray:
        """拼出一帧观测。

        参数全部是**绝对量**、全部按 Isaac 关节顺序：

        base_ang_vel      (3,)  机体系角速度 rad/s，来自 IMU 陀螺仪
        projected_gravity (3,)  重力方向在机体系下的单位向量，静止水平时约 (0, 0, -1)
        velocity_commands (3,)  我们下发的 [vx, vy, wz]，不是测出来的
        joint_pos         (12,) 编码器绝对角 rad —— 别在外面减默认角，这里会减
        joint_vel         (12,) 编码器角速度 rad/s
        last_action       (12,) 上一步**网络原始输出**，不是 p_des。首帧传全零

        返回 (obs_dim,) 的 float32。
        """
        c = self.c
        n = self._n_joints
        raw = {
            "base_ang_vel": self._check("base_ang_vel", base_ang_vel, 3),
            "projected_gravity": self._check("projected_gravity", projected_gravity, 3),
            "velocity_commands": self._check("velocity_commands", velocity_commands, 3),
            "joint_pos": self._check("joint_pos", joint_pos, n),
            "joint_vel": self._check("joint_vel", joint_vel, n),
            "actions": self._check("last_action", last_action, n),
        }

        obs = np.zeros(c.obs_dim, dtype=np.float32)
        for b in c.obs_blocks:
            v = raw[b.name]
            off = self._offsets.get(b.name)
            if off is not None:
                v = v - off
            if b.scale != 1.0:
                v = v * np.float32(b.scale)
            if b.clip is not None:
                v = np.clip(v, b.clip[0], b.clip[1])
            obs[b.start : b.stop] = v
        return obs

    # ---------- 动作侧换算 ----------

    def action_to_joint_targets(self, action) -> np.ndarray:
        """网络输出 → 关节目标角。

        p_des[j] = default_joint_pos[j] + action_scale * action[j]

        这个式子的两个数（默认角、缩放）都来自契约，不在代码里写死。
        返回的就是要填进 SDK `joint_control_t.p_des` 的值，单位 rad。
        """
        a = self._check("action", action, self._n_joints)
        return self._default_pos + np.float32(self.c.action_scale) * a

    @property
    def default_joint_pos(self) -> np.ndarray:
        """默认站姿角的只读副本。软启动（D09）的目标姿态要用它。"""
        return self._default_pos.copy()
