# AGIself · D1 edu 平地速度跟踪环境配置
"""继承 Isaac Lab 内置的 LocomotionVelocityRoughEnvCfg，改成 D1 edu 的平地版。

只训平地、不训粗地形，是冲刺期的明确取舍（见 [[项目主线-训狗D1]]）。

本文件负责钉死的契约（D05）：
  块间顺序 = [角速度3 | 重力投影3 | 速度指令3 | 关节角12 | 关节角速度12 | 上一步动作12] = 45 维
  块内顺序 = Isaac 关节顺序，由 D06 收账
  缩放系数 = 全部 1.0（骨架里没有任何 ObsTerm 设 scale）
  两个相对量:
      joint_pos 观测 = 当前角 - 默认角（mdp.joint_pos_rel）
      动作      p_des = 默认角 + 0.25 * 网络输出（JointPositionActionCfg, use_default_offset=True）
"""

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    EventCfg,
    LocomotionVelocityRoughEnvCfg,
)

from .AGIself_robot_cfg import D1_EDU_CFG


@configclass
class AGIselfD1EduEventCfg(EventCfg):
    """在骨架的事件集上追加执行器增益随机化。

    D08 拍板：开，正负 30%，stiffness 与 damping 同时随机。
    动因不是「稳妥起见加一点随机」，而是 D03 查明官方从未给出 PD 增益推荐值 ——
    我们的 25 / 0.6 是反推来的，真值未知。随机化在这里是唯一兜底：
    逼策略在整个增益区间内都能走，实机真值只要落在区间内就可用。
    """

    randomize_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.7, 1.3),
            "damping_distribution_params": (0.7, 1.3),
            "operation": "scale",
            "distribution": "uniform",
        },
    )


@configclass
class AGIselfD1EduFlatEnvCfg(LocomotionVelocityRoughEnvCfg):
    """D1 edu 平地 locomotion 训练环境。"""

    events: AGIselfD1EduEventCfg = AGIselfD1EduEventCfg()

    def __post_init__(self):
        super().__post_init__()

        # ---------- 机器人 ----------
        self.scene.robot = D1_EDU_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # ---------- 地形改平地 ----------
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        self.curriculum.terrain_levels = None

        # ---------- 观测：钉死 45 维 ----------
        # D11 拍板：删掉 base_lin_vel。IMU 只给 acc / gyro / 四元数，不给本体线速度，
        # 留着它等于让策略依赖一个实机上根本拿不到的量。
        self.observations.policy.base_lin_vel = None
        # 平地不需要高度扫描，且实机无此传感器
        self.observations.policy.height_scan = None

        # ---------- 动作 ----------
        # 缩放 0.25 对标 Go2（rough_env_cfg.py:30）。这个数进部署契约表。
        self.actions.joint_pos.scale = 0.25

        # ---------- 速度指令 ----------
        # heading_command 关掉：开着的话 ang_vel_z 会被朝向控制器重算，
        # 观测里第三块就不再是「我们下发的三个数」，部署时对不上。关掉后
        # 指令块严格等于 [vx, vy, wz]，仿真与实机同构。
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.rel_standing_envs = 0.2
        self.commands.base_velocity.ranges.lin_vel_x = (-0.3, 0.8)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.3, 0.3)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.6, 0.6)

        # ---------- 事件 / 域随机化 ----------
        self.events.push_robot = None
        self.events.base_com = None
        self.events.add_base_mass.params["asset_cfg"].body_names = "BASE_LINK"
        self.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 3.0)
        self.events.base_external_force_torque.params["asset_cfg"].body_names = "BASE_LINK"
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }

        # ---------- 奖励 ----------
        # 权重整体对标 Go2 平地版，只在一处主动偏离，见下面 dof_pos_limits
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = ".*_FOOT_LINK"
        # 2026-08-29 R2：治后腿拖地（D18）。门槛与权重必须一起动，只动权重会更糟。
        #
        # 这一项只在「脚落地那一帧」结算，分数 = (这次腾空时长 - threshold)：
        #     腾空 > 门槛  ->  正分
        #     腾空 < 门槛  ->  负分（抬了但不够久，反而挨罚）
        #     从不抬脚      ->  落地事件不发生，这一项完全看不见它，0 分
        # 于是对一条抬不满门槛的腿，「干脆不抬」严格优于「抬一点」。
        # 8/27 与 8/29 两轮该项一直为负，正是这个局面：抬的腿在挨罚、拖的腿不计分。
        #
        # 0.5 s 对 D1 偏高 —— 指令上限 0.8 m/s，trot 摆动相通常 0.2~0.3 s。
        # 降到 0.3 让「抬起来」变得够得着，权重再提到 0.5 让它值得去够。
        self.rewards.feet_air_time.params["threshold"] = 0.3
        self.rewards.feet_air_time.weight = 0.5
                # 2026-08-30 R3：门槛 0.3 -> 0.1
        #
        # 探针实测各腿平均腾空 FL 0.051 / FR 0.021 / RR 0.152 秒，RL 从不离地。
        # 落地事件加权平均 0.096 秒 —— 门槛 0.3 比最好的那条腿还高一倍，
        # 意味着连 RR 那条正常迈步的腿都在挨罚。降到 0.1，让 RR 转正、
        # 让 FL 够得着，反向激励的幅度也随之缩小。
        #
        # ⚠ 门槛调低只是把反向激励「缩小」，没有「消除」：
        #    从不抬脚仍然得 0 分，仍然优于抬得不够久的负分。
        #    真正堵死这个口子的是下面的 feet_slide —— 见那一段的说明。
        self.rewards.feet_air_time.params["threshold"] = 0.1

        # 2026-08-30 R3：新增 feet_slide，治拖地（D18）
        #
        # 罚的是「脚踩着地却还在动」。取的是足端在世界系里的水平速度，
        # 所以量的是真正的对地滑动，不是相对机身的运动。
        #
        # ★ 它与 feet_air_time 的关系是互补，不是重复：
        #
        #     腿的行为        feet_air_time      feet_slide
        #     正常迈步        正分                接近 0（脚踩住不动）
        #     抬得不够久      负分                小罚
        #     从不抬脚        0 分（看不见它）    最大罚（全程贴地滑行）
        #
        #   feet_air_time 对「从不抬脚」是瞎的，那正是 RL 钻的空子；
        #   feet_slide 对它罚得最重。两项合起来，逃生口才被堵上。
        #
        # 权重 -0.25 的来历（不是抄的，是算的）：
        #   记录值满足  Episode_Reward/feet_slide = weight x s_bar
        #   （中间的 dt x 步数 / 回合时长 = 0.02 x 1000 / 20 = 1，正好消掉）
        #
        #   当前 s_bar = 1.298  ->  罚金 -0.32
        #   若四条腿都做到 RR 那样（单腿 slide 0.044）s_bar = 0.176  ->  罚金 -0.044
        #   修好步态能省下 0.28，而多抬腿增加的力矩/加速度代价约 0.1
        #   => 修步态严格划算，策略有动机去做
        #
        #   参考值：上游 h1 用 -0.25，g1 用 -0.1（都是双足）。
        #   取 -0.25 是因为上面那笔账在 -0.1 时只有 0.11 收益，与代价打平，推不动。
        #
        # 两个 body_names 必须写成一样的：这一项拿 sensor 侧的接触掩码
        # 逐元素乘 asset 侧的速度，两套下标顺序不同就会把腿配错。
        # 上游 CHANGELOG 0.10.10 修的正是这个 bug。
        # （2026-08-30 探针已实测本环境两套顺序一致，均为 FL/FR/RL/RR）
        self.rewards.feet_slide = RewTerm(
            func=mdp.feet_slide,
            weight=-0.25,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_FOOT_LINK"),
                "asset_cfg": SceneEntityCfg("robot", body_names=".*_FOOT_LINK"),
            },
        )


        # 2026-08-27 第一轮训练的教训：这一项曾照抄 Go2 设成 None，结果策略学会了「跪着走」——
        # 小腿贴地当接触点，稳、不摔、速度指令照样跟得上，于是 base_contact 终止率 0.0000、
        # 跟踪奖励拿到理论上限 96%，所有数值指标都漂亮，而步态是错的。只有看录像才发现。
        #
        # 根因是没有任何代价约束非足端接触，可用接触点就从「四个脚」扩大到「所有连杆」。
        # D1 默认姿态本就屈（机身 0.26 m，Go2 约 0.32 m），小腿离地极近，跪走是近在咫尺的局部最优。
        # Go2 关掉它没事，D1 关掉就出事 —— 抄配方要抄到形态差异为止。
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [".*_KNEE_LINK", ".*_HIP_LINK"]
        self.rewards.undesired_contacts.params["threshold"] = 1.0
        self.rewards.undesired_contacts.weight = -1.0
        self.rewards.dof_torques_l2.weight = -0.00005
        self.rewards.dof_acc_l2.weight = -2.5e-7
        self.rewards.track_lin_vel_xy_exp.weight = 1.5
        self.rewards.track_ang_vel_z_exp.weight = 0.75
        self.rewards.flat_orientation_l2.weight = -2.5
        # 主动偏离 Go2 的一项：Go2 把它留在 0.0（不惩罚），我们开到 -1.0。
        # 理由是 D1 的 KNEE 行程只有 2.12 rad（[-2.723, -0.602]），比 Go2 窄，
        # 而实机撞限位是硬件安全问题，不是「性能差一点」。
        self.rewards.dof_pos_limits.weight = -1.0

        # ---------- 终止 ----------
        self.terminations.base_contact.params["sensor_cfg"].body_names = "BASE_LINK"


@configclass
class AGIselfD1EduFlatEnvCfg_PLAY(AGIselfD1EduFlatEnvCfg):
    """回放 / 导出用的小规模无噪声版本。sim2sim 与 policy 导出都从这个环境走。"""

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 16
        self.scene.env_spacing = 2.5
        # 关掉观测噪声与随机推搡，导出的行为才是确定性的、可与部署侧逐帧对账
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
