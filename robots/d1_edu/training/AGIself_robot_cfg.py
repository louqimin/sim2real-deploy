# AGIself · D1 edu 的 ArticulationCfg
"""智元 D1 edu 四足机器人的 Isaac Lab 本体配置。

每一个数字都有出处，禁止凭印象填：
  关节名 / 连杆名 / 限位 / 力矩 / 速度  ->  assets/urdf_doctor_report.json（URDF 体检，D15 已验）
  Isaac 侧关节顺序                      ->  docs/contracts/d1_edu_joint_map.json（D06 实测，14 项全 PASS）
  kp = 25 / kd = 0.6                    ->  D03 拍板（2026-08-29 试过 80/1.0 已退回，见项目档案）

参照物是 Isaac Lab 自带的 UNITREE_GO2_CFG（isaaclab_assets/robots/unitree.py:140）。
Go2 官方整备 15 kg、D1 edu 的 URDF 质量 15.186 kg，量级吻合，故其结构可直接对标。
"""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

##
# 资产路径
##

# 本文件位于 robots/d1_edu/training/，向上两级即 robots/d1_edu/
_D1_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D1_EDU_USD_PATH = os.path.join(_D1_ROOT, "assets", "usd", "d1_edu.usd")

##
# 拍板参数（改这里，不要改下面的 CFG 字面量）
##

D1_EDU_STIFFNESS = 50.0  # 25->50,腿太软，kp，十二关节统一。D03；2026-08-29 试过 80.0，策略被力矩惩罚吓住、knee 冻住不动，已退回
D1_EDU_DAMPING = 0.85  # 跟着改，kd，十二关节统一。D03；随 kp 一同退回，试验值曾为 1.0

# 站立默认姿态。这不是「可选的初值」而是必填项：
# KNEE 限位 [-2.723, -0.602] 全在负区间，缺省的 0 位会被 ArticulationCfg 的校验判为越限，
# 直接导致实例化失败。本组合已由 tools/dump_joint_order.py 实例化通过。
#
# 它同时是部署契约的一部分——实机上 joint_pos 观测是 (当前角 - 本默认角)，
# 动作是 p_des = 本默认角 + 0.25 * 网络输出。两处用的必须是同一组数。
D1_EDU_DEFAULT_JOINT_POS = {
    ".*_ABAD_JOINT": 0.0,
    # 后腿 HIP 抬到 1.2 —— D20 配平。2026-08-29 起。
    #
    # 依据不是估算：2026-08-27 续3 部署侧自检喂一帧理想站姿，读出策略自己想要的
    # HIP 目标角是前腿 0.87/0.78、后腿 1.23/1.22。四条腿共用 0.8 时，策略每一步都在
    # 花动作把后腿往上顶 0.4 rad —— 那是本该由默认姿态解决的偏差。
    # 这里只是把它已经在做的事写进默认角，省下它的动作预算。
    #
    # 参照：Go2 也是前后不同（thigh 前 0.8 / 后 1.0，unitree.py:162-163）。
    # 质心不在几何中心，四条腿共用一个角度本就站不平。
    #
    # ⚠ 四条 HIP 必须逐条写全，不能用「通配 + 专用覆盖」的写法：
    # Isaac Lab 的 resolve_matching_names_values 不接受一个关节被两个模式同时命中，
    # 直接抛 ValueError（isaaclab/utils/string.py:328），没有「后写的赢」这回事。
    # 2026-08-29 写成 "R.*_HIP_JOINT" + ".*_HIP_JOINT"，当场炸在 RL_HIP_JOINT 上。
    "FL_HIP_JOINT": 0.8,
    "FR_HIP_JOINT": 0.8,
    "RL_HIP_JOINT": 1.2,
    "RR_HIP_JOINT": 1.2,
    ".*_KNEE_JOINT": -1.5,
}

##
# 本体配置
##

D1_EDU_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=D1_EDU_USD_PATH,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # 0.31 = AGIself_check_stance.py 实测稳态机身高度 0.2892 m 上浮 2 cm。
        # 2026-08-29 后腿 HIP 配平到 1.2 之后重测；配平前是 0.2626 m（对应 z=0.28）。
        # 上浮是为了 reset 时不穿模；上浮太多则每回合开局都要摔一下，白白污染早期样本。
        pos=(0.0, 0.0, 0.31),
        joint_pos=D1_EDU_DEFAULT_JOINT_POS,
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*_ABAD_JOINT", ".*_HIP_JOINT", ".*_KNEE_JOINT"],
            # 力矩上限 48 N·m / 速度上限 28 rad/s 已经写在 URDF 里，转换进 USD 时保留下来了
            # （dump_joint_order.py 读回值与 URDF 体检逐位吻合）。这里不重复指定，
            # 避免 effort_limit 与 effort_limit_sim 在不同 Isaac Lab 小版本间的字段名分歧。
            stiffness=D1_EDU_STIFFNESS,
            damping=D1_EDU_DAMPING,
            friction=0.0,
        ),
    },
)
"""D1 edu 的隐式执行器（Implicit Actuator）配置。

选 ImplicitActuatorCfg 而不是 Go2 用的 DCMotorCfg，理由是与实机 SDK 对齐：
D1 的电机侧执行 tau = kp * (p_des - p) + kd * (v_des - v) + t_ff（D01 收账，lowlevel.h 原文），
这正是隐式执行器的语义。DCMotorCfg 会额外模拟力矩-转速饱和曲线，而 SDK 并未暴露该特性，
仿真里建模了实机上却无法复现，反而扩大 sim2real 落差。
"""
