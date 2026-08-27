# AGIself · D1 edu 的 ArticulationCfg
"""智元 D1 edu 四足机器人的 Isaac Lab 本体配置。

每一个数字都有出处，禁止凭印象填：
  关节名 / 连杆名 / 限位 / 力矩 / 速度  ->  assets/urdf_doctor_report.json（URDF 体检，D15 已验）
  Isaac 侧关节顺序                      ->  docs/contracts/d1_edu_joint_map.json（D06 实测，14 项全 PASS）
  kp = 25 / kd = 0.6                    ->  D03 拍板（理论反推 + 同级参考机对标）

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

D1_EDU_STIFFNESS = 25.0  # kp，十二关节统一。D03
D1_EDU_DAMPING = 0.6  # kd，十二关节统一。D03

# 站立默认姿态。这不是「可选的初值」而是必填项：
# KNEE 限位 [-2.723, -0.602] 全在负区间，缺省的 0 位会被 ArticulationCfg 的校验判为越限，
# 直接导致实例化失败。本组合已由 tools/dump_joint_order.py 实例化通过。
#
# 它同时是部署契约的一部分——实机上 joint_pos 观测是 (当前角 - 本默认角)，
# 动作是 p_des = 本默认角 + 0.25 * 网络输出。两处用的必须是同一组数。
D1_EDU_DEFAULT_JOINT_POS = {
    ".*_ABAD_JOINT": 0.0,
    ".*_HIP_JOINT": 0.8,
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
        # 0.28 = AGIself_check_stance.py 实测稳态机身高度 0.2626 m 上浮 2 cm。
        # 上浮是为了 reset 时不穿模；上浮太多则每回合开局都要摔一下，白白污染早期样本。
        pos=(0.0, 0.0, 0.28),
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
