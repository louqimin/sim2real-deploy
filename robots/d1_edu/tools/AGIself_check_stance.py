#!/usr/bin/env python3
# AGIself · 站姿验证
"""让 D1 edu 用默认关节角在平地上站一秒，看它站不站得住。

这个脚本回答一个仿真里唯一能回答、但不问就一直悬着的问题：
    AGIself_robot_cfg.py 里那组默认关节角，
    在 D1 的实际符号约定下，究竟是「站着」还是「腿朝天翻过去」？

判据（不接受「看起来跑起来了」）：
  PASS  机身高度稳定在 0.20 ~ 0.45 m，重力在机体系 z 轴投影 < -0.9（即基本水平）
  FAIL  机身塌到 0.15 m 以下，或重力投影 z > -0.7（机身翻倒 / 大幅倾斜）

同时输出实测站立高度，用于回填 init_state.pos 的 z。

用法：
    python AGIself_check_stance.py
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Check D1 edu default standing pose.")
parser.add_argument("--seconds", type=float, default=1.5, help="站立观察时长（秒）")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import os  # noqa: E402
import sys  # noqa: E402

import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.AGIself_robot_cfg import D1_EDU_CFG  # noqa: E402

DT = 1.0 / 200.0


def main():
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=DT, device=args.device))

    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=750.0).func("/World/light", sim_utils.DomeLightCfg(intensity=750.0))

    robot = Articulation(D1_EDU_CFG.replace(prim_path="/World/Robot"))
    sim.reset()

    # 目标角就是默认角：什么都不做，只让 PD 把机器人保持在默认姿态上
    target = robot.data.default_joint_pos.clone()

    n_steps = int(args.seconds / DT)
    heights, grav_z = [], []
    for i in range(n_steps):
        robot.set_joint_position_target(target)
        robot.write_data_to_sim()
        sim.step()
        robot.update(DT)
        if i > n_steps // 2:  # 只统计后半段，前半段是落地瞬态
            heights.append(float(robot.data.root_pos_w[0, 2]))
            grav_z.append(float(robot.data.projected_gravity_b[0, 2]))

    h = sum(heights) / len(heights)
    g = sum(grav_z) / len(grav_z)
    h_std = float(torch.tensor(heights).std())

    print("\n" + "=" * 62)
    # 从仿真读回实际生效的默认角，不写死字面量。
    # 与上面 target = robot.data.default_joint_pos.clone() 同源，
    # 所以打印的就是 PD 正在保持的那组数，不存在「显示与实际分叉」的可能。
    #
    # 2026-08-29 教训：这里原本硬编码 "ABAD 0.0 / HIP 0.8 / KNEE -1.5"，
    # 而当天后腿 HIP 已改成 1.2 —— 一个验证工具报错了自己的输入。
    # 那份输出贴进档案就是「HIP 0.8 配 z=-0.9999」这种不存在的组合，且看不出破绽。
    _names = getattr(robot.data, "joint_names", None) or robot.joint_names
    _defaults = robot.data.default_joint_pos[0].tolist()
    print("默认关节角（仿真读回）:")
    for _i in range(0, len(_names), 4):
        _row = "  ".join(
            f"{_n.replace('_JOINT', ''):<8}{_v:+.3f}"
            for _n, _v in zip(_names[_i:_i + 4], _defaults[_i:_i + 4])
        )
        print(f"  {_row}")
    print(f"稳态机身高度   : {h:.4f} m   (标准差 {h_std:.4f})")
    print(f"重力机体系 z   : {g:.4f}     (-1.0 = 完全水平)")
    print("-" * 62)

    ok_h = 0.20 <= h <= 0.45
    ok_g = g < -0.9
    if ok_h and ok_g:
        print(f"PASS —— 姿态成立。把 init_state.pos 的 z 回填为 {h + 0.02:.2f} 更贴切")
    else:
        print("FAIL —— 默认姿态站不住，判定如下：")
        if not ok_h:
            print(f"  机身高度 {h:.4f} m 落在 [0.20, 0.45] 之外 —— 腿的伸展方向很可能反了")
        if not ok_g:
            print(f"  重力投影 {g:.4f} 说明机身已倾倒")
        print("  下一步：把 HIP 改成 -0.8、KNEE 改成 -1.5 再跑一次，对比两组结果")
    print("=" * 62 + "\n")


if __name__ == "__main__":
    import traceback

    code = 0
    try:
        main()
    except BaseException:
        print("\n脚本失败，traceback：", flush=True)
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        code = 1

    simulation_app.close()
    sys.exit(code)
