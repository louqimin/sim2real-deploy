#!/usr/bin/env python3
# AGIself · D1 edu 步态探针
"""在钉死的速度指令下跑已训练策略，量出每条腿各自的步态统计。

动因
----
训练日志里的 Episode_Reward/feet_air_time 是四条腿求和之后的一个标量，
分腿信息在写进日志之前就被 torch.sum 抹掉了。要知道「哪条腿在拖」，
只能在仿真跑的那一刻把 ContactSensor 的张量读出来 —— 它从来没被存过盘。

量四组数
--------
  占空比       每条腿踩在地上的帧数占比           它到底离不离地
  步频 / 腾空  落地事件数与平均腾空时长           定 feet_air_time 的 threshold
  离地高度     摆动期足端 z 减去触地期足端 z      会不会绊（用户的原始诉求）
  滑动量       触地期足端的世界系水平速度         定 feet_slide 的 weight

与官方 play.py 的三处关键差别
-----------------------------
1. 不导出策略。
   play.py 每跑一次都会覆盖 <日志目录>/exported/policy.pt 与 policy.onnx，
   那两份是交付物（见钉子 D34），探针不许碰。

2. 指令钉死。
   不钉死的话，测出来的占空比是「站着 + 慢走 + 快走 + 转弯」的混合平均，无法解读。
   尤其 rel_standing_envs —— 那批被命令原地站着的环境会把占空比顶到接近 1.0，
   而且看不出破绽，只会以为狗在拖地。

3. 启动时核对两套足端顺序。
   sensor 侧与 asset 侧的下标各自解析，顺序不一致就会把腿配错。
   feet_slide 正是这样逐元素相乘的，上游 CHANGELOG 0.10.10 修过同类 bug。

用法（从仓库根跑）
------------------
    python robots/d1_edu/tools/AGIself_probe_gait.py --headless --num_envs 64

默认自动挑最新一轮训练；要指定轮次加 --load_run 2026-08-29_17-25-28
"""

import argparse
import os
import sys

# ---- 先注册我们的 gym 环境（与 AGIself_play.py 同一手法，已验证可用）----
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))  # -> robots/d1_edu

import training  # noqa: E402,F401  仅执行 gym.register

# ---- 借用官方 play.py 同目录的 cli_args ----
_STOCK_DIR = os.path.expanduser("~/IsaacLab/scripts/reinforcement_learning/rsl_rl")
if not os.path.isdir(_STOCK_DIR):
    raise SystemExit(
        f"找不到官方 rsl_rl 脚本目录：{_STOCK_DIR}\n"
        "用 find ~/IsaacLab -path '*rsl_rl/cli_args.py' 定位后改本文件的 _STOCK_DIR。"
    )
sys.path.insert(0, _STOCK_DIR)

from isaaclab.app import AppLauncher  # noqa: E402

import cli_args  # noqa: E402  isort: skip

parser = argparse.ArgumentParser(description="D1 edu 步态探针")
parser.add_argument("--task", type=str, default="AGIself-D1-Edu-Flat-Play-v0", help="任务名")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", help="agent 配置入口")
parser.add_argument("--num_envs", type=int, default=64, help="并行环境数，越多统计越稳")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--disable_fabric", action="store_true", default=False)
# ---- 探针自己的参数 ----
parser.add_argument("--probe_steps", type=int, default=600, help="采样步数，50 Hz ⇒ 600 步 = 12 秒")
parser.add_argument("--probe_warmup", type=int, default=50, help="开头丢弃多少步，等步态进入稳态")
parser.add_argument("--probe_vx", type=float, default=0.5, help="钉死的前进速度指令 m/s")
parser.add_argument("--foot_regex", type=str, default=".*_FOOT_LINK", help="足端连杆名正则")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# Hydra 会自己再解析一遍 sys.argv，这里只留它认得的部分
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""以下是仿真启动之后才能 import 的部分。"""

import importlib.metadata as metadata  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from packaging import version  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab.envs import ManagerBasedRLEnvCfg  # noqa: E402
from isaaclab.managers import SceneEntityCfg  # noqa: E402
from isaaclab.utils.assets import retrieve_file_path  # noqa: E402

from isaaclab_rl.rsl_rl import (  # noqa: E402
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
    handle_deprecated_rsl_rl_cfg,
)

import isaaclab_tasks  # noqa: E402,F401
from isaaclab_tasks.utils import get_checkpoint_path  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

_INSTALLED = metadata.version("rsl-rl-lib")


def _div(num, den):
    """分母为 0 时返回 None。

    分母为 0 在这里是有意义的结果而不是异常：
    某条腿一次都没落地，正是我们在找的东西。返回 None 让打印端显示 n/a。
    """
    return (num / den) if den > 0 else None


def _fmt(v, width=9, prec=4):
    """None 打成 n/a，保持列宽。"""
    return f"{'n/a':>{width}}" if v is None else f"{v:>{width}.{prec}f}"


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    # ================= 与官方 play.py 相同的样板 =================
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, _INSTALLED)
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    env_cfg.log_dir = os.path.dirname(resume_path)
    print(f"[探针] 载入权重：{resume_path}", flush=True)

    # ================= 探针专属：把指令钉死 =================
    cmd = env_cfg.commands.base_velocity
    cmd.resampling_time_range = (1.0e9, 1.0e9)  # 整段测量期间不重抽指令
    cmd.rel_standing_envs = 0.0                 # ★ 不许有「原地站着」的环境
    cmd.heading_command = False                 # ★ yaw 速度不再由朝向误差派生
    cmd.rel_heading_envs = 0.0
    cmd.ranges.lin_vel_x = (args_cli.probe_vx, args_cli.probe_vx)
    cmd.ranges.lin_vel_y = (0.0, 0.0)
    cmd.ranges.ang_vel_z = (0.0, 0.0)
    print(f"[探针] 指令钉死：vx={args_cli.probe_vx} m/s, vy=0, wz=0", flush=True)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    # ★ 此处刻意不调用 export_policy_to_jit / export_policy_to_onnx，理由见文件头第 1 条

    # ================= 解析足端下标，核对两套顺序 =================
    scene = env.unwrapped.scene
    sensor_cfg = SceneEntityCfg("contact_forces", body_names=args_cli.foot_regex)
    sensor_cfg.resolve(scene)
    asset_cfg = SceneEntityCfg("robot", body_names=args_cli.foot_regex)
    asset_cfg.resolve(scene)

    cs = scene.sensors["contact_forces"]
    robot = scene["robot"]

    sensor_ids = list(sensor_cfg.body_ids)
    asset_ids = list(asset_cfg.body_ids)
    asset_names = [robot.data.body_names[i] for i in asset_ids]

    all_sensor_names = list(getattr(cs, "body_names", []) or [])
    if all_sensor_names:
        sensor_names = [all_sensor_names[i] for i in sensor_ids]
        print(f"[探针] 传感器侧足端：{sensor_names}", flush=True)
        print(f"[探针] 本体侧足端  ：{asset_names}", flush=True)
        if sensor_names != asset_names:
            raise SystemExit(
                "两套足端顺序不一致 —— 逐元素相乘会把腿配错。\n"
                "这同时意味着 feet_slide 在本环境里也会配错，因为它正是这样相乘的。"
            )
        print("[探针] 顺序核对通过", flush=True)
    else:
        print(f"[探针] 本体侧足端  ：{asset_names}", flush=True)
        print("[探针] ⚠ 拿不到接触传感器的 body_names，顺序核对这一项没验成", flush=True)

    if len(asset_ids) != 4:
        raise SystemExit(f"匹配到 {len(asset_ids)} 个足端，期望 4 个。检查 --foot_regex")

    # ================= 累加器 =================
    dt = env.unwrapped.step_dt
    n_envs = env.unwrapped.num_envs
    n_feet = len(asset_ids)
    dev = env.unwrapped.device

    def _zeros():
        return torch.zeros(n_feet, device=dev)

    contact_frames, air_frames = _zeros(), _zeros()
    land_count, air_time_sum = _zeros(), _zeros()
    z_contact_sum, z_air_sum, slide_sum = _zeros(), _zeros(), _zeros()
    z_air_max = torch.full((n_feet,), -1.0e9, device=dev)

    # ================= 主循环 =================
    obs = env.get_observations()
    total_steps = args_cli.probe_warmup + args_cli.probe_steps
    print(f"[探针] 开始采样：预热 {args_cli.probe_warmup} 步 + 计数 {args_cli.probe_steps} 步", flush=True)

    for step in range(total_steps):
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            if version.parse(_INSTALLED) >= version.parse("4.0.0"):
                policy.reset(dones)

        # compute_first_contact 必须每步调用（它按 dt 判定「这一帧是不是刚落地」）
        first_contact = cs.compute_first_contact(dt)[:, sensor_ids].float()
        last_air = cs.data.last_air_time[:, sensor_ids]
        in_contact = (cs.data.current_contact_time[:, sensor_ids] > 0.0).float()
        in_air = 1.0 - in_contact
        foot_z = robot.data.body_pos_w[:, asset_ids, 2]
        foot_v = robot.data.body_lin_vel_w[:, asset_ids, :2].norm(dim=-1)

        if step < args_cli.probe_warmup:
            continue

        contact_frames += in_contact.sum(0)
        air_frames += in_air.sum(0)
        land_count += first_contact.sum(0)
        air_time_sum += (last_air * first_contact).sum(0)
        z_contact_sum += (foot_z * in_contact).sum(0)
        z_air_sum += (foot_z * in_air).sum(0)
        z_air_max = torch.maximum(
            z_air_max,
            torch.where(in_air > 0, foot_z, torch.full_like(foot_z, -1.0e9)).max(0).values,
        )
        slide_sum += (foot_v * in_contact).sum(0)

    env.close()

    # ================= 汇总 =================
    n_frames = n_envs * args_cli.probe_steps  # 每条腿被观察到的总帧数
    leg_seconds = n_frames * dt               # 每条腿被观察到的总秒数

    cf = contact_frames.cpu().tolist()
    af = air_frames.cpu().tolist()
    lc = land_count.cpu().tolist()
    ats = air_time_sum.cpu().tolist()
    zcs = z_contact_sum.cpu().tolist()
    zas = z_air_sum.cpu().tolist()
    zam = z_air_max.cpu().tolist()
    ss = slide_sum.cpu().tolist()

    line = "=" * 84
    print("\n" + line)
    print(
        f"  步态探针 · vx={args_cli.probe_vx} m/s · "
        f"{n_envs} 环境 x {args_cli.probe_steps} 步 = 每腿 {leg_seconds:.0f} 腿秒"
    )
    print(line)
    # 表头用 ASCII：Python 的字段宽度按字符数算，中文是双宽字符，混用会错位
    print(f"{'leg':<16}{'duty':>9}{'land/s':>9}{'air_s':>9}{'clr_avg':>9}{'clr_max':>9}{'slide':>9}")
    print("-" * 84)

    slide_total = 0.0
    for i, name in enumerate(asset_names):
        duty = _div(cf[i], n_frames)
        freq = _div(lc[i], leg_seconds)
        mean_air = _div(ats[i], lc[i])
        z_ground = _div(zcs[i], cf[i])
        z_swing = _div(zas[i], af[i])
        clr_avg = None if (z_swing is None or z_ground is None) else z_swing - z_ground
        clr_max = None if (z_ground is None or zam[i] < -1.0e8) else zam[i] - z_ground
        slide = _div(ss[i], n_frames)
        if slide is not None:
            slide_total += slide
        print(
            f"{name:<16}{_fmt(duty)}{_fmt(freq)}{_fmt(mean_air)}"
            f"{_fmt(clr_avg)}{_fmt(clr_max)}{_fmt(slide)}"
        )

    print("-" * 84)
    print(f"{'SUM slide (s_bar)':<16}{'':>45}{slide_total:>9.4f}  m/s")
    print(line)
    print(
        """
列名对照
  leg      足端连杆名
  duty     占空比。踩在地上的时间占比。正常小跑约 0.5，越接近 1 越是贴地拖行
  land/s   这条腿每秒完成几个步态周期
  air_s    平均腾空时长（秒）。feet_air_time 的 threshold 应设在这个值以下
  clr_avg  离地高度均值。摆动期足端 z 减去触地期足端 z，
           足端原点位置在相减时被消掉，所以这是真实离地量
  clr_max  离地高度峰值。跨越障碍的余量看这个
  slide    触地期足端的世界系水平速度。脚正常踩住时接近 0，拖地时接近机身速度

s_bar 的用法
  Episode_Reward/feet_slide = weight x s_bar
  中间的 dt x 步数 / 回合时长 = 0.02 x 1000 / 20 = 1，正好消掉。
  例：想让这项罚金落在 -0.05，而 s_bar = 0.5  =>  weight = -0.1

读法提醒
  · 某腿的 air_s 打 n/a = 它在整段测量里一次都没离过地。这是结果，不是错误。
  · 测量在 Play 配置下进行（推力扰动本就关闭），所以这张表量的是纯步态，
    不含抗扰动能力。抗扰动要看 base_contact 终止率与录像。
"""
    )


if __name__ == "__main__":
    main()
    simulation_app.close()
