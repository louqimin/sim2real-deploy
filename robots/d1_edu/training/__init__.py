# AGIself · gym 环境注册
"""把 D1 edu 的训练环境注册进 gymnasium。

注意本文件**只允许 import gymnasium**。
entry_point 与 env_cfg_entry_point 都写成字符串，gym.register 不会去解析它们，
真正的 import 推迟到 AppLauncher 启动之后才发生。
若在这里直接 import 配置模块，会连带拉起 isaaclab -> pxr，
而 pxr 只在 Isaac Sim 运行时里存在，导致注册阶段就崩。
"""

import gymnasium as gym

gym.register(
    id="AGIself-D1-Edu-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.AGIself_flat_env_cfg:AGIselfD1EduFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.AGIself_rsl_rl_ppo_cfg:AGIselfD1EduFlatPPORunnerCfg",
    },
)

gym.register(
    id="AGIself-D1-Edu-Flat-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.AGIself_flat_env_cfg:AGIselfD1EduFlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{__name__}.AGIself_rsl_rl_ppo_cfg:AGIselfD1EduFlatPPORunnerCfg",
    },
)
