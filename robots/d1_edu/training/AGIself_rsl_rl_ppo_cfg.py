# AGIself · D1 edu 的 PPO 训练配置
"""rsl-rl-lib 5.0.1 的 PPO 超参，对标 UnitreeGo2FlatPPORunnerCfg。

算法层超参在四足 locomotion 领域是一组几乎不动的常数，
这里原样沿用参照配置，不做「调优」——冲刺期的迭代预算要留给奖励与 sim2real，
而不是花在裁剪系数上。
"""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class AGIselfD1EduFlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    # 参照 Go2 平地版是 300。这里给到 1000 并保持 50 一存，
    # 于是可以随时按 Ctrl-C 停下取最近的 checkpoint，不必事先押注迭代数。
    max_iterations = 1000
    save_interval = 50
    experiment_name = "agiself_d1_edu_flat"

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        # ==== D07 就是这两行 ====
        # rsl-rl 5.0.1 里旧字段 empirical_normalization 已废弃，拆成了这两个。
        # 必须是 False：经验归一化的均值方差是训练过程产生的、存在 checkpoint 里的隐式状态，
        # 部署时得原样搬到实机才能对上。冲刺期每多一个必须搬运的隐式状态，
        # 就多一个静默出错的口子。改用写死在契约表里的固定缩放（本项目全部为 1.0）。
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        # =======================
        actor_hidden_dims=[128, 128, 128],
        critic_hidden_dims=[128, 128, 128],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",  # 按 KL 散度自动调学习率，省掉手调
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
