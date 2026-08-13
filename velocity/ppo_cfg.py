from mjlab.rl import (
    RslRlModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)


def velocity_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:

    return RslRlOnPolicyRunnerCfg(

        # -----------------------------------------------------
        # Actor
        # -----------------------------------------------------

        actor=RslRlModelCfg(
            hidden_dims=(256, 128, 64),
            activation="elu",
            obs_normalization=True,

            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        ),

        # -----------------------------------------------------
        # Critic
        # -----------------------------------------------------

        critic=RslRlModelCfg(
            hidden_dims=(256, 128, 64),
            activation="elu",
            obs_normalization=True,
        ),

        # -----------------------------------------------------
        # PPO
        # -----------------------------------------------------

        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,

            clip_param=0.2,

            entropy_coef=0.01,

            num_learning_epochs=5,
            num_mini_batches=4,

            learning_rate=1.0e-4,
            schedule="adaptive",

            gamma=0.99,
            lam=0.95,

            desired_kl=0.01,

            max_grad_norm=1.0,
        ),

        # -----------------------------------------------------
        # Runner
        # -----------------------------------------------------

        experiment_name="velocity_v1",
        run_name = "x_tracking_v3",

        resume=True,

        load_run="2026-08-13_10-36-53_x_tracking_v2",
        load_checkpoint="model_998.pt",

        num_steps_per_env=24,

        save_interval=50,

        max_iterations=500,

        # Giới hạn raw policy action trong [-1, 1].
        clip_actions=1.0,

        # V1 dùng TensorBoard trước cho đơn giản.
        logger="tensorboard",

        upload_model=False,
    )