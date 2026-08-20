from mjlab.rl import (
    RslRlModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)


def balance_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:

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
                "init_std": 0.5,
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

        experiment_name="balance_v1",
        run_name="baseline_v1",

        resume=False,

        # load_run="2026-08-20_13-21-58_baseline_v1",
        # load_checkpoint="model_499.pt",

        num_steps_per_env=24,

        # Lưu checkpoint mỗi 50 iteration
        save_interval=50,

        # Tổng số learning iterations
        max_iterations=500,

        # Raw policy action nằm trong [-1, 1]
        clip_actions=1.0,

        logger="tensorboard",

        upload_model=False,
    )