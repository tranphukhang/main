from mjlab.scripts.train import launch_training
from mjlab.tasks.registry import register_mjlab_task

from standing_env_cfg import standing_env_cfg
from ppo_cfg import standing_ppo_runner_cfg


TASK_ID = "Mjlab-Standing-Flat-Custom"


def main():

    # ---------------------------------------------------------
    # Training environment
    # ---------------------------------------------------------

    train_env_cfg = standing_env_cfg()

    train_env_cfg.scene.num_envs = 1024

    # ---------------------------------------------------------
    # Play environment
    # ---------------------------------------------------------

    play_env_cfg = standing_env_cfg()

    play_env_cfg.scene.num_envs = 5

    # ---------------------------------------------------------
    # PPO
    # ---------------------------------------------------------

    rl_cfg = standing_ppo_runner_cfg()

    # ---------------------------------------------------------
    # Register local task
    # ---------------------------------------------------------

    register_mjlab_task(
        task_id=TASK_ID,
        env_cfg=train_env_cfg,
        play_env_cfg=play_env_cfg,
        rl_cfg=rl_cfg,
    )

    # ---------------------------------------------------------
    # Train
    # ---------------------------------------------------------

    launch_training(TASK_ID)


if __name__ == "__main__":
    main()