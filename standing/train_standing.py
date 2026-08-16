import mjlab.scripts.train as mjlab_train
from velocity.tensorboard_video import TensorBoardVideoRecorder
mjlab_train.VideoRecorder = TensorBoardVideoRecorder
from mjlab.tasks.registry import register_mjlab_task

from standing.standing_env_cfg import standing_env_cfg
from standing.ppo_cfg import standing_ppo_runner_cfg


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

    train_cfg = mjlab_train.TrainConfig(
        env=train_env_cfg,
        agent=rl_cfg,

        video=True,

        # Standing chạy 50 Hz:
        # 300 steps = 6 giây.
        video_length=300,

        # 24 steps / PPO iteration
        # 2400 steps = khoảng 100 iterations.
        video_interval=2400,
    )

    mjlab_train.launch_training(
        TASK_ID,
        args=train_cfg,
    )


if __name__ == "__main__":
    main()