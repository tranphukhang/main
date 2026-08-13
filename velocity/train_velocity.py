import mjlab.scripts.train as mjlab_train

from velocity.tensorboard_video import TensorBoardVideoRecorder
# Dùng recorder custom của project thay cho recorder mặc định.
mjlab_train.VideoRecorder = TensorBoardVideoRecorder

from mjlab.tasks.registry import register_mjlab_task

from velocity.velocity_env_cfg import velocity_env_cfg
from velocity.ppo_cfg import velocity_ppo_runner_cfg


TASK_ID = "Mjlab-Velocity-Flat-Custom"


def main():

    # ---------------------------------------------------------
    # Training environment
    # ---------------------------------------------------------

    train_env_cfg = velocity_env_cfg()

    train_env_cfg.scene.num_envs = 1024

    # ---------------------------------------------------------
    # Play environment
    # ---------------------------------------------------------

    play_env_cfg = velocity_env_cfg()

    play_env_cfg.scene.num_envs = 5

    # ---------------------------------------------------------
    # PPO
    # ---------------------------------------------------------

    rl_cfg = velocity_ppo_runner_cfg()

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

        # Bật recording.
        video=True,

        # Mỗi video dài 200 environment steps.
        video_length=200,

        # Cứ 2400 environment steps record một video.
        video_interval=2400,
    )

    mjlab_train.launch_training(
        TASK_ID,
        args=train_cfg,
    )


if __name__ == "__main__":
    main()