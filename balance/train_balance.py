import mjlab.scripts.train as mjlab_train

from velocity.tensorboard_video import TensorBoardVideoRecorder

# Dùng custom recorder để đưa video lên TensorBoard
mjlab_train.VideoRecorder = TensorBoardVideoRecorder

from mjlab.tasks.registry import register_mjlab_task

from balance.balance_env_cfg import balance_env_cfg
from balance.ppo_cfg import balance_ppo_runner_cfg


TASK_ID = "Mjlab-Balance-Flat-Custom"


def main():

    # ---------------------------------------------------------
    # Training environment
    # ---------------------------------------------------------

    train_env_cfg = balance_env_cfg()

    # ---------------------------------------------------------
    # Resume training:
    # policy đã hoàn thành curriculum 3 -> 15 N
    # nên giữ disturbance ở mức cuối 15 N
    # ---------------------------------------------------------

    train_env_cfg.curriculum = {}

    train_env_cfg.events[
        "body_impulse"
    ].params[
        "force_range"
    ] = (-15.0, 15.0)

    # Số môi trường chạy song song khi train
    train_env_cfg.scene.num_envs = 1024

    # ---------------------------------------------------------
    # Play environment
    # ---------------------------------------------------------

    play_env_cfg = balance_env_cfg()

    # Chỉ dùng ít env khi visualize / play
    play_env_cfg.scene.num_envs = 5

    # ---------------------------------------------------------
    # PPO configuration
    # ---------------------------------------------------------

    rl_cfg = balance_ppo_runner_cfg()

    # ---------------------------------------------------------
    # Register task
    # ---------------------------------------------------------

    register_mjlab_task(
        task_id=TASK_ID,
        env_cfg=train_env_cfg,
        play_env_cfg=play_env_cfg,
        rl_cfg=rl_cfg,
    )

    # ---------------------------------------------------------
    # Training configuration
    # ---------------------------------------------------------

    train_cfg = mjlab_train.TrainConfig(
        env=train_env_cfg,
        agent=rl_cfg,

        # -----------------------------------------------------
        # Record video
        # -----------------------------------------------------

        video=True,

        # Robot chạy ở 50 Hz
        # 300 control steps = 6 s
        video_length=300,

        # PPO:
        # 24 steps / iteration
        #
        # 2400 / 24 = 100 iterations
        #
        # -> Record khoảng mỗi 100 iterations
        video_interval=2400,
    )

    # ---------------------------------------------------------
    # Launch training
    # ---------------------------------------------------------

    mjlab_train.launch_training(
        TASK_ID,
        args=train_cfg,
    )


if __name__ == "__main__":
    main()