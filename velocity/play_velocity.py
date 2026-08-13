from pathlib import Path

import mjlab.scripts.play as play_script
from mjlab.scripts.play import PlayConfig
from mjlab.tasks.registry import register_mjlab_task

from velocity.velocity_env_cfg import velocity_env_cfg
from velocity.ppo_cfg import velocity_ppo_runner_cfg


TASK_ID = "Mjlab-Velocity-Flat-Custom"


def main():

    # ---------------------------------------------------------
    # Find model_100.pt
    # ---------------------------------------------------------

    checkpoints = list(
        Path("logs/rsl_rl/velocity_v1/2026-08-13_08-22-35_x_tracking_v1").glob("*/model_499.pt")
    )

    # Nếu có nhiều run, lấy model_100.pt của run mới nhất.
    checkpoint = max(
        checkpoints,
        key=lambda path: path.stat().st_mtime,
    )

    print(f"Loading checkpoint: {checkpoint}")

    # ---------------------------------------------------------
    # Register task
    # ---------------------------------------------------------

    train_env_cfg = velocity_env_cfg()

    play_env_cfg = velocity_env_cfg()
    play_env_cfg.scene.num_envs = 1
    play_env_cfg.episode_length_s = 30.0


    register_mjlab_task(
        task_id=TASK_ID,
        env_cfg=train_env_cfg,
        play_env_cfg=play_env_cfg,
        rl_cfg=velocity_ppo_runner_cfg(),
    )

    # ---------------------------------------------------------
    # Play
    # ---------------------------------------------------------

    play_cfg = PlayConfig(
        checkpoint_file=str(checkpoint),
        viewer="native",
    )

    play_script.run_play(
        TASK_ID,
        play_cfg,
    )


if __name__ == "__main__":
    main()