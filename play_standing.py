from pathlib import Path

from mjlab.scripts.play import PlayConfig, run_play
from mjlab.tasks.registry import register_mjlab_task

from standing_env_cfg import standing_env_cfg
from ppo_cfg import standing_ppo_runner_cfg


TASK_ID = "Mjlab-Standing-Flat-Custom"


def main():

    # ---------------------------------------------------------
    # Find model_100.pt
    # ---------------------------------------------------------

    checkpoints = list(
        Path("logs/rsl_rl/standing_v1").glob("*/model_100.pt")
    )

    if not checkpoints:
        raise FileNotFoundError("Không tìm thấy model_100.pt")

    # Nếu có nhiều run, lấy model_100.pt của run mới nhất.
    checkpoint = max(
        checkpoints,
        key=lambda path: path.stat().st_mtime,
    )

    print(f"Loading checkpoint: {checkpoint}")

    # ---------------------------------------------------------
    # Register task
    # ---------------------------------------------------------

    train_env_cfg = standing_env_cfg()

    play_env_cfg = standing_env_cfg()
    play_env_cfg.scene.num_envs = 1

    register_mjlab_task(
        task_id=TASK_ID,
        env_cfg=train_env_cfg,
        play_env_cfg=play_env_cfg,
        rl_cfg=standing_ppo_runner_cfg(),
    )

    # ---------------------------------------------------------
    # Play
    # ---------------------------------------------------------

    play_cfg = PlayConfig(
        checkpoint_file=str(checkpoint),
        viewer="viser",
    )

    run_play(
        TASK_ID,
        play_cfg,
    )


if __name__ == "__main__":
    main()