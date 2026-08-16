from pathlib import Path

import mjlab.scripts.play as play_script
from mjlab.scripts.play import PlayConfig
from mjlab.tasks.registry import register_mjlab_task

from standing.standing_env_cfg import standing_env_cfg
from standing.ppo_cfg import standing_ppo_runner_cfg
from standing.standing_viewer import StandingViserViewer


TASK_ID = "Mjlab-Standing-Flat-Custom"


def main():

    # ---------------------------------------------------------
    # Find model_100.pt
    # ---------------------------------------------------------

    checkpoints = list(
        Path("logs/rsl_rl/standing_v1").glob("*/model_2595.pt")
    )
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
    play_env_cfg.episode_length_s = 30.0
    play_env_cfg.events["body_impulse"].params["cooldown_s"] = (2.0, 2.0)

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

    # Dùng custom Viser viewer nhưng vẫn giữ nguyên toàn bộ checkpoint loading của mjlab.
    play_script.ViserPlayViewer = StandingViserViewer

    play_script.run_play(
        TASK_ID,
        play_cfg,
    )


if __name__ == "__main__":
    main()