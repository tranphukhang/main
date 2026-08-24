from pathlib import Path

import mujoco

import mjlab.scripts.play as play_script
from mjlab.scripts.play import PlayConfig
from mjlab.tasks.registry import register_mjlab_task
from mjlab.viewer import NativeMujocoViewer

from velocity.ppo_cfg import velocity_ppo_runner_cfg
from velocity.velocity_env_cfg import velocity_env_cfg


TASK_ID = "Mjlab-Velocity-Flat-Custom"


class VelocityNativeViewer(NativeMujocoViewer):
    """Native viewer with contact visualization."""

    def setup(self):
        super().setup()

        self.viewer.opt.flags[
            mujoco.mjtVisFlag.mjVIS_CONTACTPOINT
        ] = 1

        self.viewer.opt.flags[
            mujoco.mjtVisFlag.mjVIS_CONTACTFORCE
        ] = 1


def find_latest_checkpoint() -> Path:
    """Return the latest velocity-policy checkpoint."""

    log_root = Path(
        "logs/rsl_rl/velocity_v1"
    )

    checkpoints = list(
        log_root.glob("*/model_*.pt")
    )

    if not checkpoints:
        raise FileNotFoundError(
            f"Không tìm thấy checkpoint trong: {log_root}"
        )

    return max(
        checkpoints,
        key=lambda path: path.stat().st_mtime,
    )


def main():
    checkpoint = find_latest_checkpoint()

    print(
        f"Loading checkpoint: {checkpoint}"
    )

    # Environment config dùng để register task.
    train_env_cfg = velocity_env_cfg()
    train_env_cfg.scene.num_envs = 1024

    # Play bằng một robot để quan sát rõ hơn.
    play_env_cfg = velocity_env_cfg()
    play_env_cfg.scene.num_envs = 1
    play_env_cfg.episode_length_s = 30.0

    # Không cần curriculum khi inference/play.
    play_env_cfg.curriculum = {}

    register_mjlab_task(
        task_id=TASK_ID,
        env_cfg=train_env_cfg,
        play_env_cfg=play_env_cfg,
        rl_cfg=velocity_ppo_runner_cfg(),
    )

    play_cfg = PlayConfig(
        checkpoint_file=str(checkpoint),
        viewer="native",
        num_envs=1,
    )

    play_script.NativeMujocoViewer = (
        VelocityNativeViewer
    )

    play_script.run_play(
        TASK_ID,
        play_cfg,
    )


if __name__ == "__main__":
    main()