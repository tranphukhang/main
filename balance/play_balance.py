from pathlib import Path

import mujoco

import mjlab.scripts.play as play_script
from mjlab.scripts.play import PlayConfig
from mjlab.tasks.registry import register_mjlab_task
from mjlab.viewer import NativeMujocoViewer

from balance.balance_env_cfg import balance_env_cfg
from balance.ppo_cfg import balance_ppo_runner_cfg


TASK_ID = "Mjlab-Balance-Flat-Custom"


# ============================================================
# Native viewer
# ============================================================

class BalanceNativeViewer(NativeMujocoViewer):

    def setup(self):

        # Giữ toàn bộ setup mặc định của MJLab
        super().setup()

        # -----------------------------------------------------
        # Hiển thị contact points
        # -----------------------------------------------------

        self.viewer.opt.flags[
            mujoco.mjtVisFlag.mjVIS_CONTACTPOINT
        ] = 1

        # -----------------------------------------------------
        # Hiển thị contact forces
        # -----------------------------------------------------

        self.viewer.opt.flags[
            mujoco.mjtVisFlag.mjVIS_CONTACTFORCE
        ] = 1


# ============================================================
# Find latest checkpoint
# ============================================================

def find_latest_checkpoint() -> Path:

    log_root = Path(
        "logs/rsl_rl/balance_v1"
    )

    checkpoints = list(
        log_root.glob(
            "*/model_*.pt"
        )
    )

    if not checkpoints:
        raise FileNotFoundError(
            f"Không tìm thấy checkpoint trong: {log_root}"
        )

    # Lấy checkpoint được tạo gần nhất
    checkpoint = max(
        checkpoints,
        key=lambda path: path.stat().st_mtime,
    )

    return checkpoint


# ============================================================
# Main
# ============================================================

def main():

    # --------------------------------------------------------
    # 1. Checkpoint
    # --------------------------------------------------------

    checkpoint = find_latest_checkpoint()

    print(
        f"Loading checkpoint: {checkpoint}"
    )

    # --------------------------------------------------------
    # 2. Environment
    # --------------------------------------------------------

    train_env_cfg = balance_env_cfg()

    play_env_cfg = balance_env_cfg()

    # Native viewer chỉ cần 1 robot
    play_env_cfg.scene.num_envs = 1

    # Cho mỗi episode dài hơn để quan sát
    play_env_cfg.episode_length_s = 30.0

    # --------------------------------------------------------
    # 3. Play tại mức disturbance cuối cùng
    # --------------------------------------------------------

    # Không dùng curriculum khi play.
    # Nếu giữ curriculum thì play bắt đầu lại từ stage 3 N.
    play_env_cfg.curriculum = {}

    # Test policy với mức push cuối cùng
    play_env_cfg.events[
        "body_impulse"
    ].params[
        "force_range"
    ] = (-15.0, 15.0)

    # Giữ duration giống training
    play_env_cfg.events[
        "body_impulse"
    ].params[
        "duration_s"
    ] = (0.10, 0.16)

    # Giữ cooldown giống training
    play_env_cfg.events[
        "body_impulse"
    ].params[
        "cooldown_s"
    ] = (2.0, 4.0)

    # --------------------------------------------------------
    # 4. Register task
    # --------------------------------------------------------

    register_mjlab_task(
        task_id=TASK_ID,
        env_cfg=train_env_cfg,
        play_env_cfg=play_env_cfg,
        rl_cfg=balance_ppo_runner_cfg(),
    )

    # --------------------------------------------------------
    # 5. Native viewer
    # --------------------------------------------------------

    play_cfg = PlayConfig(
        checkpoint_file=str(
            checkpoint
        ),

        viewer="native",

        num_envs=1,
    )

    # Thay Native viewer mặc định
    # bằng viewer có contact visualization
    play_script.NativeMujocoViewer = (
        BalanceNativeViewer
    )

    # --------------------------------------------------------
    # 6. Play
    # --------------------------------------------------------

    play_script.run_play(
        TASK_ID,
        play_cfg,
    )


if __name__ == "__main__":
    main()