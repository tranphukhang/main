from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import mujoco
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import (
    MjlabOnPolicyRunner,
    RslRlVecEnvWrapper,
)
from mjlab.utils.wrappers import VideoRecorder

from evaluation.cop_support_logger import (
    CopSupportLogger,
)
from velocity.ppo_cfg import velocity_ppo_runner_cfg
from velocity.velocity_env_cfg import velocity_env_cfg


# ============================================================
# Evaluation configuration
# ============================================================

CHECKPOINT = Path(
    "logs/rsl_rl/velocity_v1/"
    "2026-08-24_11-39-09_baseline_v1/"
    "model_1998.pt"
)

EPISODE_LENGTH_S = 15.0

# Case đầu tiên: đi tới theo trục x.
TEST_COMMAND = (
    0.15,
    0.0,
    0.0,
)

CASE_NAME = "forward_x_0p15"

def build_eval_env() -> ManagerBasedRlEnv:
    env_cfg = velocity_env_cfg()

    env_cfg.viewer.width = 1280
    env_cfg.viewer.height = 720

    env_cfg.scene.num_envs = 1
    env_cfg.episode_length_s = EPISODE_LENGTH_S

    # Không dùng curriculum khi evaluation.
    env_cfg.curriculum = {}

    # Không cho sampler đổi command trong episode.
    twist_cfg = env_cfg.commands["twist"]

    twist_cfg.resampling_time_range = (
        EPISODE_LENGTH_S + 1.0,
        EPISODE_LENGTH_S + 1.0,
    )

    # Không tạo standing command ngẫu nhiên.
    twist_cfg.rel_standing_envs = 0.0

    device = (
        "cuda:0"
        if torch.cuda.is_available()
        else "cpu"
    )

    return ManagerBasedRlEnv(
        cfg=env_cfg,
        device=device,
        render_mode="rgb_array",
    )


def lock_test_command(
    env: ManagerBasedRlEnv,
) -> None:
    """
    Ghi đè command sampler bằng command evaluation cố định.
    """

    twist_term = env.command_manager.get_term(
        "twist"
    )

    fixed_command = torch.tensor(
        TEST_COMMAND,
        device=env.device,
        dtype=twist_term.vel_command_b.dtype,
    )

    twist_term.vel_command_b[:] = fixed_command

    # Bảo đảm UniformVelocityCommand không biến command
    # thành (0, 0, 0) ở bước update tiếp theo.
    twist_term.is_standing_env[:] = False


def main():
    env = build_eval_env()

    try:
        # reset() làm MjLab khởi tạo mọi manager.
        env.reset()

        # Sau reset mới ghi đè command ngẫu nhiên ban đầu.
        lock_test_command(env)

        actual_command = env.command_manager.get_command(
            "twist"
        )[0]

        print(
            "Fixed evaluation command: "
            f"{actual_command.cpu().tolist()}"
        )

    finally:
        env.close()


if __name__ == "__main__":
    main()