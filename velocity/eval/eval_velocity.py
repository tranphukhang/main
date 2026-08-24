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