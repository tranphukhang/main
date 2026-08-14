from __future__ import annotations

from typing import TYPE_CHECKING

from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


import torch


def ankle_passive_soft_limit_penalty(
    env,
    asset_cfg: SceneEntityCfg,
    soft_limit: float,
    hard_limit: float,
):
    robot = env.scene[asset_cfg.name]

    q = robot.data.joint_pos[:, asset_cfg.joint_ids]

    # Chỉ bắt đầu phạt khi vượt soft limit.
    excess = torch.clamp(
        torch.abs(q) - soft_limit,
        min=0.0,
    )

    span = hard_limit - soft_limit

    # 0 tại soft limit, 1 tại hard limit.
    penalty = (excess / span) ** 2

    return torch.sum(penalty, dim=1)