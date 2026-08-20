from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.envs import mdp


if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


# ============================================================
# Base linear velocity XY penalty
# ============================================================

def base_lin_vel_xy_l2(
    env: ManagerBasedRlEnv,
) -> torch.Tensor:
    """
    Phạt vận tốc tuyến tính của base trên mặt phẳng XY.

    Cost:
        vx^2 + vy^2

    Mục tiêu:
        vx -> 0
        vy -> 0
    """

    base_lin_vel = mdp.base_lin_vel(
        env
    )

    return torch.sum(
        torch.square(
            base_lin_vel[:, :2]
        ),
        dim=1,
    )


# ============================================================
# Base angular velocity XY penalty
# ============================================================

def base_ang_vel_xy_l2(
    env: ManagerBasedRlEnv,
) -> torch.Tensor:
    """
    Phạt vận tốc góc roll/pitch của base.

    Cost:
        wx^2 + wy^2

    Mục tiêu:
        wx -> 0
        wy -> 0
    """

    base_ang_vel = mdp.base_ang_vel(
        env
    )

    return torch.sum(
        torch.square(
            base_ang_vel[:, :2]
        ),
        dim=1,
    )


def joint_soft_limit_penalty(
    env: ManagerBasedRlEnv,
    asset_cfg,
    soft_ratio: float = 0.8,
) -> torch.Tensor:
    """
    Phạt khi vị trí khớp đi vào vùng gần giới hạn cơ khí.

    soft_ratio = 0.8:
        - Không phạt trong 80% khoảng chuyển động.
        - Bắt đầu phạt từ 80% đến hard limit.
        - Penalty = 1 tại hard limit của mỗi khớp.
    """

    robot = env.scene[asset_cfg.name]

    q = robot.data.joint_pos[
        :,
        asset_cfg.joint_ids,
    ]

    hard_limits = robot.data.joint_pos_limits[
        :,
        asset_cfg.joint_ids,
        :,
    ]

    hard_lower = hard_limits[:, :, 0]
    hard_upper = hard_limits[:, :, 1]

    # Tâm của khoảng giới hạn
    center = 0.5 * (
        hard_lower + hard_upper
    )

    # Một nửa khoảng chuyển động
    half_range = 0.5 * (
        hard_upper - hard_lower
    )

    # Soft limits = 80% hard range
    soft_half_range = (
        soft_ratio * half_range
    )

    soft_lower = center - soft_half_range
    soft_upper = center + soft_half_range

    # Khoảng từ soft limit tới hard limit
    margin = (
        half_range - soft_half_range
    )

    # Mức vượt soft limit
    lower_excess = torch.clamp(
        soft_lower - q,
        min=0.0,
    )

    upper_excess = torch.clamp(
        q - soft_upper,
        min=0.0,
    )

    excess = (
        lower_excess
        + upper_excess
    )

    # Chuẩn hóa:
    # 0 tại soft limit
    # 1 tại hard limit
    normalized_excess = (
        excess
        / torch.clamp(
            margin,
            min=1e-6,
        )
    )

    penalty = (
        normalized_excess ** 2
    )

    return torch.sum(
        penalty,
        dim=1,
    )