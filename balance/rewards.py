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