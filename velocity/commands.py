from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.tasks.velocity.mdp import (
    UniformVelocityCommand,
    UniformVelocityCommandCfg,
)

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


class AxisAlignedVelocityCommand(
    UniformVelocityCommand
):
    """
    Command chỉ hoạt động trên một trục:

    - v_x != 0 thì v_y = 0.
    - v_y != 0 thì v_x = 0.
    - standing environment: v_x = v_y = 0.
    """

    cfg: AxisAlignedVelocityCommandCfg

    def _resample_command(
        self,
        env_ids: torch.Tensor,
    ) -> None:

        # Lấy mẫu command bằng logic gốc của MjLab.
        super()._resample_command(env_ids)

        # Chọn môi trường đi theo x hoặc y.
        select_x = (
            torch.rand(
                len(env_ids),
                device=self.device,
            )
            < self.cfg.rel_x_commands
        )

        x_env_ids = env_ids[select_x]
        y_env_ids = env_ids[~select_x]

        # Đi theo x thì khóa y.
        self.vel_command_b[
            x_env_ids,
            1,
        ] = 0.0

        # Đi theo y thì khóa x.
        self.vel_command_b[
            y_env_ids,
            0,
        ] = 0.0


@dataclass(kw_only=True)
class AxisAlignedVelocityCommandCfg(
    UniformVelocityCommandCfg
):
    # Trong các environment chuyển động:
    # 50% đi theo x, 50% đi theo y.
    rel_x_commands: float = 0.5

    def build(
        self,
        env: ManagerBasedRlEnv,
    ) -> AxisAlignedVelocityCommand:

        return AxisAlignedVelocityCommand(
            self,
            env,
        )

    def __post_init__(self):

        super().__post_init__()

        if not 0.0 <= self.rel_x_commands <= 1.0:
            raise ValueError(
                "rel_x_commands must be "
                "between 0 and 1."
            )