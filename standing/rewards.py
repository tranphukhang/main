from __future__ import annotations

from typing import TYPE_CHECKING

from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def ankle_mechanical_limit_penalty(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    limit: float,
):
    robot = env.scene[asset_cfg.name]

    joint_pos = robot.data.joint_pos[:, asset_cfg.joint_ids]

    ankle_left = joint_pos[:, 0]
    calf_left = joint_pos[:, 1]

    ankle_right = joint_pos[:, 2]
    calf_right = joint_pos[:, 3]

    passive_angle_left = ankle_left - calf_left
    passive_angle_right = ankle_right - calf_right

    penalty_left = (passive_angle_left / limit) ** 4
    penalty_right = (passive_angle_right / limit) ** 4

    return penalty_left + penalty_right