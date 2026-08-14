import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor


def stand_still_joint_deviation_l1(
    env,
    command_name: str,
    command_threshold: float,
    asset_cfg: SceneEntityCfg,
):
    robot = env.scene[asset_cfg.name]

    q = robot.data.joint_pos[:, asset_cfg.joint_ids]
    q0 = robot.data.default_joint_pos[:, asset_cfg.joint_ids]

    deviation = torch.sum(
        torch.abs(q - q0),
        dim=1,
    )

    command = env.command_manager.get_command(command_name)

    command_norm = (
        torch.linalg.norm(command[:, :2], dim=1)
        + torch.abs(command[:, 2])
    )

    standing = command_norm < command_threshold

    return deviation * standing.float()


def joint_deviation_l1(
    env,
    asset_cfg: SceneEntityCfg,
):
    robot = env.scene[asset_cfg.name]

    q = robot.data.joint_pos[:, asset_cfg.joint_ids]
    q0 = robot.data.default_joint_pos[:, asset_cfg.joint_ids]

    return torch.sum(
        torch.abs(q - q0),
        dim=1,
    )


def no_jumps(
    env,
    sensor_name: str,
    threshold: float,
):
    sensor: ContactSensor = env.scene[sensor_name]

    force = torch.linalg.norm(
        sensor.data.force,
        dim=-1,
    )

    in_contact = force > threshold

    # Phạt khi cả hai chân đều không contact.
    zero_contact = (~in_contact).all(dim=1)

    return zero_contact.float()