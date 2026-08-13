import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor


def gait_phase(env, period: float, command_name: str):
    phase = (env.episode_length_buf * env.step_dt) % period / period

    obs = torch.stack(
        (
            torch.sin(2.0 * torch.pi * phase),
            torch.cos(2.0 * torch.pi * phase),
        ),
        dim=1,
    )

    command = env.command_manager.get_command(command_name)

    # Khi đứng yên thì không cần gait phase.
    stand = torch.linalg.norm(command, dim=1) < 0.05

    return torch.where(
        stand.unsqueeze(1),
        torch.zeros_like(obs),
        obs,
    )


def feet_gait(
    env,
    period: float,
    offset: list[float],
    threshold: float,
    command_threshold: float,
    command_name: str,
    sensor_name: str,
):
    sensor: ContactSensor = env.scene[sensor_name]

    # True nếu chân đang chạm đất.
    is_contact = sensor.data.current_contact_time > 0.0

    global_phase = (
        (env.episode_length_buf * env.step_dt) / period
    ).unsqueeze(1)

    offsets = torch.tensor(
        offset,
        device=env.device,
        dtype=global_phase.dtype,
    ).view(1, -1)

    leg_phase = (global_phase + offsets) % 1.0

    # threshold > 0.5 tạo một khoảng double support nhỏ.
    desired_stance = leg_phase < threshold

    reward = (
        desired_stance == is_contact
    ).float().mean(dim=1)

    command = env.command_manager.get_command(command_name)

    active = (
        torch.linalg.norm(command[:, :2], dim=1)
        + torch.abs(command[:, 2])
    ) > command_threshold

    return reward * active.float()

def feet_clearance_flat(
    env,
    target_height: float,
    command_name: str,
    command_threshold: float,
    asset_cfg: SceneEntityCfg,
):
    asset: Entity = env.scene[asset_cfg.name]

    # Độ cao của foot sites so với world.
    # Flat terrain hiện tại nằm tại z = 0.
    foot_z = asset.data.site_pos_w[
        :, asset_cfg.site_ids, 2
    ]

    # Vận tốc ngang của chân.
    foot_vel_xy = asset.data.site_lin_vel_w[
        :, asset_cfg.site_ids, :2
    ]

    vel_norm = torch.norm(
        foot_vel_xy,
        dim=-1,
    )

    # Sai lệch so với clearance mong muốn.
    delta = torch.abs(
        foot_z - target_height
    )

    # Chỉ phạt mạnh khi chân đang chuyển động ngang.
    cost = torch.sum(
        delta * vel_norm,
        dim=1,
    )

    command = env.command_manager.get_command(
        command_name
    )

    active = (
        torch.norm(command[:, :2], dim=1)
        + torch.abs(command[:, 2])
    ) > command_threshold

    return cost * active.float()