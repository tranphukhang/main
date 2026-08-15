import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor


def feet_air_time_positive_biped(
    env,
    threshold: float,
    command_name: str,
    command_threshold: float,
    sensor_name: str,
):
    sensor: ContactSensor = env.scene[sensor_name]

    air_time = sensor.data.current_air_time
    contact_time = sensor.data.current_contact_time

    assert air_time is not None
    assert contact_time is not None

    # Chân nào đang chạm đất.
    in_contact = contact_time > 0.0

    # Nếu contact -> dùng contact time.
    # Nếu swing   -> dùng air time.
    in_mode_time = torch.where(
        in_contact,
        contact_time,
        air_time,
    )

    # Biped: chỉ thưởng khi CHÍNH XÁC một chân chạm đất.
    single_stance = (
        torch.sum(in_contact.int(), dim=1) == 1
    )

    reward = torch.min(
        torch.where(
            single_stance.unsqueeze(1),
            in_mode_time,
            0.0,
        ),
        dim=1,
    ).values

    # Không cần single stance dài vô hạn.
    reward = torch.clamp(
        reward,
        max=threshold,
    )

    command = env.command_manager.get_command(
        command_name
    )

    active = (
        torch.linalg.norm(command[:, :2], dim=1)
        + torch.abs(command[:, 2])
    ) > command_threshold

    return reward * active.float()