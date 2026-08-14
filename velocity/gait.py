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

def feet_flat_contact(
    env,
    period: float,
    offset: list[float],
    stance_threshold: float,
    edge_margin: float,
    min_force: float,
    command_threshold: float,
    command_name: str,
    toe_sensor_name: str,
    heel_sensor_name: str,
):
    toe_sensor: ContactSensor = env.scene[toe_sensor_name]
    heel_sensor: ContactSensor = env.scene[heel_sensor_name]

    # ---------------------------------------------------------
    # Toe / heel contact
    # ---------------------------------------------------------

    toe_force = torch.linalg.norm(
        toe_sensor.data.force,
        dim=-1,
    )

    heel_force = torch.linalg.norm(
        heel_sensor.data.force,
        dim=-1,
    )

    # Mức contact liên tục từ 0 -> 1.
    toe_score = torch.clamp(
        toe_force / min_force,
        min=0.0,
        max=1.0,
    )

    heel_score = torch.clamp(
        heel_force / min_force,
        min=0.0,
        max=1.0,
    )

    # Flat-foot bị quyết định bởi đầu chân đang chịu tải ít hơn.
    flat_score = torch.minimum(
        toe_score,
        heel_score,
    )

    # ---------------------------------------------------------
    # Gait phase của từng chân
    # ---------------------------------------------------------

    global_phase = (
        (env.episode_length_buf * env.step_dt) / period
    ).unsqueeze(1)

    offsets = torch.tensor(
        offset,
        device=env.device,
        dtype=global_phase.dtype,
    ).view(1, -1)

    leg_phase = (global_phase + offsets) % 1.0

    # Chỉ yêu cầu flat-foot ở giữa stance.
    mid_stance = (
        (leg_phase >= edge_margin)
        & (leg_phase < (stance_threshold - edge_margin))
    )

    # Thông thường chỉ một chân nằm trong mid-stance.
    walking_reward = (
        flat_score * mid_stance.float()
    ).sum(dim=1)

    standing_reward = flat_score.mean(dim=1)

    command = env.command_manager.get_command(command_name)

    moving = (
        torch.linalg.norm(command[:, :2], dim=1)
        + torch.abs(command[:, 2])
    ) > command_threshold

    return torch.where(
        moving,
        walking_reward,
        standing_reward,
    )

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