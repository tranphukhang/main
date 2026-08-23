from __future__ import annotations

from typing import TYPE_CHECKING

import mujoco_warp as mjwarp
import torch

from balance.rewards import support_contact_substep

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


class velocity_contact_substep(support_contact_substep):
    """
    Kế thừa bộ lọc raw contact từ balance và bổ sung:
    - trạng thái contact của từng chân;
    - thời gian air/contact liên tục của từng chân.

    Vẫn tạo support_sum/support_count để dùng lại
    support_contact_reward của balance.
    """

    def __init__(self, cfg, env):

        super().__init__(cfg, env)

        self.left_in_contact = torch.zeros(
            env.num_envs,
            device=env.device,
            dtype=torch.bool,
        )

        self.right_in_contact = torch.zeros(
            env.num_envs,
            device=env.device,
            dtype=torch.bool,
        )

        self.left_air_time = torch.zeros(
            env.num_envs,
            device=env.device,
            dtype=torch.float32,
        )

        self.right_air_time = torch.zeros(
            env.num_envs,
            device=env.device,
            dtype=torch.float32,
        )

        self.left_contact_time = torch.zeros(
            env.num_envs,
            device=env.device,
            dtype=torch.float32,
        )

        self.right_contact_time = torch.zeros(
            env.num_envs,
            device=env.device,
            dtype=torch.float32,
        )

        env._left_foot_contact = self.left_in_contact
        env._right_foot_contact = self.right_in_contact

        env._left_air_time = self.left_air_time
        env._right_air_time = self.right_air_time

        env._left_contact_time = self.left_contact_time
        env._right_contact_time = self.right_contact_time

    def __call__(
        self,
        env,
        min_normal_force: float = 1.0,
    ) -> torch.Tensor:

        mjwarp.contact_force(
            env.sim.wp_model,
            env.sim.wp_data,
            self.contact_ids_wp,
            False,
            self.contact_force_wp,
        )

        normal_force = self.contact_force_torch[:, 0]

        contact_geom = env.sim.data.contact.geom

        geom1 = contact_geom[:, 0].long()
        geom2 = contact_geom[:, 1].long()

        floor_is_geom1 = geom1 == self.floor_geom_id
        floor_is_geom2 = geom2 == self.floor_geom_id

        foot_geom = torch.where(
            floor_is_geom1,
            geom2,
            torch.where(
                floor_is_geom2,
                geom1,
                torch.full_like(geom1, -1),
            ),
        )

        valid_foot_geom = foot_geom >= 0

        safe_foot_geom = torch.clamp(
            foot_geom,
            min=0,
        )

        foot_body = self.geom_bodyid[safe_foot_geom]

        valid_force = normal_force > min_normal_force

        left_contact = (
            valid_foot_geom
            & valid_force
            & (foot_body == self.left_foot_body_id)
        )

        right_contact = (
            valid_foot_geom
            & valid_force
            & (foot_body == self.right_foot_body_id)
        )

        world_id = env.sim.data.contact.worldid.long()

        valid_world = (
            (world_id >= 0)
            & (world_id < env.num_envs)
        )

        left_count = torch.zeros(
            env.num_envs,
            device=env.device,
            dtype=torch.float32,
        )

        right_count = torch.zeros(
            env.num_envs,
            device=env.device,
            dtype=torch.float32,
        )

        left_mask = left_contact & valid_world
        right_mask = right_contact & valid_world

        left_count.scatter_add_(
            0,
            world_id[left_mask],
            torch.ones_like(
                world_id[left_mask],
                dtype=torch.float32,
            ),
        )

        right_count.scatter_add_(
            0,
            world_id[right_mask],
            torch.ones_like(
                world_id[right_mask],
                dtype=torch.float32,
            ),
        )

        # -----------------------------------------------------
        # Cập nhật trạng thái contact và timer hai chân
        # -----------------------------------------------------

        left_in_contact = left_count > 0
        right_in_contact = right_count > 0

        physics_dt = env.physics_dt

        self.left_air_time.copy_(
            torch.where(
                left_in_contact,
                torch.zeros_like(self.left_air_time),
                self.left_air_time + physics_dt,
            )
        )

        self.right_air_time.copy_(
            torch.where(
                right_in_contact,
                torch.zeros_like(self.right_air_time),
                self.right_air_time + physics_dt,
            )
        )

        self.left_contact_time.copy_(
            torch.where(
                left_in_contact,
                self.left_contact_time + physics_dt,
                torch.zeros_like(self.left_contact_time),
            )
        )

        self.right_contact_time.copy_(
            torch.where(
                right_in_contact,
                self.right_contact_time + physics_dt,
                torch.zeros_like(self.right_contact_time),
            )
        )

        self.left_in_contact.copy_(left_in_contact)
        self.right_in_contact.copy_(right_in_contact)

        # -----------------------------------------------------
        # Giữ nguyên support-contact logic của balance
        # -----------------------------------------------------

        total_count = left_count + right_count

        double_support = (
            (left_count >= 1)
            & (right_count >= 1)
            & (total_count >= 3)
        )

        left_single_support = (
            (left_count >= 3)
            & (right_count == 0)
        )

        right_single_support = (
            (right_count >= 3)
            & (left_count == 0)
        )

        valid_support = (
            double_support
            | left_single_support
            | right_single_support
        )

        valid_support_float = valid_support.float()

        self.support_sum += valid_support_float
        self.support_count += 1.0

        return valid_support_float

    def reset(
        self,
        env_ids: torch.Tensor | None = None,
    ):

        super().reset(env_ids)

        if env_ids is None:

            self.left_in_contact.zero_()
            self.right_in_contact.zero_()

            self.left_air_time.zero_()
            self.right_air_time.zero_()

            self.left_contact_time.zero_()
            self.right_contact_time.zero_()

            return

        self.left_in_contact[env_ids] = False
        self.right_in_contact[env_ids] = False

        self.left_air_time[env_ids] = 0.0
        self.right_air_time[env_ids] = 0.0

        self.left_contact_time[env_ids] = 0.0
        self.right_contact_time[env_ids] = 0.0


def feet_air_time(
    env: ManagerBasedRlEnv,
    threshold: float = 0.4,
    command_name: str | None = None,
    command_threshold: float = 0.1,
) -> torch.Tensor:
    """Thưởng gait single-support có air/contact time gần threshold."""

    in_contact = torch.stack(
        (
            env._left_foot_contact,
            env._right_foot_contact,
        ),
        dim=1,
    )

    air_time = torch.stack(
        (
            env._left_air_time,
            env._right_air_time,
        ),
        dim=1,
    )

    contact_time = torch.stack(
        (
            env._left_contact_time,
            env._right_contact_time,
        ),
        dim=1,
    )

    in_mode_time = torch.where(
        in_contact,
        contact_time,
        air_time,
    )

    single_stance = (
        torch.mean(in_contact.float(), dim=1) == 0.5
    )

    mode_time = torch.min(
        torch.where(
            single_stance.unsqueeze(-1),
            in_mode_time,
            torch.zeros_like(in_mode_time),
        ),
        dim=1,
    )[0]

    reward = torch.clamp(
        threshold - torch.abs(mode_time - threshold),
        min=0.0,
    )

    if command_name is not None:

        command = env.command_manager.get_command(command_name)

        if command is not None:

            command_norm = (
                torch.norm(command[:, :2], dim=1)
                + torch.abs(command[:, 2])
            )

            reward *= (
                command_norm > command_threshold
            ).float()

    return reward


def feet_slip(
    env: ManagerBasedRlEnv,
    command_name: str,
    asset_cfg,
    command_threshold: float = 0.1,
) -> torch.Tensor:
    """Phạt vận tốc ngang của chân đang contact."""

    robot = env.scene[asset_cfg.name]

    in_contact = torch.stack(
        (
            env._left_foot_contact,
            env._right_foot_contact,
        ),
        dim=1,
    ).float()

    foot_vel_xy = robot.data.site_lin_vel_w[
        :,
        asset_cfg.site_ids,
        :2,
    ]

    slip_cost = torch.sum(
        torch.sum(
            torch.square(foot_vel_xy),
            dim=-1,
        )
        * in_contact,
        dim=1,
    )

    command = env.command_manager.get_command(command_name)

    if command is not None:

        command_norm = (
            torch.norm(command[:, :2], dim=1)
            + torch.abs(command[:, 2])
        )

        slip_cost *= (
            command_norm > command_threshold
        ).float()

    return slip_cost

def com_height_l2(
    env: ManagerBasedRlEnv,
    asset_cfg,
    target_height: float = 0.29,
    std: float = 0.02,
) -> torch.Tensor:
    """Phạt sai lệch chiều cao COM toàn robot."""

    robot = env.scene[
        asset_cfg.name
    ]

    root_body_id = (
        robot.indexing.root_body_id
    )

    com_height = env.sim.data.subtree_com[
        :,
        root_body_id,
        2,
    ]

    height_error = (
        com_height - target_height
    ) / std

    return torch.square(
        height_error
    )

def feet_lift(
    env: ManagerBasedRlEnv,
    asset_cfg,
    command_name: str | None = None,
    command_threshold: float = 0.05,
    height_saturation: float = 0.15,
) -> torch.Tensor:
    """
    Thưởng chân vung được nhấc cao hơn mặt đất.

    Reward tăng tuyến tính từ 0 đến 1 khi độ cao chân
    tăng từ 0 đến height_saturation; cao hơn ngưỡng này
    không nhận thêm lợi ích.
    """

    robot = env.scene[asset_cfg.name]

    # Với terrain plane hiện tại, z=0 là mặt đất.
    foot_height = robot.data.site_pos_w[
        :,
        asset_cfg.site_ids,
        2,
    ]

    in_contact = torch.stack(
        (
            env._left_foot_contact,
            env._right_foot_contact,
        ),
        dim=1,
    )

    # Chỉ reward khi đúng một chân đang đỡ cơ thể.
    # Tránh trường hợp robot ngã, cả hai chân đều rời đất
    # mà vẫn nhận reward.
    single_support = (
        torch.sum(in_contact, dim=1) == 1
    )

    swing_foot = (
        ~in_contact
    ) & single_support.unsqueeze(1)

    normalized_height = torch.clamp(
        foot_height / height_saturation,
        min=0.0,
        max=1.0,
    )

    # sqrt làm reward nhạy hơn với các độ cao thấp.
    # Ví dụ 2 cm: 0.133 -> 0.365 thay vì chỉ 0.133.
    lift_score = torch.sqrt(normalized_height)

    reward = torch.sum(
        lift_score * swing_foot.float(),
        dim=1,
    )

    if command_name is not None:

        command = env.command_manager.get_command(
            command_name
        )

        command_norm = (
            torch.norm(command[:, :2], dim=1)
            + torch.abs(command[:, 2])
        )

        reward *= (
            command_norm > command_threshold
        ).float()

    return reward