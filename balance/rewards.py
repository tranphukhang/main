from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import mujoco
import mujoco_warp as mjwarp
import numpy as np
import warp as wp

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


class support_contact_reward:
    """
    Reward khi robot tạo được tập contact chân-ground
    đủ để hình thành support hợp lệ.

    Điều kiện:
    1. Hai chân contact:
       - mỗi chân >= 1 contact
       - tổng contact >= 3

    2. Một chân contact:
       - chân đó phải có >= 3 contact

    Chỉ tính contact có normal force > min_normal_force.
    """

    def __init__(self, cfg, env):

        # -----------------------------------------------------
        # ID terrain và hai bàn chân
        # -----------------------------------------------------

        self.floor_geom_id = mujoco.mj_name2id(
            env.sim.mj_model,
            mujoco.mjtObj.mjOBJ_GEOM,
            "terrain",
        )

        self.left_foot_body_id = mujoco.mj_name2id(
            env.sim.mj_model,
            mujoco.mjtObj.mjOBJ_BODY,
            "robot/feet_left",
        )

        self.right_foot_body_id = mujoco.mj_name2id(
            env.sim.mj_model,
            mujoco.mjtObj.mjOBJ_BODY,
            "robot/feet_right",
        )

        # -----------------------------------------------------
        # Mapping geom -> body
        # -----------------------------------------------------

        self.geom_bodyid = torch.as_tensor(
            env.sim.mj_model.geom_bodyid,
            device=env.device,
            dtype=torch.long,
        )

        # -----------------------------------------------------
        # Buffer lấy contact force
        # -----------------------------------------------------

        self.max_contacts = (
            env.sim.data.contact.pos.shape[0]
        )

        self.contact_ids_wp = wp.array(
            np.arange(
                self.max_contacts,
                dtype=np.int32,
            ),
            dtype=wp.int32,
            device=env.sim.wp_device,
        )

        self.contact_force_wp = wp.zeros(
            self.max_contacts,
            dtype=wp.spatial_vector,
            device=env.sim.wp_device,
        )

        self.contact_force_torch = wp.to_torch(
            self.contact_force_wp
        )

    def __call__(
        self,
        env,
        min_normal_force: float = 1.0,
    ) -> torch.Tensor:

        # -----------------------------------------------------
        # 1. Tính contact force hiện tại
        # -----------------------------------------------------

        mjwarp.contact_force(
            env.sim.wp_model,
            env.sim.wp_data,
            self.contact_ids_wp,
            False,
            self.contact_force_wp,
        )

        normal_force = (
            self.contact_force_torch[:, 0]
        )

        # -----------------------------------------------------
        # 2. Hai geom tạo mỗi contact
        # -----------------------------------------------------

        contact_geom = (
            env.sim.data.contact.geom
        )

        geom1 = contact_geom[:, 0].long()
        geom2 = contact_geom[:, 1].long()

        # -----------------------------------------------------
        # 3. Chỉ lấy contact terrain <-> foot
        # -----------------------------------------------------

        floor_is_geom1 = (
            geom1 == self.floor_geom_id
        )

        floor_is_geom2 = (
            geom2 == self.floor_geom_id
        )

        foot_geom = torch.where(
            floor_is_geom1,
            geom2,
            torch.where(
                floor_is_geom2,
                geom1,
                torch.full_like(
                    geom1,
                    -1,
                ),
            ),
        )

        valid_foot_geom = (
            foot_geom >= 0
        )

        safe_foot_geom = torch.clamp(
            foot_geom,
            min=0,
        )

        foot_body = self.geom_bodyid[
            safe_foot_geom
        ]

        # -----------------------------------------------------
        # 4. Xác định contact chân trái / phải
        # -----------------------------------------------------

        valid_force = (
            normal_force > min_normal_force
        )

        left_contact = (
            valid_foot_geom
            & valid_force
            & (
                foot_body
                == self.left_foot_body_id
            )
        )

        right_contact = (
            valid_foot_geom
            & valid_force
            & (
                foot_body
                == self.right_foot_body_id
            )
        )

        # -----------------------------------------------------
        # 5. Contact thuộc environment nào
        # -----------------------------------------------------

        world_id = (
            env.sim.data.contact.worldid.long()
        )

        valid_world = (
            (world_id >= 0)
            & (world_id < env.num_envs)
        )

        # -----------------------------------------------------
        # 6. Đếm contact cho từng chân, từng env
        # -----------------------------------------------------

        left_count = torch.zeros(
            env.num_envs,
            device=env.device,
        )

        right_count = torch.zeros(
            env.num_envs,
            device=env.device,
        )

        left_mask = (
            left_contact
            & valid_world
        )

        right_mask = (
            right_contact
            & valid_world
        )

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
        # 7. Điều kiện support
        # -----------------------------------------------------

        total_count = (
            left_count + right_count
        )

        # Hai chân cùng tiếp xúc
        double_support = (
            (left_count >= 1)
            & (right_count >= 1)
            & (total_count >= 3)
        )

        # Chỉ chân trái
        left_single_support = (
            (left_count >= 3)
            & (right_count == 0)
        )

        # Chỉ chân phải
        right_single_support = (
            (right_count >= 3)
            & (left_count == 0)
        )

        valid_support = (
            double_support
            | left_single_support
            | right_single_support
        )

        return valid_support.float()