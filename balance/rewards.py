from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import mujoco
import mujoco_warp as mjwarp
import numpy as np
import warp as wp

from mjlab.envs import mdp

from balance.observations import (
    pre_push_position_error_xy,
    pre_push_orientation_error,
)


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


# ============================================================
# Pre-push position recovery penalty
# ============================================================

def recovery_position_xy_l2(
    env,
    asset_cfg,
) -> torch.Tensor:
    """
    Phạt sai lệch vị trí XY của root/base so với vị trí
    ngay trước khi push xảy ra.

    Chỉ active khi:
        - pre-push reference đã tồn tại
        - impulse đã kết thúc

    Trong lúc force đang tác động:
        penalty = 0

    Cost:
        (x - x_ref)^2 + (y - y_ref)^2
    """

    error_xy = pre_push_position_error_xy(
        env,
        asset_cfg,
    )

    cost = torch.sum(
        torch.square(error_xy),
        dim=1,
    )

    # Recovery chỉ bắt đầu sau khi push kết thúc
    recovery_active = (
        env._pre_push_pose_valid
        & (~env._push_active)
    )

    return torch.where(
        recovery_active,
        cost,
        torch.zeros_like(cost),
    )


# ============================================================
# Pre-push orientation recovery penalty
# ============================================================

def recovery_orientation_l2(
    env,
    asset_cfg,
) -> torch.Tensor:
    """
    Phạt sai lệch orientation của root/base so với
    orientation ngay trước push.

    Orientation error dùng axis-angle vector:

        e_R = AxisAngle(q_ref^-1 * q)

    Cost:
        ||e_R||^2

    Chỉ active sau khi impulse đã kết thúc.
    """

    error_rotvec = pre_push_orientation_error(
        env,
        asset_cfg,
    )

    cost = torch.sum(
        torch.square(error_rotvec),
        dim=1,
    )

    recovery_active = (
        env._pre_push_pose_valid
        & (~env._push_active)
    )

    return torch.where(
        recovery_active,
        cost,
        torch.zeros_like(cost),
    )


class support_contact_substep:
    """
    Kiểm tra support contact tại từng physics substep.

    Điều kiện support hợp lệ:

    1. Double support:
       - chân trái >= 1 contact
       - chân phải >= 1 contact
       - tổng số contact >= 3

    2. Single support:
       - chỉ chân trái contact và >= 3 contact
       hoặc
       - chỉ chân phải contact và >= 3 contact

    Chỉ tính contact foot <-> terrain
    có normal force > min_normal_force.

    Mỗi lần class được gọi:
        - tính valid_support tại physics step hiện tại
        - cộng kết quả vào accumulator

    Sau 40 physics steps:
        support_contact_reward()
        sẽ lấy giá trị trung bình.
    """

    def __init__(self, cfg, env):

        # =====================================================
        # 1. ID terrain và hai bàn chân
        # =====================================================

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

        # =====================================================
        # 2. Mapping geom -> body
        # =====================================================

        self.geom_bodyid = torch.as_tensor(
            env.sim.mj_model.geom_bodyid,
            device=env.device,
            dtype=torch.long,
        )

        # =====================================================
        # 3. Buffer lấy contact force từ MuJoCo Warp
        # =====================================================

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

        # =====================================================
        # 4. Accumulator trong một control step
        # =====================================================

        self.support_sum = torch.zeros(
            env.num_envs,
            device=env.device,
            dtype=torch.float32,
        )

        self.support_count = torch.zeros(
            env.num_envs,
            device=env.device,
            dtype=torch.float32,
        )

        # Reward function sẽ đọc trực tiếp hai buffer này
        env._support_contact_sum = self.support_sum
        env._support_contact_count = self.support_count

    # =========================================================
    # Physics-substep evaluation
    # =========================================================

    def __call__(
        self,
        env,
        min_normal_force: float = 1.0,
    ) -> torch.Tensor:

        # =====================================================
        # 1. Contact force hiện tại
        # =====================================================

        mjwarp.contact_force(
            env.sim.wp_model,
            env.sim.wp_data,
            self.contact_ids_wp,
            False,
            self.contact_force_wp,
        )

        # Thành phần đầu tiên là normal force
        normal_force = (
            self.contact_force_torch[:, 0]
        )

        # =====================================================
        # 2. Hai geom của mỗi contact
        # =====================================================

        contact_geom = (
            env.sim.data.contact.geom
        )

        geom1 = contact_geom[:, 0].long()
        geom2 = contact_geom[:, 1].long()

        # =====================================================
        # 3. Chỉ lấy contact terrain <-> foot
        # =====================================================

        floor_is_geom1 = (
            geom1 == self.floor_geom_id
        )

        floor_is_geom2 = (
            geom2 == self.floor_geom_id
        )

        # Nếu terrain là geom1 -> foot là geom2
        # Nếu terrain là geom2 -> foot là geom1
        # Nếu không liên quan terrain -> -1
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

        # Tránh index = -1
        safe_foot_geom = torch.clamp(
            foot_geom,
            min=0,
        )

        foot_body = self.geom_bodyid[
            safe_foot_geom
        ]

        # =====================================================
        # 4. Contact hợp lệ theo normal force
        # =====================================================

        valid_force = (
            normal_force > min_normal_force
        )

        # Contact chân trái
        left_contact = (
            valid_foot_geom
            & valid_force
            & (
                foot_body
                == self.left_foot_body_id
            )
        )

        # Contact chân phải
        right_contact = (
            valid_foot_geom
            & valid_force
            & (
                foot_body
                == self.right_foot_body_id
            )
        )

        # =====================================================
        # 5. Contact thuộc environment nào
        # =====================================================

        world_id = (
            env.sim.data.contact.worldid.long()
        )

        valid_world = (
            (world_id >= 0)
            & (world_id < env.num_envs)
        )

        # =====================================================
        # 6. Đếm contact từng chân cho từng environment
        # =====================================================

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

        # =====================================================
        # 7. Xác định support hợp lệ
        # =====================================================

        total_count = (
            left_count + right_count
        )

        # -----------------------------------------------------
        # Double support
        # -----------------------------------------------------

        double_support = (
            (left_count >= 1)
            & (right_count >= 1)
            & (total_count >= 3)
        )

        # -----------------------------------------------------
        # Left single support
        # -----------------------------------------------------

        left_single_support = (
            (left_count >= 3)
            & (right_count == 0)
        )

        # -----------------------------------------------------
        # Right single support
        # -----------------------------------------------------

        right_single_support = (
            (right_count >= 3)
            & (left_count == 0)
        )

        valid_support = (
            double_support
            | left_single_support
            | right_single_support
        )

        valid_support_float = (
            valid_support.float()
        )

        # =====================================================
        # 8. Accumulate physics-substep result
        # =====================================================

        self.support_sum += (
            valid_support_float
        )

        self.support_count += 1.0

        # MetricsManager cũng nhận giá trị của substep này
        return valid_support_float

    # =========================================================
    # Reset khi environment reset
    # =========================================================

    def reset(
        self,
        env_ids: torch.Tensor | None = None,
    ):

        if env_ids is None:

            self.support_sum.zero_()
            self.support_count.zero_()

            return

        self.support_sum[
            env_ids
        ] = 0.0

        self.support_count[
            env_ids
        ] = 0.0


# ============================================================
# Support contact reward
# ============================================================

def support_contact_reward(
    env,
) -> torch.Tensor:
    """
    Reward support được tính bằng tỷ lệ physics substeps
    có support hợp lệ trong một control step.

    Với:
        physics_dt = 0.0005 s
        decimation = 40

    Một control step có 40 physics substeps.

    Reward:
        r_support
        = số physics steps support hợp lệ
          / tổng physics steps

    Ví dụ:
        40 / 40 -> 1.00
        36 / 40 -> 0.90
        20 / 40 -> 0.50
         0 / 40 -> 0.00
    """

    support_sum = (
        env._support_contact_sum
    )

    support_count = (
        env._support_contact_count
    )

    # =========================================================
    # Mean support trong control interval vừa qua
    # =========================================================

    reward = (
        support_sum
        / torch.clamp(
            support_count,
            min=1.0,
        )
    )

    # Clone trước khi reset accumulator
    reward = reward.clone()

    # =========================================================
    # Reset accumulator cho control step tiếp theo
    # =========================================================

    support_sum.zero_()
    support_count.zero_()

    return reward