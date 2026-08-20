from __future__ import annotations

import torch

from mjlab.utils.lab_api.math import (
    quat_conjugate,
    quat_mul,
    axis_angle_from_quat,
)


# ============================================================
# Pre-push position error XY
# ============================================================

def pre_push_position_error_xy(
    env,
    asset_cfg,
) -> torch.Tensor:
    """
    Sai lệch vị trí XY của root/base so với vị trí ngay trước push.

    Output:
        [x - x_ref,
         y - y_ref]

    Trước khi push xảy ra:
        output = [0, 0]

    Không xét z để robot vẫn được phép crouch / thay đổi độ cao
    khi thực hiện chiến lược giữ thăng bằng.
    """

    robot = env.scene[
        asset_cfg.name
    ]

    # Vị trí root hiện tại trong world frame
    current_pos = (
        robot.data.root_link_pos_w
    )

    # Reference được lưu ngay trước push
    ref_pos = (
        env._pre_push_pos_w
    )

    # Chỉ lấy XY
    error_xy = (
        current_pos[:, :2]
        - ref_pos[:, :2]
    )

    # Trước khi có push thì chưa có reference
    valid = (
        env._pre_push_pose_valid
    )

    error_xy = torch.where(
        valid.unsqueeze(-1),
        error_xy,
        torch.zeros_like(
            error_xy
        ),
    )

    return error_xy


# ============================================================
# Pre-push orientation error
# ============================================================

def pre_push_orientation_error(
    env,
    asset_cfg,
) -> torch.Tensor:
    """
    Sai lệch orientation của root/base so với orientation
    ngay trước push.

    Quaternion:

        q_error = q_ref^{-1} * q_current

    Sau đó chuyển thành axis-angle vector.

    Output:
        [e_rx, e_ry, e_rz]

    Độ lớn vector:

        ||e_R||

    chính là góc sai lệch orientation [rad].

    Trước khi push xảy ra:
        output = [0, 0, 0]
    """

    robot = env.scene[
        asset_cfg.name
    ]

    # Quaternion hiện tại
    current_quat = (
        robot.data.root_link_quat_w
    )

    # Quaternion reference trước push
    ref_quat = (
        env._pre_push_quat_w
    )

    # --------------------------------------------------------
    # Relative quaternion:
    #
    # q_error = q_ref^(-1) * q_current
    # --------------------------------------------------------

    error_quat = quat_mul(
        quat_conjugate(
            ref_quat
        ),
        current_quat,
    )

    # --------------------------------------------------------
    # Quaternion -> axis-angle vector
    # --------------------------------------------------------

    error_rotvec = (
        axis_angle_from_quat(
            error_quat
        )
    )

    # --------------------------------------------------------
    # Chưa có pre-push reference -> observation = 0
    # --------------------------------------------------------

    valid = (
        env._pre_push_pose_valid
    )

    error_rotvec = torch.where(
        valid.unsqueeze(-1),
        error_rotvec,
        torch.zeros_like(
            error_rotvec
        ),
    )

    return error_rotvec