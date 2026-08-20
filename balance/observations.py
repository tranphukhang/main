from __future__ import annotations

import torch


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