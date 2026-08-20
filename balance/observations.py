from __future__ import annotations

import torch


# ============================================================
# Initial position error XY
# ============================================================

def initial_position_error_xy(
    env,
    asset_cfg,
) -> torch.Tensor:
    """
    Sai lệch vị trí XY của root/base so với vị trí ban đầu
    của robot trong environment.

    Error:

        [x - x0,
         y - y0]

    Target position:

        default_root_position + env_origin

    Observation này active trong toàn bộ episode:
        - trước push
        - trong push
        - sau push
    """

    robot = env.scene[
        asset_cfg.name
    ]

    # --------------------------------------------------------
    # Vị trí XY hiện tại trong world frame
    # --------------------------------------------------------

    current_xy = (
        robot.data.root_link_pos_w[
            :,
            :2,
        ]
    )

    # --------------------------------------------------------
    # Default root state của robot
    # --------------------------------------------------------

    default_root_state = (
        robot.data.default_root_state
    )

    assert default_root_state is not None

    # --------------------------------------------------------
    # Target XY của từng environment trong world frame
    #
    # target = default position + env origin
    # --------------------------------------------------------

    target_xy = (
        default_root_state[
            :,
            :2,
        ]
        + env.scene.env_origins[
            :,
            :2,
        ]
    )

    # --------------------------------------------------------
    # Position error
    # --------------------------------------------------------

    error_xy = (
        current_xy
        - target_xy
    )

    return error_xy