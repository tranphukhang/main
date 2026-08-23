import torch


def velocity_command_curriculum(
    env,
    env_ids,
    command_name,
    stages,
):
    """Cập nhật khoảng command vận tốc tiến."""

    del env_ids

    command_cfg = (
        env.command_manager.get_term_cfg(
            command_name
        )
    )

    lin_vel_x = stages[0][
        "lin_vel_x"
    ]

    for stage in stages:
        if (
            env.common_step_counter
            >= stage["step"]
        ):
            lin_vel_x = stage[
                "lin_vel_x"
            ]

    command_cfg.ranges.lin_vel_x = (
        float(lin_vel_x[0]),
        float(lin_vel_x[1]),
    )

    return {
        "command_lin_vel_x_min_mps": (
            torch.tensor(
                lin_vel_x[0],
                device=env.device,
            )
        ),
        "command_lin_vel_x_max_mps": (
            torch.tensor(
                lin_vel_x[1],
                device=env.device,
            )
        ),
    }