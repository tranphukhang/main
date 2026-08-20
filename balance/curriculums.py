import torch


def push_force_curriculum(
    env,
    env_ids,
    event_name,
    stages,
):

    del env_ids

    event_cfg = env.event_manager.get_term_cfg(
        event_name
    )

    max_force = stages[0]["max_force"]

    for stage in stages:
        if env.common_step_counter >= stage["step"]:
            max_force = stage["max_force"]

    event_cfg.params["force_range"] = (
        -max_force,
        max_force,
    )

    return {
        "max_push_force_N": torch.tensor(
            max_force,
            device=env.device,
        )
    }