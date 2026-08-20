from dataclasses import asdict
from pathlib import Path

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import (
    MjlabOnPolicyRunner,
    RslRlVecEnvWrapper,
)
from mjlab.utils.wrappers import VideoRecorder

from standing.standing_env_cfg import standing_env_cfg
from standing.ppo_cfg import standing_ppo_runner_cfg
from standing.joint_plotter import JointPlotter


def main():

    IMPULSE_DIRECTION = "y-"

    direction_map = {
        "x+": (1.0, 0.0, 0.0),
        "x-": (-1.0, 0.0, 0.0),
        "y+": (0.0, 1.0, 0.0),
        "y-": (0.0, -1.0, 0.0),
    }

    # =========================================================
    # 1. Checkpoint
    # =========================================================

    checkpoint = Path(
        "logs/rsl_rl/standing_v1/"
        "2026-08-15_15-46-11_push_v3/"
        "model_2595.pt"
    )

    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Không tìm thấy checkpoint: {checkpoint}"
        )

    print(f"Loading checkpoint: {checkpoint}")

    # =========================================================
    # 2. Environment config
    # =========================================================

    env_cfg = standing_env_cfg()

    if IMPULSE_DIRECTION not in direction_map:
        raise ValueError(
            f"Hướng xung lực không hợp lệ: "
            f"{IMPULSE_DIRECTION}"
        )

    env_cfg.events[
        "body_impulse"
    ].params[
        "force_direction"
    ] = direction_map[
        IMPULSE_DIRECTION
    ]

    env_cfg.scene.num_envs = 1
    env_cfg.episode_length_s = 8.0

    # Không reset vì timeout
    env_cfg.terminations.pop(
        "time_out",
        None,
    )

    # Xung lực mỗi 4 s
    env_cfg.events[
        "body_impulse"
    ].params[
        "cooldown_s"
    ] = (4.0, 4.0)

    env_cfg.events["body_impulse"].params["force_range"] = (25, 25)

    # =========================================================
    # 3. Device
    # =========================================================

    device = (
        "cuda:0"
        if torch.cuda.is_available()
        else "cpu"
    )

    agent_cfg = standing_ppo_runner_cfg()

    # =========================================================
    # 4. Output
    # =========================================================

    output_dir = Path(
        "logs/standing_eval"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # =========================================================
    # 5. Headless environment
    # =========================================================

    env = ManagerBasedRlEnv(
        cfg=env_cfg,
        device=device,

        # Không mở GUI.
        # Chỉ render frame để lưu video.
        render_mode="rgb_array",
    )

    num_steps = int(
        8.0 / env.step_dt
    )

    print(
        f"Running headless evaluation: "
        f"{num_steps} steps"
    )

    # =========================================================
    # 6. Video recorder
    # =========================================================

    env = VideoRecorder(
        env,
        video_folder=output_dir,

        # Bắt đầu record ngay step 0
        step_trigger=lambda step: step == 0,

        # 400 frame
        video_length=num_steps,

        name_prefix="standing_eval",

        disable_logger=False,
    )

    # =========================================================
    # 7. RL wrapper
    # =========================================================

    env = RslRlVecEnvWrapper(
        env,
        clip_actions=agent_cfg.clip_actions,
    )

    # =========================================================
    # 8. Load trained policy
    # =========================================================

    runner = MjlabOnPolicyRunner(
        env,
        asdict(agent_cfg),
        device=device,
    )

    runner.load(
        str(checkpoint),
        load_cfg={
            "actor": True,
        },
        strict=True,
        map_location=device,
    )

    policy = runner.get_inference_policy(
        device=device
    )

    # =========================================================
    # 9. Data logger
    # =========================================================

    logger = JointPlotter(
        env=env.unwrapped,
        env_idx=0,
    )

    logger.install_physics_hook()

    # =========================================================
    # Contact visualization cho standing_eval video
    # =========================================================

    base_env = env.unwrapped

    original_update_visualizers = (
        base_env.update_visualizers
    )


    def update_visualizers_with_contacts(
        visualizer,
    ):

        # Giữ các visualization hiện tại,
        # ví dụ external impulse
        original_update_visualizers(
            visualizer
        )

        # Thêm contact point + contact force
        logger.add_contact_visualization(
            visualizer
        )


    base_env.update_visualizers = (
        update_visualizers_with_contacts
    )

    # =========================================================
    # 10. Headless simulation
    # =========================================================

    try:

        with torch.no_grad():

            for _ in range(num_steps):

                obs = env.get_observations()

                actions = policy(obs)

                env.step(actions)

        # Sau 8 s mới xử lý dữ liệu
        logger.finalize()

    finally:
        base_env.update_visualizers = (
            original_update_visualizers
        )

        logger.remove_physics_hook()

        env.close()

    print("Evaluation finished.")


if __name__ == "__main__":
    main()