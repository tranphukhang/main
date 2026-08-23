from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import torch
import mujoco

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.utils.wrappers import VideoRecorder

from balance.balance_env_cfg import balance_env_cfg
from balance.ppo_cfg import balance_ppo_runner_cfg
from evaluation.cop_support_logger import (
    CopSupportLogger,
)


# ============================================================
# Configuration
# ============================================================

CHECKPOINT = Path(
    "logs/rsl_rl/balance_v1/"
    "2026-08-21_10-22-44_baseline_v1/"
    "model_599.pt"
)

EPISODE_LENGTH_S = 12.0
TEST_FORCE_N = 15.0
COUNT_DOWN_S = 3.0


def main():

    # --------------------------------------------------------
    # 1. Kiểm tra checkpoint
    # --------------------------------------------------------

    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Không tìm thấy checkpoint:\n{CHECKPOINT}"
        )

    print(f"Loading checkpoint:\n{CHECKPOINT}")

    # --------------------------------------------------------
    # 2. Cấu hình môi trường đánh giá
    # --------------------------------------------------------

    env_cfg = balance_env_cfg()
    env_cfg.viewer.width = 1280
    env_cfg.viewer.height = 720
    env_cfg.scene.num_envs = 1
    env_cfg.episode_length_s = EPISODE_LENGTH_S

    # Không dùng curriculum khi đánh giá
    env_cfg.curriculum = {}

    # Đánh giá trực tiếp với lực tối đa 15 N
    env_cfg.events["body_impulse"].params["force_range"] = (
        TEST_FORCE_N,
        TEST_FORCE_N,
    )
    env_cfg.events["body_impulse"].params["cooldown_s"] = (
        COUNT_DOWN_S,
        COUNT_DOWN_S,
    )

    device = (
        "cuda:0"
        if torch.cuda.is_available()
        else "cpu"
    )

    # --------------------------------------------------------
    # 3. Khởi tạo môi trường
    # --------------------------------------------------------

    env = ManagerBasedRlEnv(
        cfg=env_cfg,
        device=device,
        render_mode="rgb_array",
    )

    # ========================================================
    # Off-screen visualization
    # ========================================================

    offscreen_renderer = env._offline_renderer
    renderer_option = offscreen_renderer._opt
    render_model = offscreen_renderer._model

    # --------------------------------------------------------
    # Hiển thị contact point và contact force trong video
    # --------------------------------------------------------

    renderer_option.flags[
        mujoco.mjtVisFlag.mjVIS_CONTACTPOINT
    ] = 1

    renderer_option.flags[
        mujoco.mjtVisFlag.mjVIS_CONTACTFORCE
    ] = 1

    # Hiển thị hệ trục tọa độ world
    renderer_option.frame = (
        mujoco.mjtFrame.mjFRAME_WORLD.value
    )

    # Kích thước contact point
    render_model.vis.scale.contactwidth = 0.9
    render_model.vis.scale.contactheight = 0.3

    # Độ dày và chiều dài vector contact force
    render_model.vis.scale.forcewidth = 0.3
    render_model.vis.map.force = 0.015

    # Kích thước hệ trục tọa độ
    render_model.vis.scale.framelength = 3.0
    render_model.vis.scale.framewidth = 0.3

    num_steps = int(
        env_cfg.episode_length_s / env.step_dt
    )

    # --------------------------------------------------------
    # 4. Folder lưu video
    # --------------------------------------------------------

    timestamp = datetime.now().strftime(
        "%Y-%m-%d_%H-%M-%S"
    )

    video_dir = (
        Path("balance/eval/logs")
        / timestamp
    )

    video_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(f"Video folder:\n{video_dir}")

    cop_logger = CopSupportLogger(
        env=env,
        output_dir=video_dir,
        env_idx=0,
        min_normal_force=1.0,
    )

    cop_logger.install_physics_hook()

    # --------------------------------------------------------
    # 5. Gắn bộ ghi video
    # --------------------------------------------------------

    env = VideoRecorder(
        env,
        video_folder=video_dir,
        step_trigger=lambda step: step == 0,
        video_length=num_steps,
        name_prefix="balance_eval",
        disable_logger=False,
    )

    # --------------------------------------------------------
    # 6. Load PPO policy
    # --------------------------------------------------------

    agent_cfg = balance_ppo_runner_cfg()

    env = RslRlVecEnvWrapper(
        env,
        clip_actions=agent_cfg.clip_actions,
    )

    runner = MjlabOnPolicyRunner(
        env,
        asdict(agent_cfg),
        device=device,
    )

    runner.load(
        str(CHECKPOINT),
        load_cfg={"actor": True},
        strict=True,
        map_location=device,
    )

    policy = runner.get_inference_policy(
        device=device
    )

    print("Policy loaded.")

    # --------------------------------------------------------
    # 7. Chạy đánh giá
    # --------------------------------------------------------

    try:
        with torch.no_grad():

            for step in range(num_steps):

                obs = env.get_observations()
                actions = policy(obs)
                env.step(actions)

                if step % 100 == 0:
                    print(
                        f"Step {step}/{num_steps}"
                    )

    finally:
        cop_logger.remove_physics_hook()
        cop_logger.finalize()
        env.close()

    print("Evaluation finished.")


if __name__ == "__main__":
    main()