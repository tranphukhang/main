from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import mujoco
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import (
    MjlabOnPolicyRunner,
    RslRlVecEnvWrapper,
)
from mjlab.tasks.velocity.mdp import (
    UniformVelocityCommandCfg,
)
from mjlab.utils.wrappers import VideoRecorder

from evaluation.velocity_cop_support_logger import (
    VelocityCopSupportLogger,
)
from velocity.ppo_cfg import velocity_ppo_runner_cfg
from velocity.velocity_env_cfg import velocity_env_cfg


# ============================================================
# Configuration
# ============================================================

CHECKPOINT = Path(
    "logs/rsl_rl/velocity_v1/"
    "2026-08-24_14-13-14_baseline_v1/"
    "model_1998.pt"
)

EPISODE_LENGTH_S = 15.0

# Đi tới:
# TEST_COMMAND = (0.15, 0.0, 0.0)
#
# Đi lùi:
# TEST_COMMAND = (-0.15, 0.0, 0.0)
#
# Sang trái:
# TEST_COMMAND = (0.0, 0.15, 0.0)
#
# Sang phải:
# TEST_COMMAND = (0.0, -0.15, 0.0)

TEST_COMMAND = (
    0.15,
    0.0,
    0.0,
)

CASE_NAME = "forward_x_0p15"


def main():

    # --------------------------------------------------------
    # 1. Kiểm tra checkpoint
    # --------------------------------------------------------

    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Không tìm thấy checkpoint:\n"
            f"{CHECKPOINT}"
        )

    print(
        f"Loading checkpoint:\n"
        f"{CHECKPOINT}"
    )

    # --------------------------------------------------------
    # 2. Cấu hình môi trường đánh giá
    # --------------------------------------------------------

    env_cfg = velocity_env_cfg()

    env_cfg.viewer.width = 1280
    env_cfg.viewer.height = 720

    env_cfg.scene.num_envs = 1
    env_cfg.episode_length_s = (
        EPISODE_LENGTH_S
    )

    # Không dùng curriculum khi evaluation.
    env_cfg.curriculum = {}

    # Dùng UniformVelocityCommandCfg có sẵn
    # của MjLab.
    #
    # Min = max nên command luôn cố định.
    #
    # MjLab bắt buộc phải khai báo
    # resampling_time_range. Ta đặt khoảng này
    # dài hơn episode nên command không được
    # resample trong quá trình evaluation.

    env_cfg.commands["twist"] = (
        UniformVelocityCommandCfg(
            entity_name="robot",

            resampling_time_range=(
                EPISODE_LENGTH_S + 1.0,
                EPISODE_LENGTH_S + 1.0,
            ),

            rel_standing_envs=0.0,

            rel_heading_envs=0.0,

            heading_command=False,

            debug_vis=True,

            ranges=(
                UniformVelocityCommandCfg.Ranges(
                    lin_vel_x=(
                        TEST_COMMAND[0],
                        TEST_COMMAND[0],
                    ),

                    lin_vel_y=(
                        TEST_COMMAND[1],
                        TEST_COMMAND[1],
                    ),

                    ang_vel_z=(
                        TEST_COMMAND[2],
                        TEST_COMMAND[2],
                    ),
                )
            ),
        )
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

    actual_command = (
        env.command_manager.get_command(
            "twist"
        )[0]
    )

    print(
        "Fixed evaluation command: "
        f"{actual_command.cpu().tolist()}"
    )

    # ========================================================
    # Off-screen visualization
    # ========================================================

    offscreen_renderer = (
        env._offline_renderer
    )

    renderer_option = (
        offscreen_renderer._opt
    )

    render_model = (
        offscreen_renderer._model
    )

    # Hiển thị contact point.
    renderer_option.flags[
        mujoco.mjtVisFlag.mjVIS_CONTACTPOINT
    ] = 1

    # Hiển thị contact force.
    renderer_option.flags[
        mujoco.mjtVisFlag.mjVIS_CONTACTFORCE
    ] = 1

    # Hiển thị hệ tọa độ world.
    renderer_option.frame = (
        mujoco.mjtFrame.mjFRAME_WORLD.value
    )

    render_model.vis.scale.contactwidth = 0.9
    render_model.vis.scale.contactheight = 0.3

    render_model.vis.scale.forcewidth = 0.3
    render_model.vis.map.force = 0.015

    render_model.vis.scale.framelength = 3.0
    render_model.vis.scale.framewidth = 0.3

    num_steps = int(
        env_cfg.episode_length_s
        / env.step_dt
    )

    # --------------------------------------------------------
    # 4. Folder lưu video
    # --------------------------------------------------------

    timestamp = datetime.now().strftime(
        "%Y-%m-%d_%H-%M-%S"
    )

    video_dir = (
        Path("velocity/eval/logs")
        / CASE_NAME
        / timestamp
    )

    video_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"Video folder:\n"
        f"{video_dir}"
    )

    velocity_logger = (
        VelocityCopSupportLogger(
            env=env,
            output_dir=video_dir,
            command_name="twist",
            env_idx=0,
            min_normal_force=1.0,
        )
    )

    velocity_logger.install_physics_hook()

    # --------------------------------------------------------
    # 5. Gắn bộ ghi video
    # --------------------------------------------------------

    env = VideoRecorder(
        env,
        video_folder=video_dir,
        step_trigger=(
            lambda step: step == 0
        ),
        video_length=num_steps,
        name_prefix=(
            f"velocity_{CASE_NAME}"
        ),
        disable_logger=False,
    )

    # --------------------------------------------------------
    # 6. Load PPO policy
    # --------------------------------------------------------

    agent_cfg = (
        velocity_ppo_runner_cfg()
    )

    env = RslRlVecEnvWrapper(
        env,
        clip_actions=(
            agent_cfg.clip_actions
        ),
    )

    runner = MjlabOnPolicyRunner(
        env,
        asdict(agent_cfg),
        device=device,
    )

    runner.load(
        str(CHECKPOINT),
        load_cfg={
            "actor": True,
        },
        strict=True,
        map_location=device,
    )

    policy = (
        runner.get_inference_policy(
            device=device
        )
    )

    print("Policy loaded.")

    # --------------------------------------------------------
    # 7. Chạy evaluation
    # --------------------------------------------------------

    try:
        with torch.no_grad():

            for step in range(
                num_steps
            ):
                obs = (
                    env.get_observations()
                )

                actions = policy(obs)

                env.step(actions)

                if step % 100 == 0:
                    print(
                        f"Step "
                        f"{step}/{num_steps}"
                    )

    finally:
        velocity_logger.remove_physics_hook()
        velocity_logger.finalize()
        env.close()

    print("Evaluation finished.")


if __name__ == "__main__":
    main()