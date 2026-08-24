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
from mjlab.utils.wrappers import VideoRecorder

from evaluation.cop_support_logger import (
    CopSupportLogger,
)
from velocity.ppo_cfg import velocity_ppo_runner_cfg
from velocity.velocity_env_cfg import velocity_env_cfg


# ============================================================
# Evaluation configuration
# ============================================================

CHECKPOINT = Path(
    "logs/rsl_rl/velocity_v1/"
    "2026-08-24_14-13-14_baseline_v1/"
    "model_1998.pt"
)

EPISODE_LENGTH_S = 15.0

# Đổi ba giá trị này để test các case khác.
#
# Đi tới:  ( 0.15,  0.0, 0.0)
# Đi lùi:  (-0.15,  0.0, 0.0)
# Sang trái:  (0.0,  0.15, 0.0)
# Sang phải:  (0.0, -0.15, 0.0)
TEST_COMMAND = (
    0.15,
    0.0,
    0.0,
)

CASE_NAME = "forward_x_0p15"


# ============================================================
# Environment
# ============================================================

def build_eval_env() -> ManagerBasedRlEnv:
    env_cfg = velocity_env_cfg()

    env_cfg.viewer.width = 1280
    env_cfg.viewer.height = 720

    env_cfg.scene.num_envs = 1
    env_cfg.episode_length_s = EPISODE_LENGTH_S

    # Không dùng curriculum khi evaluation.
    env_cfg.curriculum = {}

    # Không cho command sampler đổi command trong episode.
    twist_cfg = env_cfg.commands["twist"]

    twist_cfg.resampling_time_range = (
        EPISODE_LENGTH_S + 1.0,
        EPISODE_LENGTH_S + 1.0,
    )

    # Không tạo standing command ngẫu nhiên.
    twist_cfg.rel_standing_envs = 0.0

    device = (
        "cuda:0"
        if torch.cuda.is_available()
        else "cpu"
    )

    return ManagerBasedRlEnv(
        cfg=env_cfg,
        device=device,
        render_mode="rgb_array",
    )


def lock_test_command(
    env: ManagerBasedRlEnv,
) -> None:
    """Ghi đè command sampler bằng command evaluation cố định."""

    twist_term = env.command_manager.get_term(
        "twist"
    )

    fixed_command = torch.tensor(
        TEST_COMMAND,
        device=env.device,
        dtype=twist_term.vel_command_b.dtype,
    )

    twist_term.vel_command_b[:] = fixed_command

    # Bảo đảm command không bị chuyển sang standing.
    twist_term.is_standing_env[:] = False


# ============================================================
# Main
# ============================================================

def main():

    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Không tìm thấy checkpoint:\n{CHECKPOINT}"
        )

    print(f"Loading checkpoint:\n{CHECKPOINT}")

    env = build_eval_env()
    cop_logger = None

    try:
        # Reset manager trước, sau đó mới khóa command.
        env.reset()
        lock_test_command(env)

        actual_command = env.command_manager.get_command(
            "twist"
        )[0]

        print(
            "Fixed evaluation command: "
            f"{actual_command.cpu().tolist()}"
        )

        # ====================================================
        # Off-screen visualization
        # ====================================================

        offscreen_renderer = env._offline_renderer
        renderer_option = offscreen_renderer._opt
        render_model = offscreen_renderer._model

        renderer_option.flags[
            mujoco.mjtVisFlag.mjVIS_CONTACTPOINT
        ] = 1

        renderer_option.flags[
            mujoco.mjtVisFlag.mjVIS_CONTACTFORCE
        ] = 1

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
            EPISODE_LENGTH_S / env.step_dt
        )

        # ====================================================
        # Output folder
        # ====================================================

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

        print(f"Output folder:\n{video_dir}")

        # ====================================================
        # COP, support polygon, COM và Capture Point
        # ====================================================

        cop_logger = CopSupportLogger(
            env=env,
            output_dir=video_dir,
            env_idx=0,
            min_normal_force=1.0,
        )

        cop_logger.install_physics_hook()

        # ====================================================
        # Video recorder
        # ====================================================

        env = VideoRecorder(
            env,
            video_folder=video_dir,
            step_trigger=lambda step: step == 0,
            video_length=num_steps,
            name_prefix=f"velocity_{CASE_NAME}",
            disable_logger=False,
        )

        # ====================================================
        # Load PPO policy
        # ====================================================

        agent_cfg = velocity_ppo_runner_cfg()

        env = RslRlVecEnvWrapper(
            env,
            clip_actions=agent_cfg.clip_actions,
        )

        runner = MjlabOnPolicyRunner(
            env,
            asdict(agent_cfg),
            device=env.device,
        )

        runner.load(
            str(CHECKPOINT),
            load_cfg={"actor": True},
            strict=True,
            map_location=env.device,
        )

        policy = runner.get_inference_policy(
            device=env.device
        )

        print("Policy loaded.")

        # ====================================================
        # Evaluation rollout
        # ====================================================

        with torch.no_grad():

            for step in range(num_steps):

                obs = env.get_observations()
                actions = policy(obs)

                _, _, dones, extras = env.step(
                    actions
                )

                if step % 100 == 0:
                    print(
                        f"Step {step}/{num_steps}"
                    )

                # Dừng nếu robot ngã hoặc episode timeout.
                if bool(dones[0].item()):

                    time_outs = extras.get(
                        "time_outs",
                        torch.zeros_like(
                            dones,
                            dtype=torch.bool,
                        ),
                    )

                    if bool(time_outs[0].item()):
                        print("Episode completed by timeout.")
                    else:
                        print(
                            "Robot terminated early."
                        )

                    break

    finally:
        if cop_logger is not None:
            cop_logger.remove_physics_hook()
            cop_logger.finalize()

        env.close()

    print("Evaluation finished.")


if __name__ == "__main__":
    main()