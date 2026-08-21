from dataclasses import asdict
from pathlib import Path
from datetime import datetime

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import (
    MjlabOnPolicyRunner,
    RslRlVecEnvWrapper,
)

from mjlab.utils.wrappers import VideoRecorder


from balance.balance_env_cfg import balance_env_cfg
from balance.ppo_cfg import balance_ppo_runner_cfg


# ============================================================
# Configuration
# ============================================================

CHECKPOINT = Path(
    "logs/rsl_rl/balance_v1/"
    "2026-08-21_10-22-44_baseline_v1/"
    "model_599.pt"
)

VIDEO_FPS = 50


# ============================================================
# Main
# ============================================================

def main():

    # --------------------------------------------------------
    # 1. Checkpoint
    # --------------------------------------------------------

    checkpoint = CHECKPOINT

    if not checkpoint.exists():

        raise FileNotFoundError(
            f"Checkpoint not found:\n{checkpoint}"
        )

    print(
        f"Loading checkpoint:\n{checkpoint}"
    )


    # --------------------------------------------------------
    # 2. Environment
    # --------------------------------------------------------

    env_cfg = balance_env_cfg()

    env_cfg.scene.num_envs = 1

    env_cfg.episode_length_s = 10.0

    device = (
        "cuda:0"
        if torch.cuda.is_available()
        else "cpu"
    )


    # --------------------------------------------------------
    # 3. Create environment
    # --------------------------------------------------------

    env = ManagerBasedRlEnv(
        cfg=env_cfg,

        device=device,

        # cần cho VideoRecorder
        render_mode="rgb_array",
    )


    num_steps = int(
        env_cfg.episode_length_s
        /
        env.step_dt
    )


    print(
        f"Simulation steps: {num_steps}"
    )


    # --------------------------------------------------------
    # 4. Video folder
    # --------------------------------------------------------

    timestamp = datetime.now().strftime(
        "%Y-%m-%d_%H-%M-%S"
    )


    video_dir = (
        Path("balance/logs")
        /
        timestamp
        /
        "video"
    )


    video_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    print(
        f"Video folder:\n{video_dir}"
    )


    # --------------------------------------------------------
    # 5. Attach video recorder
    # --------------------------------------------------------

    env = VideoRecorder(
        env,

        video_folder=video_dir,

        # record ngay từ frame đầu
        step_trigger=lambda step: step == 0,

        video_length=num_steps,

        name_prefix="balance_eval",

        disable_logger=False,
    )


    # --------------------------------------------------------
    # 6. PPO wrapper
    # --------------------------------------------------------

    agent_cfg = balance_ppo_runner_cfg()


    env = RslRlVecEnvWrapper(
        env,

        clip_actions=
        agent_cfg.clip_actions,
    )


    # --------------------------------------------------------
    # 7. Load policy
    # --------------------------------------------------------

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


    print(
        "Policy loaded."
    )


    # --------------------------------------------------------
    # 8. Simulation
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

        env.close()


    print(
        "Evaluation finished."
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":

    main()