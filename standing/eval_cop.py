from pathlib import Path

import mjlab.scripts.play as play_script
from mjlab.scripts.play import PlayConfig
from mjlab.tasks.registry import register_mjlab_task

from standing.standing_env_cfg import standing_env_cfg
from standing.ppo_cfg import standing_ppo_runner_cfg


TASK_ID = "Mjlab-Standing-COP-Eval"


def main():

    # =========================================================
    # 1. Chọn model đã train
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
    # 2. Tạo environment riêng cho evaluation
    # =========================================================

    eval_env_cfg = standing_env_cfg()

    # Chỉ chạy 1 robot
    eval_env_cfg.scene.num_envs = 1

    # Cho episode dài hơn để quan sát
    eval_env_cfg.episode_length_s = 20.0

    # Giữ xung lực lặp lại mỗi 2 giây để test trước
    eval_env_cfg.events["body_impulse"].params[
        "cooldown_s"
    ] = (4.0, 4.0)

    # =========================================================
    # 3. Register task evaluation
    # =========================================================

    register_mjlab_task(
        task_id=TASK_ID,
        env_cfg=eval_env_cfg,
        play_env_cfg=eval_env_cfg,
        rl_cfg=standing_ppo_runner_cfg(),
    )

    # =========================================================
    # 4. Chạy model
    # =========================================================

    play_cfg = PlayConfig(
        checkpoint_file=str(checkpoint),
        viewer="native",
    )

    play_script.run_play(
        TASK_ID,
        play_cfg,
    )


if __name__ == "__main__":
    main()