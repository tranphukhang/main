import mujoco
import numpy as np


def compute_cop(
    mj_model,
    contact_pos_history,
    contact_geom_history,
    normal_force_history,
    nacon_history,
    min_normal_force=1.0,
):

    # =========================================================
    # 1. Tìm ID terrain và hai bàn chân
    # =========================================================

    floor_geom_id = mujoco.mj_name2id(
        mj_model,
        mujoco.mjtObj.mjOBJ_GEOM,
        "terrain",
    )

    left_foot_body_id = mujoco.mj_name2id(
        mj_model,
        mujoco.mjtObj.mjOBJ_BODY,
        "robot/feet_left",
    )

    right_foot_body_id = mujoco.mj_name2id(
        mj_model,
        mujoco.mjtObj.mjOBJ_BODY,
        "robot/feet_right",
    )

    if floor_geom_id < 0:
        raise RuntimeError(
            "Không tìm thấy geom 'terrain'."
        )

    if left_foot_body_id < 0:
        raise RuntimeError(
            "Không tìm thấy body 'robot/feet_left'."
        )

    if right_foot_body_id < 0:
        raise RuntimeError(
            "Không tìm thấy body 'robot/feet_right'."
        )

    # =========================================================
    # 2. Khởi tạo output
    # =========================================================

    num_samples = len(
        contact_pos_history
    )

    cop_history = np.full(
        (num_samples, 2),
        np.nan,
        dtype=float,
    )

    total_normal_force = np.zeros(
        num_samples,
        dtype=float,
    )

    # =========================================================
    # 3. Tính COP tại từng physics sample
    # =========================================================

    for k in range(num_samples):

        weighted_position = np.zeros(
            2,
            dtype=float,
        )

        total_force = 0.0

        ncon = int(
            nacon_history[k]
        )

        for j in range(ncon):

            geom1 = int(
                contact_geom_history[
                    k, j, 0
                ]
            )

            geom2 = int(
                contact_geom_history[
                    k, j, 1
                ]
            )

            normal_force = float(
                normal_force_history[
                    k, j
                ]
            )

            # Bỏ contact gần như không chịu tải
            if normal_force <= min_normal_force:
                continue

            # Chỉ xét contact bàn chân <-> terrain
            if geom1 == floor_geom_id:
                foot_geom = geom2

            elif geom2 == floor_geom_id:
                foot_geom = geom1

            else:
                continue

            if foot_geom < 0:
                continue

            body_id = int(
                mj_model.geom_bodyid[
                    foot_geom
                ]
            )

            if body_id not in (
                left_foot_body_id,
                right_foot_body_id,
            ):
                continue

            contact_xy = contact_pos_history[
                k,
                j,
                :2,
            ]

            weighted_position += (
                normal_force
                * contact_xy
            )

            total_force += normal_force

        # =====================================================
        # 4. COP
        # =====================================================

        total_normal_force[k] = total_force

        if total_force > 0.0:

            cop_history[k] = (
                weighted_position
                / total_force
            )

    return (
        cop_history,
        total_normal_force,
    )