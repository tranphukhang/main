from pathlib import Path

import mujoco
import numpy as np
import matplotlib.pyplot as plt


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


def plot_cop_position(
    cop_history,
    time_history,
    support_polygons,
    impulse_active,
    output_path,
):

    num_samples = len(
        time_history
    )

    # =========================================================
    # 1. Support limits theo từng thời điểm
    # =========================================================

    x_min = np.full(
        num_samples,
        np.nan,
    )

    x_max = np.full(
        num_samples,
        np.nan,
    )

    y_min = np.full(
        num_samples,
        np.nan,
    )

    y_max = np.full(
        num_samples,
        np.nan,
    )

    cop_x_rel = np.full(
        num_samples,
        np.nan,
    )

    cop_y_rel = np.full(
        num_samples,
        np.nan,
    )

    x_min_rel = np.full(
        num_samples,
        np.nan,
    )

    x_max_rel = np.full(
        num_samples,
        np.nan,
    )

    y_min_rel = np.full(
        num_samples,
        np.nan,
    )

    y_max_rel = np.full(
        num_samples,
        np.nan,
    )

    # =========================================================
    # 2. Tính vị trí tương đối với trung điểm
    #    của khoảng support theo từng trục
    # =========================================================

    for k in range(num_samples):

        polygon = support_polygons[k]

        if len(polygon) < 3:
            continue

        x_min[k] = np.min(
            polygon[:, 0]
        )

        x_max[k] = np.max(
            polygon[:, 0]
        )

        y_min[k] = np.min(
            polygon[:, 1]
        )

        y_max[k] = np.max(
            polygon[:, 1]
        )

        center_x = 0.5 * (
            x_min[k]
            + x_max[k]
        )

        center_y = 0.5 * (
            y_min[k]
            + y_max[k]
        )

        x_min_rel[k] = (
            x_min[k]
            - center_x
        )

        x_max_rel[k] = (
            x_max[k]
            - center_x
        )

        y_min_rel[k] = (
            y_min[k]
            - center_y
        )

        y_max_rel[k] = (
            y_max[k]
            - center_y
        )

        if np.all(
            np.isfinite(
                cop_history[k]
            )
        ):

            cop_x_rel[k] = (
                cop_history[k, 0]
                - center_x
            )

            cop_y_rel[k] = (
                cop_history[k, 1]
                - center_y
            )

    # =========================================================
    # 3. Lưu dữ liệu COP để vẽ lại trên MATLAB
    # =========================================================

    output_path = Path(
        output_path
    )

    cop_csv_data = np.column_stack(
        (
            time_history,
            cop_history[:, 0],
            cop_history[:, 1],
            cop_x_rel,
            cop_y_rel,
            x_min_rel,
            x_max_rel,
            y_min_rel,
            y_max_rel,
            impulse_active.astype(
                np.int8
            ),
        )
    )

    cop_csv_header = [
        "time_s",
        "cop_x_m",
        "cop_y_m",
        "cop_x_rel_m",
        "cop_y_rel_m",
        "support_x_min_rel_m",
        "support_x_max_rel_m",
        "support_y_min_rel_m",
        "support_y_max_rel_m",
        "impulse_active",
    ]

    cop_csv_path = (
        output_path.parent
        / "cop_position_data.csv"
    )

    np.savetxt(
        cop_csv_path,
        cop_csv_data,
        delimiter=",",
        header=",".join(
            cop_csv_header
        ),
        comments="",
    )

    print(
        f"COP plot data saved to: "
        f"{cop_csv_path}"
    )

    # =========================================================
    # 4. Xác định các khoảng external impulse
    # =========================================================

    active = impulse_active.astype(
        np.int8
    )

    padded = np.pad(
        active,
        (1, 1),
        constant_values=0,
    )

    changes = np.diff(
        padded
    )

    impulse_starts = np.where(
        changes == 1
    )[0]

    impulse_ends = (
        np.where(
            changes == -1
        )[0]
        - 1
    )

    # =========================================================
    # 5. Figure
    # =========================================================

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(11, 8),
        sharex=True,
        constrained_layout=True,
    )

    # =========================================================
    # 6. COP theo X
    # =========================================================

    axes[0].fill_between(
        time_history,
        x_min_rel,
        x_max_rel,
        alpha=0.2,
        label="Support region",
    )

    axes[0].plot(
        time_history,
        cop_x_rel,
        linewidth=1.0,
        label="COP X",
    )

    for i, (start, end) in enumerate(
        zip(
            impulse_starts,
            impulse_ends,
        )
    ):

        axes[0].axvspan(
            time_history[start],
            time_history[end],
            color="red",
            alpha=0.30,
            label=(
                "External impulse"
                if i == 0
                else None
            ),
        )

    axes[0].axhline(
        0.0,
        linewidth=0.8,
        linestyle="--",
    )

    axes[0].set_ylabel(
        "Relative X [m]"
    )

    axes[0].set_title(
        "COP Position along X"
    )

    axes[0].grid(True)
    axes[0].legend()

    # =========================================================
    # 7. COP theo Y
    # =========================================================

    axes[1].fill_between(
        time_history,
        y_min_rel,
        y_max_rel,
        alpha=0.2,
        label="Support region",
    )

    axes[1].plot(
        time_history,
        cop_y_rel,
        linewidth=1.0,
        label="COP Y",
    )

    for i, (start, end) in enumerate(
        zip(
            impulse_starts,
            impulse_ends,
        )
    ):

        axes[1].axvspan(
            time_history[start],
            time_history[end],
            color="red",
            alpha=0.30,
            label=(
                "External impulse"
                if i == 0
                else None
            ),
        )

    axes[1].axhline(
        0.0,
        linewidth=0.8,
        linestyle="--",
    )

    axes[1].set_xlabel(
        "Time [s]"
    )

    axes[1].set_ylabel(
        "Relative Y [m]"
    )

    axes[1].set_title(
        "COP Position along Y"
    )

    axes[1].grid(True)
    axes[1].legend()

    # =========================================================
    # 8. Save
    # =========================================================

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"COP position plot saved to: "
        f"{output_path}"
    )