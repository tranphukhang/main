import mujoco
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.collections import (
    LineCollection,
)

from matplotlib.patches import Polygon

from standing.support_polygon import (
    convex_hull_2d,
)

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

def plot_cop_trajectory(
    cop_history,
    time_history,
    support_polygons,
    output_path,
):

    # =========================================================
    # 1. COP hợp lệ
    # =========================================================

    valid = np.all(
        np.isfinite(cop_history),
        axis=1,
    )

    if np.sum(valid) < 2:
        print(
            "Không đủ dữ liệu COP để vẽ."
        )
        return

    # =========================================================
    # 2. Tạo support-region envelope
    # =========================================================

    valid_polygons = [
        polygon
        for polygon in support_polygons
        if len(polygon) >= 3
    ]

    if len(valid_polygons) == 0:
        print(
            "Không có support polygon hợp lệ."
        )
        return

    all_support_points = np.vstack(
        valid_polygons
    )

    support_envelope = convex_hull_2d(
        all_support_points
    )

    # =========================================================
    # 3. Tạo các đoạn COP liên tiếp
    # =========================================================

    pair_valid = (
        valid[:-1]
        & valid[1:]
    )

    p0 = cop_history[:-1][
        pair_valid
    ]

    p1 = cop_history[1:][
        pair_valid
    ]

    segments = np.stack(
        (
            p0,
            p1,
        ),
        axis=1,
    )

    segment_time = 0.5 * (
        time_history[:-1][pair_valid]
        + time_history[1:][pair_valid]
    )

    # =========================================================
    # 4. Figure
    # =========================================================

    fig, ax = plt.subplots(
        figsize=(8, 7)
    )

    support_patch = Polygon(
        support_envelope,
        closed=True,
        alpha=0.2,
        label="Support region envelope",
    )

    ax.add_patch(
        support_patch
    )

    # =========================================================
    # 5. COP trajectory có màu theo thời gian
    # =========================================================

    line = LineCollection(
        segments,
        cmap="viridis",
        linewidth=1.5,
    )

    line.set_array(
        segment_time
    )

    line.set_clim(
        time_history[0],
        time_history[-1],
    )

    ax.add_collection(
        line
    )

    colorbar = fig.colorbar(
        line,
        ax=ax,
    )

    colorbar.set_label(
        "Time [s]"
    )

    # =========================================================
    # 6. Start / End
    # =========================================================

    valid_indices = np.where(
        valid
    )[0]

    first = valid_indices[0]
    last = valid_indices[-1]

    ax.plot(
        cop_history[first, 0],
        cop_history[first, 1],
        "o",
        markersize=8,
        label="Start",
    )

    ax.plot(
        cop_history[last, 0],
        cop_history[last, 1],
        "x",
        markersize=9,
        markeredgewidth=2,
        label="End",
    )

    # =========================================================
    # 7. Giới hạn hình
    # =========================================================

    all_plot_points = np.vstack(
        (
            support_envelope,
            cop_history[valid],
        )
    )

    margin = 0.03

    ax.set_xlim(
        np.min(all_plot_points[:, 0]) - margin,
        np.max(all_plot_points[:, 0]) + margin,
    )

    ax.set_ylim(
        np.min(all_plot_points[:, 1]) - margin,
        np.max(all_plot_points[:, 1]) + margin,
    )

    ax.set_aspect(
        "equal",
        adjustable="box",
    )

    ax.set_xlabel(
        "X [m]"
    )

    ax.set_ylabel(
        "Y [m]"
    )

    ax.set_title(
        "COP Trajectory over Time"
    )

    ax.grid(True)
    ax.legend()

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"COP trajectory saved to: "
        f"{output_path}"
    )