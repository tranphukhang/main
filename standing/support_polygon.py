import mujoco
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import (
    FuncAnimation,
    FFMpegWriter,
)
from matplotlib.patches import Polygon


def convex_hull_2d(points):
    """
    Tính convex hull 2D bằng monotonic chain.

    Input:
        points: array shape (N, 2)

    Output:
        hull: array shape (M, 2)
    """

    points = np.asarray(points, dtype=float)

    if len(points) == 0:
        return np.empty((0, 2))

    # Loại các điểm trùng nhau
    points = np.unique(points, axis=0)

    if len(points) <= 2:
        return points

    # Sắp xếp theo x, sau đó y
    points = points[
        np.lexsort(
            (points[:, 1], points[:, 0])
        )
    ]

    def cross(o, a, b):
        return (
            (a[0] - o[0]) * (b[1] - o[1])
            - (a[1] - o[1]) * (b[0] - o[0])
        )

    lower = []

    for p in points:

        while (
            len(lower) >= 2
            and cross(
                lower[-2],
                lower[-1],
                p,
            ) <= 0
        ):
            lower.pop()

        lower.append(p)

    upper = []

    for p in reversed(points):

        while (
            len(upper) >= 2
            and cross(
                upper[-2],
                upper[-1],
                p,
            ) <= 0
        ):
            upper.pop()

        upper.append(p)

    hull = np.array(
        lower[:-1] + upper[:-1]
    )

    return hull


def compute_support_polygons(
    mj_model,
    contact_pos_history,
    contact_geom_history,
    normal_force_history,
    nacon_history,
    min_normal_force=1.0,
):

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

    polygons = []

    num_samples = len(
        contact_pos_history
    )

    for k in range(num_samples):

        contact_points = []

        # Chỉ duyệt số contact thực sự active
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

            # Bỏ contact có lực quá nhỏ
            if (
                normal_force
                <= min_normal_force
            ):
                continue

            # Chỉ xét chân <-> terrain
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

            contact_points.append(
                contact_pos_history[
                    k,
                    j,
                    :2,
                ].copy()
            )

        polygon = convex_hull_2d(
            contact_points
        )

        polygons.append(
            polygon
        )

    return polygons

def create_support_polygon_animation(
    polygons,
    time_history,
    cop_history,
):

    # =========================================================
    # 1. Tìm giới hạn vùng vẽ
    # =========================================================

    valid_points = [
        polygon
        for polygon in polygons
        if len(polygon) >= 3
    ]

    if len(valid_points) == 0:
        print("Không có support polygon hợp lệ.")
        return None, None

    all_points = np.vstack(valid_points)

    x_min = np.min(all_points[:, 0])
    x_max = np.max(all_points[:, 0])

    y_min = np.min(all_points[:, 1])
    y_max = np.max(all_points[:, 1])

    margin = 0.05

    # =========================================================
    # 2. Tạo figure
    # =========================================================

    fig, ax = plt.subplots(
        figsize=(7, 7),
        num="Support Polygon",
    )

    ax.set_xlim(
        x_min - margin,
        x_max + margin,
    )

    ax.set_ylim(
        y_min - margin,
        y_max + margin,
    )

    ax.set_aspect(
        "equal",
        adjustable="box",
    )

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.grid(True)

    # Polygon patch
    polygon_patch = Polygon(
        np.zeros((3, 2)),
        closed=True,
        alpha=0.3,
    )

    ax.add_patch(polygon_patch)

    # Các đỉnh contact
    vertices_plot, = ax.plot(
        [],
        [],
        "o",
    )

    # Điểm COP
    cop_plot, = ax.plot(
        [],
        [],
        "x",
        markersize=10,
        markeredgewidth=2,
        label="COP",
    )

    ax.legend()

    # =========================================================
    # 3. Update từng frame
    # =========================================================

    def update(frame):

        polygon = polygons[frame]

        if len(polygon) >= 3:

            polygon_patch.set_xy(
                polygon
            )

            polygon_patch.set_visible(
                True
            )

            vertices_plot.set_data(
                polygon[:, 0],
                polygon[:, 1],
            )

        else:

            polygon_patch.set_visible(
                False
            )

            vertices_plot.set_data(
                [],
                [],
            )

        cop = cop_history[frame]

        if np.all(
            np.isfinite(cop)
        ):
            cop_plot.set_data(
                [cop[0]],
                [cop[1]],
            )
        else:
            cop_plot.set_data(
                [],
                [],
            )

        ax.set_title(
            f"Support Polygon - "
            f"t = {time_history[frame]:.2f} s"
        )

        return (
            polygon_patch,
            vertices_plot,
            cop_plot,
        )

    # =========================================================
    # 4. Animation 50 Hz
    # =========================================================

    animation = FuncAnimation(
        fig,
        update,
        frames=len(polygons),
        interval=20,
        blit=False,
        repeat=True,
    )

    output_path = (
        "logs/standing_eval/"
        "support_polygon.mp4"
    )

    writer = FFMpegWriter(
        fps=50,
        bitrate=3000,
    )

    animation.save(
        output_path,
        writer=writer,
        dpi=150,
    )

    plt.close(fig)

    print(
        f"Support polygon animation saved to: "
        f"{output_path}"
    )

    return None, None