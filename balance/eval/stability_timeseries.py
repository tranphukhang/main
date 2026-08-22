from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import (
    FFMpegWriter,
    FuncAnimation,
)


def _compute_y_limits(
    *signals,
    padding_ratio=0.10,
    min_span=1.0e-3,
    symmetric=False,
):
    """Tính giới hạn trục Y phù hợp dữ liệu."""

    values = np.concatenate(
        [
            np.asarray(signal).reshape(-1)
            for signal in signals
        ]
    )

    finite_values = values[
        np.isfinite(values)
    ]

    if len(finite_values) == 0:
        return -1.0, 1.0

    value_min = float(
        np.min(finite_values)
    )

    value_max = float(
        np.max(finite_values)
    )

    if symmetric:
        half_span = max(
            abs(value_min),
            abs(value_max),
            0.5 * min_span,
        )

        half_span *= (
            1.0 + padding_ratio
        )

        return -half_span, half_span

    value_span = max(
        value_max - value_min,
        min_span,
    )

    padding = (
        padding_ratio * value_span
    )

    return (
        value_min - padding,
        value_max + padding,
    )


def create_stability_timeseries_animation(
    time_history,
    com_history,
    cop_history,
    capture_point_history,
    output_path,
    fps=50,
):
    """Tạo video COM, COP và Capture Point theo thời gian."""

    time_history = np.asarray(
        time_history,
        dtype=float,
    )

    com_history = np.asarray(
        com_history,
        dtype=float,
    )

    cop_history = np.asarray(
        cop_history,
        dtype=float,
    )

    capture_point_history = np.asarray(
        capture_point_history,
        dtype=float,
    )

    output_path = Path(
        output_path
    )

    num_frames = len(
        time_history
    )

    if num_frames == 0:
        print(
            "Không có dữ liệu để tạo "
            "stability time-series video."
        )
        return

    if not (
        len(com_history)
        == len(cop_history)
        == len(capture_point_history)
        == num_frames
    ):
        raise ValueError(
            "COM, COP, Capture Point và time "
            "phải có cùng số mẫu."
        )

    # =========================================================
    # Tạo figure 2 x 2
    # =========================================================

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(14, 8),
        sharex=True,
        num="Balance Time Series",
    )

    ax_com_x = axes[0, 0]
    ax_com_y = axes[0, 1]
    ax_cop_x = axes[1, 0]
    ax_cop_y = axes[1, 1]

    fig.suptitle(
        "Balance Evaluation Time Series"
    )

    # =========================================================
    # Đồ thị 1: COM x và Capture Point x
    # =========================================================

    com_x_line, = ax_com_x.plot(
        [],
        [],
        color="red",
        linewidth=2,
        label="COM X",
    )

    cp_x_com_line, = ax_com_x.plot(
        [],
        [],
        color="purple",
        linewidth=2,
        label="Capture Point X",
    )

    ax_com_x.set_title(
        "COM X vs Capture Point X"
    )

    ax_com_x.set_ylabel(
        "X position [m]"
    )

    # =========================================================
    # Đồ thị 2: COM y và Capture Point y
    # =========================================================

    com_y_line, = ax_com_y.plot(
        [],
        [],
        color="red",
        linewidth=2,
        label="COM Y",
    )

    cp_y_com_line, = ax_com_y.plot(
        [],
        [],
        color="purple",
        linewidth=2,
        label="Capture Point Y",
    )

    ax_com_y.set_title(
        "COM Y vs Capture Point Y"
    )

    ax_com_y.set_ylabel(
        "Y position [m]"
    )

    # =========================================================
    # Đồ thị 3: COP x và Capture Point x
    # =========================================================

    cop_x_line, = ax_cop_x.plot(
        [],
        [],
        color="darkorange",
        linewidth=2,
        label="COP X",
    )

    cp_x_cop_line, = ax_cop_x.plot(
        [],
        [],
        color="purple",
        linewidth=2,
        label="Capture Point X",
    )

    ax_cop_x.set_title(
        "COP X vs Capture Point X"
    )

    ax_cop_x.set_xlabel(
        "Time [s]"
    )

    ax_cop_x.set_ylabel(
        "X position [m]"
    )

    # =========================================================
    # Đồ thị 4: COP y và Capture Point y
    # =========================================================

    cop_y_line, = ax_cop_y.plot(
        [],
        [],
        color="darkorange",
        linewidth=2,
        label="COP Y",
    )

    cp_y_cop_line, = ax_cop_y.plot(
        [],
        [],
        color="purple",
        linewidth=2,
        label="Capture Point Y",
    )

    ax_cop_y.set_title(
        "COP Y vs Capture Point Y"
    )

    ax_cop_y.set_xlabel(
        "Time [s]"
    )

    ax_cop_y.set_ylabel(
        "Y position [m]"
    )

    # =========================================================
    # Giới hạn trục
    # =========================================================

    time_min = float(
        time_history[0]
    )

    time_max = float(
        time_history[-1]
    )

    if time_max <= time_min:
        time_max = time_min + 1.0

    for ax in axes.flat:
        ax.set_xlim(
            time_min,
            time_max,
        )

        ax.grid(
            True,
            alpha=0.4,
        )

        ax.legend(
            loc="upper right"
        )

    x_position_limits = _compute_y_limits(
        com_history[:, 0],
        cop_history[:, 0],
        capture_point_history[:, 0],
    )

    y_position_limits = _compute_y_limits(
        com_history[:, 1],
        cop_history[:, 1],
        capture_point_history[:, 1],
    )

    ax_com_x.set_ylim(
        *x_position_limits
    )

    ax_cop_x.set_ylim(
        *x_position_limits
    )

    ax_com_y.set_ylim(
        *y_position_limits
    )

    ax_cop_y.set_ylim(
        *y_position_limits
    )

    # Đường thẳng biểu diễn thời điểm hiện tại
    time_cursors = [
        ax.axvline(
            time_min,
            color="black",
            linestyle="--",
            linewidth=1,
            alpha=0.7,
        )
        for ax in axes.flat
    ]

    time_text = fig.text(
        0.5,
        0.94,
        "",
        ha="center",
    )

    fig.tight_layout(
        rect=(0.0, 0.0, 1.0, 0.92)
    )

    # =========================================================
    # Cập nhật từng frame
    # =========================================================

    def update(frame):
        end = frame + 1

        current_time = float(
            time_history[frame]
        )

        current_times = (
            time_history[:end]
        )

        com_x_line.set_data(
            current_times,
            com_history[:end, 0],
        )

        cp_x_com_line.set_data(
            current_times,
            capture_point_history[:end, 0],
        )

        com_y_line.set_data(
            current_times,
            com_history[:end, 1],
        )

        cp_y_com_line.set_data(
            current_times,
            capture_point_history[:end, 1],
        )

        cop_x_line.set_data(
            current_times,
            cop_history[:end, 0],
        )

        cp_x_cop_line.set_data(
            current_times,
            capture_point_history[:end, 0],
        )

        cop_y_line.set_data(
            current_times,
            cop_history[:end, 1],
        )

        cp_y_cop_line.set_data(
            current_times,
            capture_point_history[:end, 1],
        )

        for cursor in time_cursors:
            cursor.set_xdata(
                [current_time, current_time]
            )

        time_text.set_text(
            f"Current time: "
            f"{current_time:.2f} s"
        )

        return (
            com_x_line,
            cp_x_com_line,
            com_y_line,
            cp_y_com_line,
            cop_x_line,
            cp_x_cop_line,
            cop_y_line,
            cp_y_cop_line,
            *time_cursors,
            time_text,
        )

    # =========================================================
    # Lưu video
    # =========================================================

    animation = FuncAnimation(
        fig,
        update,
        frames=num_frames,
        interval=1000.0 / fps,
        blit=False,
        repeat=True,
    )

    writer = FFMpegWriter(
        fps=fps,
        bitrate=4000,
    )

    animation.save(
        output_path,
        writer=writer,
        dpi=150,
    )

    plt.close(fig)

    print(
        "Stability time-series animation "
        f"saved to: {output_path}"
    )

def create_stability_velocity_animation(
    time_history,
    com_velocity_history,
    capture_point_velocity_history,
    output_path,
    fps=50,
):
    """Tạo video 4 đồ thị vận tốc COM và Capture Point."""

    time_history = np.asarray(
        time_history,
        dtype=float,
    )

    com_velocity_history = np.asarray(
        com_velocity_history,
        dtype=float,
    )

    capture_point_velocity_history = np.asarray(
        capture_point_velocity_history,
        dtype=float,
    )

    output_path = Path(
        output_path
    )

    num_frames = len(
        time_history
    )

    if num_frames == 0:
        print(
            "Không có dữ liệu để tạo "
            "stability velocity video."
        )
        return

    if not (
        len(com_velocity_history)
        == len(capture_point_velocity_history)
        == num_frames
    ):
        raise ValueError(
            "COM velocity, Capture Point velocity "
            "và time phải có cùng số mẫu."
        )

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(14, 8),
        sharex=True,
        num="Balance Velocity Time Series",
    )

    ax_com_vx = axes[0, 0]
    ax_com_vy = axes[0, 1]
    ax_cp_vx = axes[1, 0]
    ax_cp_vy = axes[1, 1]

    fig.suptitle(
        "COM and Capture Point Velocity"
    )

    com_vx_line, = ax_com_vx.plot(
        [],
        [],
        color="red",
        linewidth=2,
        label="COM velocity X",
    )

    ax_com_vx.set_title(
        "COM Velocity X"
    )

    com_vy_line, = ax_com_vy.plot(
        [],
        [],
        color="red",
        linewidth=2,
        label="COM velocity Y",
    )

    ax_com_vy.set_title(
        "COM Velocity Y"
    )

    cp_vx_line, = ax_cp_vx.plot(
        [],
        [],
        color="purple",
        linewidth=2,
        label="Capture Point velocity X",
    )

    ax_cp_vx.set_title(
        "Capture Point Velocity X"
    )

    cp_vy_line, = ax_cp_vy.plot(
        [],
        [],
        color="purple",
        linewidth=2,
        label="Capture Point velocity Y",
    )

    ax_cp_vy.set_title(
        "Capture Point Velocity Y"
    )

    time_min = float(
        time_history[0]
    )

    time_max = float(
        time_history[-1]
    )

    if time_max <= time_min:
        time_max = time_min + 1.0

    for ax in axes.flat:
        ax.set_xlim(
            time_min,
            time_max,
        )

        ax.set_ylabel(
            "Velocity [m/s]"
        )

        ax.grid(
            True,
            alpha=0.4,
        )

        ax.legend(
            loc="upper right"
        )

    ax_cp_vx.set_xlabel(
        "Time [s]"
    )

    ax_cp_vy.set_xlabel(
        "Time [s]"
    )

    ax_com_vx.set_ylim(
        *_compute_y_limits(
            com_velocity_history[:, 0],
            symmetric=True,
        )
    )

    ax_com_vy.set_ylim(
        *_compute_y_limits(
            com_velocity_history[:, 1],
            symmetric=True,
        )
    )

    ax_cp_vx.set_ylim(
        *_compute_y_limits(
            capture_point_velocity_history[
                :, 0
            ],
            symmetric=True,
        )
    )

    ax_cp_vy.set_ylim(
        *_compute_y_limits(
            capture_point_velocity_history[
                :, 1
            ],
            symmetric=True,
        )
    )

    time_cursors = [
        ax.axvline(
            time_min,
            color="black",
            linestyle="--",
            linewidth=1,
            alpha=0.7,
        )
        for ax in axes.flat
    ]

    time_text = fig.text(
        0.5,
        0.94,
        "",
        ha="center",
    )

    fig.tight_layout(
        rect=(0.0, 0.0, 1.0, 0.92)
    )

    def update(frame):
        end = frame + 1

        current_time = float(
            time_history[frame]
        )

        current_times = (
            time_history[:end]
        )

        com_vx_line.set_data(
            current_times,
            com_velocity_history[:end, 0],
        )

        com_vy_line.set_data(
            current_times,
            com_velocity_history[:end, 1],
        )

        cp_vx_line.set_data(
            current_times,
            capture_point_velocity_history[
                :end, 0
            ],
        )

        cp_vy_line.set_data(
            current_times,
            capture_point_velocity_history[
                :end, 1
            ],
        )

        for cursor in time_cursors:
            cursor.set_xdata(
                [current_time, current_time]
            )

        time_text.set_text(
            f"Current time: "
            f"{current_time:.2f} s"
        )

        return (
            com_vx_line,
            com_vy_line,
            cp_vx_line,
            cp_vy_line,
            *time_cursors,
            time_text,
        )

    animation = FuncAnimation(
        fig,
        update,
        frames=num_frames,
        interval=1000.0 / fps,
        blit=False,
        repeat=True,
    )

    writer = FFMpegWriter(
        fps=fps,
        bitrate=4000,
    )

    animation.save(
        output_path,
        writer=writer,
        dpi=150,
    )

    plt.close(fig)

    print(
        "Stability velocity animation "
        f"saved to: {output_path}"
    )