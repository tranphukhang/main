from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import (
    FFMpegWriter,
    FuncAnimation,
)


def _compute_y_limits(
    command,
    robot_velocity,
    padding_ratio=0.10,
    min_span=1.0e-3,
):
    """Tính giới hạn trục Y từ command và velocity thực."""

    values = np.concatenate(
        (
            np.asarray(command).reshape(-1),
            np.asarray(
                robot_velocity
            ).reshape(-1),
        )
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


def create_velocity_tracking_animation(
    time_history,
    command_history,
    robot_velocity_history,
    output_path,
    fps=50,
):
    """
    Tạo video bám vận tốc gồm:

    - Linear velocity X.
    - Linear velocity Y.
    - Yaw angular velocity Z.
    """

    time_history = np.asarray(
        time_history,
        dtype=float,
    )

    command_history = np.asarray(
        command_history,
        dtype=float,
    )

    robot_velocity_history = np.asarray(
        robot_velocity_history,
        dtype=float,
    )

    output_path = Path(output_path)

    num_frames = len(time_history)

    if num_frames == 0:
        print(
            "Không có dữ liệu để tạo "
            "velocity tracking video."
        )
        return

    expected_shape = (
        num_frames,
        3,
    )

    if (
        command_history.shape
        != expected_shape
        or robot_velocity_history.shape
        != expected_shape
    ):
        raise ValueError(
            "Dữ liệu velocity phải có "
            "kích thước [N, 3]."
        )

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(14, 9),
        sharex=True,
        num="Velocity Tracking",
    )

    fig.suptitle(
        "Velocity Command Tracking"
    )

    titles = (
        "Linear Velocity X",
        "Linear Velocity Y",
        "Yaw Angular Velocity Z",
    )

    y_labels = (
        "Velocity [m/s]",
        "Velocity [m/s]",
        "Angular velocity [rad/s]",
    )

    time_min = float(
        time_history[0]
    )

    time_max = float(
        time_history[-1]
    )

    if time_max <= time_min:
        time_max = time_min + 1.0

    command_lines = []
    robot_lines = []
    time_cursors = []

    for axis, index in zip(
        axes,
        range(3),
        strict=True,
    ):
        command_line, = axis.plot(
            [],
            [],
            color="black",
            linewidth=2,
            label="Command",
        )

        robot_line, = axis.plot(
            [],
            [],
            color="red",
            linewidth=2,
            linestyle="--",
            label="Robot",
        )

        axis.set_title(
            titles[index]
        )

        axis.set_ylabel(
            y_labels[index]
        )

        axis.set_xlim(
            time_min,
            time_max,
        )

        axis.set_ylim(
            *_compute_y_limits(
                command_history[:, index],
                robot_velocity_history[
                    :,
                    index,
                ],
            )
        )

        axis.grid(
            True,
            alpha=0.4,
        )

        axis.legend(
            loc="upper right"
        )

        command_lines.append(
            command_line
        )

        robot_lines.append(
            robot_line
        )

        time_cursors.append(
            axis.axvline(
                time_min,
                color="black",
                linestyle="--",
                linewidth=1,
                alpha=0.7,
            )
        )

    axes[-1].set_xlabel(
        "Time [s]"
    )

    time_text = fig.text(
        0.5,
        0.94,
        "",
        ha="center",
    )

    fig.tight_layout(
        rect=(
            0.0,
            0.0,
            1.0,
            0.92,
        )
    )

    def update(frame):

        end = frame + 1

        current_time = float(
            time_history[frame]
        )

        for index in range(3):

            command_lines[
                index
            ].set_data(
                time_history[:end],
                command_history[
                    :end,
                    index,
                ],
            )

            robot_lines[
                index
            ].set_data(
                time_history[:end],
                robot_velocity_history[
                    :end,
                    index,
                ],
            )

            time_cursors[
                index
            ].set_xdata(
                [
                    current_time,
                    current_time,
                ]
            )

        time_text.set_text(
            f"Current time: "
            f"{current_time:.2f} s"
        )

        return (
            *command_lines,
            *robot_lines,
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
        "Velocity tracking animation "
        f"saved to: {output_path}"
    )