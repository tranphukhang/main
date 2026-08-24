import numpy as np
import torch

from evaluation.cop_support_logger import (
    CopSupportLogger,
)
from evaluation.velocity_tracking import (
    create_velocity_tracking_animation,
)


class VelocityCopSupportLogger(
    CopSupportLogger
):
    """
    Mở rộng CopSupportLogger để ghi thêm:

    - Command vx, vy, yaw rate.
    - Velocity thực vx, vy, yaw rate.
    """

    def __init__(
        self,
        env,
        output_dir,
        command_name="twist",
        env_idx=0,
        min_normal_force=1.0,
    ):
        super().__init__(
            env=env,
            output_dir=output_dir,
            env_idx=env_idx,
            min_normal_force=min_normal_force,
        )

        self.command_name = command_name

        buffer_kwargs = {
            "device": env.device,
            "dtype": (
                self.com_pos_buffer.dtype
            ),
        }

        self.velocity_command_buffer = (
            torch.empty(
                (
                    self.max_samples,
                    3,
                ),
                **buffer_kwargs,
            )
        )

        self.robot_velocity_buffer = (
            torch.empty(
                (
                    self.max_samples,
                    3,
                ),
                **buffer_kwargs,
            )
        )

    # =========================================================
    # Record tại mỗi physics substep
    # =========================================================

    def record(self):

        i = self.sample_count

        # Ghi contact, COP và COM giống balance.
        super().record()

        # Khi buffer của parent đã đầy,
        # sample_count sẽ không tăng.
        if self.sample_count == i:
            return

        command = (
            self.env.command_manager.get_command(
                self.command_name
            )
        )

        self.velocity_command_buffer[
            i
        ].copy_(
            command[self.env_idx]
        )

        # Command của MjLab nằm trong body frame.
        # Vì vậy velocity thực cũng lấy trong body frame.

        self.robot_velocity_buffer[
            i,
            :2,
        ].copy_(
            self.robot.data.root_link_lin_vel_b[
                self.env_idx,
                :2,
            ]
        )

        self.robot_velocity_buffer[
            i,
            2,
        ].copy_(
            self.robot.data.root_link_ang_vel_b[
                self.env_idx,
                2,
            ]
        )

    # =========================================================
    # Xử lý và xuất video
    # =========================================================

    def finalize(self):

        if self.finalized:
            return

        n = self.sample_count

        if n == 0:
            super().finalize()
            return

        velocity_command = (
            self.velocity_command_buffer[:n]
            .detach()
            .cpu()
            .numpy()
        )

        robot_velocity = (
            self.robot_velocity_buffer[:n]
            .detach()
            .cpu()
            .numpy()
        )

        # Downsample giống CopSupportLogger:
        # physics rate -> control rate.
        video_stride = int(
            round(
                self.env.step_dt
                / self.env.physics_dt
            )
        )

        time_history = (
            np.arange(1, n + 1)
            * self.physics_dt
        )

        video_time = (
            time_history[
                ::video_stride
            ]
        )

        video_command = (
            velocity_command[
                ::video_stride
            ]
        )

        video_robot_velocity = (
            robot_velocity[
                ::video_stride
            ]
        )

        # Xuất ba video giống hệt task balance:
        #
        # - support_polygon.mp4
        # - stability_timeseries.mp4
        # - stability_velocity.mp4
        super().finalize()

        video_fps = int(
            round(
                1.0 / self.env.step_dt
            )
        )

        output_path = (
            self.output_dir
            / "velocity_tracking.mp4"
        )

        create_velocity_tracking_animation(
            time_history=video_time,
            command_history=video_command,
            robot_velocity_history=(
                video_robot_velocity
            ),
            output_path=output_path,
            fps=video_fps,
        )