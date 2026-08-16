import time

import matplotlib.pyplot as plt
import numpy as np
import torch

from robot_cfg import ACTUATED_JOINTS


class JointPlotter:
    """
    Thu thập và hiển thị:
        - Joint position [rad]
        - Actuator torque [N.m]

    Mỗi joint có một subplot riêng.
    """

    def __init__(self, env, env_idx=0):

        self.env = env
        self.env_idx = env_idx

        self.robot = env.scene["robot"]

        # =====================================================
        # 1. Lấy đúng 8 joint chủ động
        # =====================================================

        joint_ids, joint_names = self.robot.find_joints(
            ACTUATED_JOINTS,
            preserve_order=True,
        )

        self.joint_ids = joint_ids
        self.joint_names = joint_names

        # =====================================================
        # 2. Bộ nhớ dữ liệu
        # =====================================================

        self.step_dt = float(env.step_dt)

        # 20 s × 50 Hz = 1000 mẫu
        self.max_samples = int(
            20.0 / self.step_dt
        )

        self.sample_count = 0

        num_joints = len(self.joint_names)

        device = self.robot.data.joint_pos.device
        dtype = self.robot.data.joint_pos.dtype

        # Buffer nằm trên GPU
        self.position_buffer = torch.empty(
            (self.max_samples, num_joints),
            device=device,
            dtype=dtype,
        )

        self.torque_buffer = torch.empty(
            (self.max_samples, num_joints),
            device=device,
            dtype=dtype,
        )

        # Time không cần đọc từ GPU
        self.time_buffer = (
            np.arange(1, self.max_samples + 1)
            * self.step_dt
        )

        # =====================================================
        # 3. Khởi tạo GUI
        # =====================================================

        plt.ion()

        self._create_position_gui()
        self._create_torque_gui()

        plt.show(block=False)

    # =========================================================
    # POSITION GUI
    # =========================================================

    def _create_position_gui(self):

        self.fig_position, axes = plt.subplots(
            4,
            2,
            figsize=(12, 9),
            num="Joint Position",
            constrained_layout=True,
        )

        self.axes_position = axes.flatten()

        self.position_lines = {}

        for ax, name in zip(
            self.axes_position,
            self.joint_names,
        ):

            line, = ax.plot([], [])

            self.position_lines[name] = line

            ax.set_title(name)
            ax.set_xlabel("Time [s]")
            ax.set_ylabel("Position [rad]")
            ax.grid(True)

    # =========================================================
    # TORQUE GUI
    # =========================================================

    def _create_torque_gui(self):

        self.fig_torque, axes = plt.subplots(
            4,
            2,
            figsize=(12, 9),
            num="Joint Torque",
            constrained_layout=True,
        )

        self.axes_torque = axes.flatten()

        self.torque_lines = {}

        for ax, name in zip(
            self.axes_torque,
            self.joint_names,
        ):

            line, = ax.plot([], [])

            self.torque_lines[name] = line

            ax.set_title(name)
            ax.set_xlabel("Time [s]")
            ax.set_ylabel("Torque [N.m]")
            ax.grid(True)

    # =========================================================
    # GHI DỮ LIỆU
    # =========================================================

    def record(self):

        if self.sample_count >= self.max_samples:
            return

        i = self.sample_count

        # Joint position
        self.position_buffer[i].copy_(
            self.robot.data.joint_pos[
                self.env_idx,
                self.joint_ids,
            ]
        )

        # Actuator torque
        self.torque_buffer[i].copy_(
            self.robot.data.qfrc_actuator[
                self.env_idx,
                self.joint_ids,
            ]
        )

        self.sample_count += 1

    # =========================================================
    # UPDATE GUI
    # =========================================================

    def update(self):

        # Số mẫu hiện có
        n = self.sample_count

        if n < 2:
            return

        # =========================================================
        # 1. Lấy dữ liệu để vẽ
        # =========================================================

        # Time đã nằm trên CPU
        t = self.time_buffer[:n]

        # Chỉ chuyển GPU -> CPU khi cần update GUI
        q = (
            self.position_buffer[:n]
            .detach()
            .cpu()
            .numpy()
        )

        tau = (
            self.torque_buffer[:n]
            .detach()
            .cpu()
            .numpy()
        )

        # =========================================================
        # 2. Update Joint Position
        # =========================================================

        for i, (ax, name) in enumerate(
            zip(
                self.axes_position,
                self.joint_names,
            )
        ):

            self.position_lines[name].set_data(
                t,
                q[:, i],
            )

            ax.relim()
            ax.autoscale_view()

        # =========================================================
        # 3. Update Joint Torque
        # =========================================================

        for i, (ax, name) in enumerate(
            zip(
                self.axes_torque,
                self.joint_names,
            )
        ):

            self.torque_lines[name].set_data(
                t,
                tau[:, i],
            )

            ax.relim()
            ax.autoscale_view()

        # =========================================================
        # 4. Refresh GUI
        # =========================================================

        self.fig_position.canvas.draw_idle()
        self.fig_torque.canvas.draw_idle()

        self.fig_position.canvas.flush_events()
        self.fig_torque.canvas.flush_events()

        plt.pause(0.001)

    # =========================================================
    # CLEAR
    # =========================================================

    def clear(self):

        self.time_history.clear()

        for name in self.joint_names:
            self.position_history[name].clear()
            self.torque_history[name].clear()

    # =========================================================
    # CLOSE
    # =========================================================

    def close(self):

        plt.close(self.fig_position)
        plt.close(self.fig_torque)


# =============================================================
# Gắn JointPlotter vào một Native Viewer bất kỳ
# =============================================================

def with_joint_plots(base_viewer_class):

    class JointPlotViewer(base_viewer_class):

        def setup(self):

            # Setup Native Viewer trước
            super().setup()

            # Sau đó tạo Joint Plotter
            self._joint_plotter = JointPlotter(
                env=self.env.unwrapped,
                env_idx=self.env_idx,
            )

            self._last_plot_update = 0.0

        # -----------------------------------------------------
        # Ghi dữ liệu sau mỗi environment step
        # -----------------------------------------------------

        def _execute_step(self):

            # Đã đủ 20 s -> không step thêm
            if (
                self._joint_plotter.sample_count
                >= self._joint_plotter.max_samples
            ):
                self.pause()
                return False

            success = super()._execute_step()

            if success:

                # Lấy đúng 1 mẫu cho mỗi action step
                self._joint_plotter.record()

                # Đủ 1000 mẫu = 20 s
                if (
                    self._joint_plotter.sample_count
                    >= self._joint_plotter.max_samples
                ):
                    # Vẽ lần cuối đầy đủ 1000 mẫu
                    self._joint_plotter.update()

                    # Dừng simulation nhưng giữ GUI
                    self.pause()

            return success

        # -----------------------------------------------------
        # Update GUI plot khoảng 10 Hz
        # -----------------------------------------------------

        def sync_env_to_viewer(self):

            super().sync_env_to_viewer()

            now = time.perf_counter()

            if now - self._last_plot_update >= 0.2:

                self._joint_plotter.update()

                self._last_plot_update = now

        # -----------------------------------------------------
        # Close
        # -----------------------------------------------------

        def close(self):

            if hasattr(self, "_joint_plotter"):
                self._joint_plotter.close()

            super().close()

    return JointPlotViewer