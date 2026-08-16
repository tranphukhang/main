from collections import deque
import time

import matplotlib.pyplot as plt
import numpy as np

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

        # 20 s x 50 Hz = khoảng 1000 mẫu.
        # Để dư 2000 mẫu.
        max_samples = 2000

        self.time_history = deque(maxlen=max_samples)

        self.position_history = {
            name: deque(maxlen=max_samples)
            for name in self.joint_names
        }

        self.torque_history = {
            name: deque(maxlen=max_samples)
            for name in self.joint_names
        }

        self.last_time = None

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

        # -----------------------------------------------------
        # Thời gian hiện tại của episode
        # -----------------------------------------------------

        episode_step = int(
            self.env.episode_length_buf[
                self.env_idx
            ]
            .detach()
            .cpu()
            .item()
        )

        t = episode_step * float(self.env.step_dt)

        # Nếu episode reset thì xóa dữ liệu episode trước
        if (
            self.last_time is not None
            and t < self.last_time
        ):
            self.clear()

        self.last_time = t

        # -----------------------------------------------------
        # Joint position
        # -----------------------------------------------------

        q = (
            self.robot.data.joint_pos[
                self.env_idx,
                self.joint_ids,
            ]
            .detach()
            .cpu()
            .numpy()
        )

        # -----------------------------------------------------
        # Actuator torque
        # -----------------------------------------------------

        tau = (
            self.robot.data.qfrc_actuator[
                self.env_idx,
                self.joint_ids,
            ]
            .detach()
            .cpu()
            .numpy()
        )

        # -----------------------------------------------------
        # Lưu history
        # -----------------------------------------------------

        self.time_history.append(t)

        for i, name in enumerate(self.joint_names):

            self.position_history[name].append(
                float(q[i])
            )

            self.torque_history[name].append(
                float(tau[i])
            )

    # =========================================================
    # UPDATE GUI
    # =========================================================

    def update(self):

        if len(self.time_history) < 2:
            return

        t = np.asarray(self.time_history)

        # -----------------------------------------------------
        # Position
        # -----------------------------------------------------

        for ax, name in zip(
            self.axes_position,
            self.joint_names,
        ):

            q = np.asarray(
                self.position_history[name]
            )

            self.position_lines[name].set_data(
                t,
                q,
            )

            ax.relim()
            ax.autoscale_view()

        # -----------------------------------------------------
        # Torque
        # -----------------------------------------------------

        for ax, name in zip(
            self.axes_torque,
            self.joint_names,
        ):

            tau = np.asarray(
                self.torque_history[name]
            )

            self.torque_lines[name].set_data(
                t,
                tau,
            )

            ax.relim()
            ax.autoscale_view()

        # -----------------------------------------------------
        # Refresh GUI
        # -----------------------------------------------------

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

            success = super()._execute_step()

            if success:
                self._joint_plotter.record()

            return success

        # -----------------------------------------------------
        # Update GUI plot khoảng 10 Hz
        # -----------------------------------------------------

        def sync_env_to_viewer(self):

            super().sync_env_to_viewer()

            now = time.perf_counter()

            if now - self._last_plot_update >= 0.1:

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