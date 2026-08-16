from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from robot_cfg import ACTUATED_JOINTS


class JointPlotter:

    def __init__(self, env, env_idx=0):

        self.env = env
        self.env_idx = env_idx

        self.robot = env.scene["robot"]

        # =====================================================
        # 1. Lấy 8 joint chủ động
        # =====================================================

        joint_ids, joint_names = self.robot.find_joints(
            ACTUATED_JOINTS,
            preserve_order=True,
        )

        self.joint_ids = joint_ids
        self.joint_names = joint_names

        # =====================================================
        # 2. Thông số lấy mẫu
        # =====================================================

        self.step_dt = float(env.step_dt)

        # 20 s × 50 Hz = 1000 samples
        self.max_samples = int(
            20.0 / self.step_dt
        )

        self.sample_count = 0
        self.finalized = False

        num_joints = len(self.joint_names)

        device = self.robot.data.joint_pos.device
        dtype = self.robot.data.joint_pos.dtype

        # =====================================================
        # 3. GPU buffers
        # =====================================================

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

        # Time nằm trên CPU
        self.time_buffer = (
            np.arange(1, self.max_samples + 1)
            * self.step_dt
        )

        # Figure chỉ được tạo sau khi simulation kết thúc
        self.fig_position = None
        self.fig_torque = None

    # =========================================================
    # RECORD DATA - gọi ở 50 Hz
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
    # FINALIZE - chỉ gọi 1 lần sau 20 s
    # =========================================================

    def finalize(self):

        if self.finalized:
            return

        if self.sample_count == 0:
            return

        self.finalized = True

        n = self.sample_count

        print(
            f"Simulation finished: "
            f"{n} samples collected."
        )

        # =====================================================
        # 1. GPU -> CPU chỉ một lần
        # =====================================================

        t = self.time_buffer[:n]

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

        # =====================================================
        # 2. Lưu data
        # =====================================================

        output_dir = Path(
            "logs/standing_eval"
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # -----------------------------------------------------
        # CSV
        # -----------------------------------------------------

        data = np.column_stack(
            (
                t,
                q,
                tau,
            )
        )

        header = ["time_s"]

        header += [
            f"{name}_position_rad"
            for name in self.joint_names
        ]

        header += [
            f"{name}_torque_Nm"
            for name in self.joint_names
        ]

        csv_path = (
            output_dir
            / "joint_data.csv"
        )

        np.savetxt(
            csv_path,
            data,
            delimiter=",",
            header=",".join(header),
            comments="",
        )

        print(
            f"Joint data saved to: {csv_path}"
        )

        # =====================================================
        # 3. Plot joint position
        # =====================================================

        self.fig_position, axes = plt.subplots(
            4,
            2,
            figsize=(12, 9),
            num="Joint Position",
            constrained_layout=True,
        )

        axes = axes.flatten()

        for i, name in enumerate(
            self.joint_names
        ):

            ax = axes[i]

            ax.plot(
                t,
                q[:, i],
            )

            ax.set_title(name)
            ax.set_xlabel("Time [s]")
            ax.set_ylabel("Position [rad]")
            ax.grid(True)

        self.fig_position.savefig(
            output_dir
            / "joint_position.png",
            dpi=150,
        )

        # =====================================================
        # 4. Plot joint torque
        # =====================================================

        self.fig_torque, axes = plt.subplots(
            4,
            2,
            figsize=(12, 9),
            num="Joint Torque",
            constrained_layout=True,
        )

        axes = axes.flatten()

        for i, name in enumerate(
            self.joint_names
        ):

            ax = axes[i]

            ax.plot(
                t,
                tau[:, i],
            )

            ax.set_title(name)
            ax.set_xlabel("Time [s]")
            ax.set_ylabel("Torque [N.m]")
            ax.grid(True)

        self.fig_torque.savefig(
            output_dir
            / "joint_torque.png",
            dpi=150,
        )

        # Hiển thị sau khi simulation đã dừng
        plt.show(block=False)
        plt.pause(0.001)


# =============================================================
# Gắn logger vào Native Viewer
# =============================================================

def with_joint_plots(base_viewer_class):

    class JointPlotViewer(base_viewer_class):

        def setup(self):

            super().setup()

            self._joint_plotter = JointPlotter(
                env=self.env.unwrapped,
                env_idx=self.env_idx,
            )

        # =====================================================
        # Mỗi action step -> lưu 1 mẫu
        # =====================================================

        def _execute_step(self):

            # Đã đủ 20 s
            if (
                self._joint_plotter.sample_count
                >= self._joint_plotter.max_samples
            ):

                self._joint_plotter.finalize()
                self.pause()

                return False

            success = super()._execute_step()

            if success:

                # Sampling đúng theo action = 50 Hz
                self._joint_plotter.record()

                # Đủ 1000 samples = 20 s
                if (
                    self._joint_plotter.sample_count
                    >= self._joint_plotter.max_samples
                ):

                    self._joint_plotter.finalize()

                    # Dừng simulation
                    self.pause()

            return success

        # =====================================================
        # Sau khi kết thúc, chỉ xử lý event của Matplotlib
        # =====================================================

        def sync_env_to_viewer(self):

            super().sync_env_to_viewer()

            if self._joint_plotter.finalized:
                plt.pause(0.001)

        # =====================================================
        # Close
        # =====================================================

        def close(self):

            if self._joint_plotter.fig_position is not None:
                plt.close(
                    self._joint_plotter.fig_position
                )

            if self._joint_plotter.fig_torque is not None:
                plt.close(
                    self._joint_plotter.fig_torque
                )

            super().close()

    return JointPlotViewer