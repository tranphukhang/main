from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from robot_cfg import ACTUATED_JOINTS
from standing.support_polygon import (
    compute_support_polygons,
    create_support_polygon_animation,
)


class JointPlotter:

    def __init__(self, env, env_idx=0):

        self.env = env
        self.env_idx = env_idx

        self.robot = env.scene["robot"]

        self.fig_support = None
        self.support_animation = None

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

        # 12 s × 50 Hz = 600 samples
        self.max_samples = int(
            12.0 / self.step_dt
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

        # Full robot configuration để tính contact sau mô phỏng
        num_qpos = self.env.sim.data.qpos.shape[1]

        self.qpos_buffer = torch.empty(
            (self.max_samples, num_qpos),
            device=device,
            dtype=self.env.sim.data.qpos.dtype,
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

        # Lưu toàn bộ qpos của robot
        self.qpos_buffer[i].copy_(
            self.env.sim.data.qpos[
                self.env_idx
            ]
        )

        self.sample_count += 1

    def run(
        self,
        num_steps=None,
        catch_sigint=True,
    ):

        super().run(
            num_steps=int(
                12.0 / self.env.unwrapped.step_dt
            ),
            catch_sigint=catch_sigint,
        )

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

        qpos = (
            self.qpos_buffer[:n]
            .detach()
            .cpu()
            .numpy()
        )

        # =====================================================
        # Tính support polygon offline
        # =====================================================

        support_polygons = compute_support_polygons(
            self.env.sim.mj_model,
            qpos,
        )

        num_valid = sum(
            len(polygon) >= 3
            for polygon in support_polygons
        )

        print(
            f"Support polygons: "
            f"{num_valid}/{n} samples "
            f"có polygon hợp lệ."
        )

        print(
            "Support polygon sample đầu tiên:"
        )

        print(
            support_polygons[0]
        )

        # =====================================================
        # Animation support polygon
        # =====================================================

        (self.fig_support,self.support_animation,) = create_support_polygon_animation(
            support_polygons,
            t,
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

        plt.close(self.fig_position)

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


        plt.close(self.fig_torque)


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

            success = super()._execute_step()

            if success:

                self._joint_plotter.record()

                if (
                    self._joint_plotter.sample_count
                    >= self._joint_plotter.max_samples
                ):
                    self._joint_plotter.finalize()

            return success

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

            if self._joint_plotter.fig_support is not None:
                plt.close(
                    self._joint_plotter.fig_support
                )

            super().close()

    return JointPlotViewer