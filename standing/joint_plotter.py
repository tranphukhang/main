from pathlib import Path

import mujoco
import mujoco_warp as mjwarp
import warp as wp
import matplotlib.pyplot as plt
import numpy as np
import torch

from robot_cfg import ACTUATED_JOINTS
from standing.support_polygon import (
    compute_support_polygons,
    create_support_polygon_animation,
)
from standing.cop import (
    compute_cop,
    plot_cop_position,
)


class JointPlotter:

    def __init__(
        self,
        env,
        env_idx=0,
    ):

        self.env = env
        self.env_idx = env_idx

        self.robot = env.scene[
            "robot"
        ]

        # =====================================================
        # ID terrain và hai bàn chân cho visualization
        # =====================================================

        self.floor_geom_id = mujoco.mj_name2id(
            self.env.sim.mj_model,
            mujoco.mjtObj.mjOBJ_GEOM,
            "terrain",
        )

        self.left_foot_body_id = mujoco.mj_name2id(
            self.env.sim.mj_model,
            mujoco.mjtObj.mjOBJ_BODY,
            "robot/feet_left",
        )

        self.right_foot_body_id = mujoco.mj_name2id(
            self.env.sim.mj_model,
            mujoco.mjtObj.mjOBJ_BODY,
            "robot/feet_right",
        )

        self.fig_support = None
        self.support_animation = None

        # =====================================================
        # 1. Lấy 8 joint chủ động
        # =====================================================

        joint_ids, joint_names = (
            self.robot.find_joints(
                ACTUATED_JOINTS,
                preserve_order=True,
            )
        )

        self.joint_ids = joint_ids
        self.joint_names = joint_names

        # =====================================================
        # 2. Thông số lấy mẫu
        # =====================================================

        self.physics_dt = float(
            env.physics_dt
        )

        self.max_samples = int(
            8.0
            / self.physics_dt
        )

        self.sample_count = 0
        self.finalized = False

        num_joints = len(
            self.joint_names
        )

        device = (
            self.robot
            .data
            .joint_pos
            .device
        )

        dtype = (
            self.robot
            .data
            .joint_pos
            .dtype
        )

        # =====================================================
        # 3. GPU buffers
        # =====================================================

        self.position_buffer = (
            torch.empty(
                (
                    self.max_samples,
                    num_joints,
                ),
                device=device,
                dtype=dtype,
            )
        )

        self.torque_buffer = (
            torch.empty(
                (
                    self.max_samples,
                    num_joints,
                ),
                device=device,
                dtype=dtype,
            )
        )

        self.impulse_active_buffer = (
            torch.empty(
                self.max_samples,
                device=device,
                dtype=torch.bool,
            )
        )

        # Full robot configuration
        num_qpos = (
            self.env
            .sim
            .data
            .qpos
            .shape[1]
        )

        self.qpos_buffer = (
            torch.empty(
                (
                    self.max_samples,
                    num_qpos,
                ),
                device=device,
                dtype=(
                    self.env
                    .sim
                    .data
                    .qpos
                    .dtype
                ),
            )
        )

        # =====================================================
        # Contact data
        # =====================================================

        self.max_contacts = (
            self.env
            .sim
            .data
            .contact
            .pos
            .shape[0]
        )

        # Lưu vị trí contact
        self.contact_pos_buffer = (
            torch.empty(
                (
                    self.max_samples,
                    self.max_contacts,
                    3,
                ),
                device=device,
                dtype=dtype,
            )
        )

        # Lưu cặp geom tạo contact
        self.contact_geom_buffer = (
            torch.empty(
                (
                    self.max_samples,
                    self.max_contacts,
                    2,
                ),
                device=device,
                dtype=torch.int32,
            )
        )

        # Lưu normal contact force
        self.contact_normal_force_buffer = (
            torch.empty(
                (
                    self.max_samples,
                    self.max_contacts,
                ),
                device=device,
                dtype=dtype,
            )
        )

        # Số contact active
        self.nacon_buffer = (
            torch.empty(
                self.max_samples,
                device=device,
                dtype=torch.int32,
            )
        )

        # ID toàn bộ contact slot
        self.contact_ids_wp = (
            wp.array(
                np.arange(
                    self.max_contacts,
                    dtype=np.int32,
                ),
                dtype=wp.int32,
                device=(
                    self.env
                    .sim
                    .wp_device
                ),
            )
        )

        # Buffer contact force MuJoCo Warp
        self.contact_force_wp = (
            wp.zeros(
                self.max_contacts,
                dtype=wp.spatial_vector,
                device=(
                    self.env
                    .sim
                    .wp_device
                ),
            )
        )

        # Zero-copy Warp -> PyTorch
        self.contact_force_torch = (
            wp.to_torch(
                self.contact_force_wp
            )
        )

        # Time trên CPU
        self.time_buffer = (
            np.arange(
                1,
                self.max_samples + 1,
            )
            * self.physics_dt
        )

        self.fig_position = None
        self.fig_torque = None

    # =========================================================
    # RECORD DATA - physics rate
    # =========================================================

    def record(self):

        if (
            self.sample_count
            >= self.max_samples
        ):
            return

        i = self.sample_count

        # =====================================================
        # Joint position
        # =====================================================

        self.position_buffer[
            i
        ].copy_(
            self.robot.data.joint_pos[
                self.env_idx,
                self.joint_ids,
            ]
        )

        # =====================================================
        # Actuator torque
        # =====================================================

        self.torque_buffer[
            i
        ].copy_(
            self.robot.data.qfrc_actuator[
                self.env_idx,
                self.joint_ids,
            ]
        )

        # =====================================================
        # Full qpos
        # =====================================================

        self.qpos_buffer[
            i
        ].copy_(
            self.env.sim.data.qpos[
                self.env_idx
            ]
        )

        # =====================================================
        # Contact force thực tế
        # =====================================================

        mjwarp.contact_force(
            self.env.sim.wp_model,
            self.env.sim.wp_data,
            self.contact_ids_wp,
            False,
            self.contact_force_wp,
        )

        # Vị trí contact
        self.contact_pos_buffer[
            i
        ].copy_(
            self.env
            .sim
            .data
            .contact
            .pos[:]
        )

        # Cặp geom contact
        self.contact_geom_buffer[
            i
        ].copy_(
            self.env
            .sim
            .data
            .contact
            .geom[:]
        )

        # Normal contact force
        self.contact_normal_force_buffer[
            i
        ].copy_(
            self.contact_force_torch[
                :,
                0,
            ]
        )

        # Số contact active
        self.nacon_buffer[
            i
        ].copy_(
            self.env.sim.data.nacon[
                0
            ]
        )

        # =====================================================
        # Kiểm tra external impulse
        # =====================================================

        external_force = (
            self.env
            .sim
            .data
            .xfrc_applied[
                self.env_idx,
                :,
                0:3,
            ]
        )

        force_norm = (
            torch.linalg.vector_norm(
                external_force,
                dim=1,
            )
        )

        self.impulse_active_buffer[
            i
        ] = torch.any(
            force_norm
            > 1e-6
        )

        self.sample_count += 1

    # =========================================================
    # FINALIZE
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
        # 1. GPU -> CPU
        # =====================================================

        t = self.time_buffer[
            :n
        ]

        q = (
            self.position_buffer[
                :n
            ]
            .detach()
            .cpu()
            .numpy()
        )

        tau = (
            self.torque_buffer[
                :n
            ]
            .detach()
            .cpu()
            .numpy()
        )

        impulse_active = (
            self.impulse_active_buffer[
                :n
            ]
            .detach()
            .cpu()
            .numpy()
        )

        contact_pos = (
            self.contact_pos_buffer[
                :n
            ]
            .detach()
            .cpu()
            .numpy()
        )

        contact_geom = (
            self.contact_geom_buffer[
                :n
            ]
            .detach()
            .cpu()
            .numpy()
        )

        contact_normal_force = (
            self.contact_normal_force_buffer[
                :n
            ]
            .detach()
            .cpu()
            .numpy()
        )

        nacon = (
            self.nacon_buffer[
                :n
            ]
            .detach()
            .cpu()
            .numpy()
        )

        # =====================================================
        # 2. Tính COP
        # =====================================================

        (
            cop,
            total_normal_force,
        ) = compute_cop(
            self.env.sim.mj_model,
            contact_pos,
            contact_geom,
            contact_normal_force,
            nacon,
            min_normal_force=1.0,
        )

        valid_cop = (
            np.isfinite(
                cop[:, 0]
            )
            & np.isfinite(
                cop[:, 1]
            )
        )

        num_valid_cop = np.sum(
            valid_cop
        )

        print(
            f"COP valid: "
            f"{num_valid_cop}/{n} samples"
        )

        # =====================================================
        # 3. Tính support polygon offline
        # =====================================================

        support_polygons = (
            compute_support_polygons(
                self.env.sim.mj_model,
                contact_pos,
                contact_geom,
                contact_normal_force,
                nacon,
                min_normal_force=1.0,
            )
        )

        num_valid = sum(
            len(polygon) >= 3
            for polygon
            in support_polygons
        )

        print(
            f"Support polygons: "
            f"{num_valid}/{n} samples "
            f"có polygon hợp lệ."
        )

        print(
            "Support polygon "
            "sample đầu tiên:"
        )

        print(
            support_polygons[0]
        )

        # =====================================================
        # 4. Animation support polygon
        # =====================================================

        video_stride = int(
            round(
                self.env.step_dt
                / self.env.physics_dt
            )
        )

        video_polygons = (
            support_polygons[
                ::video_stride
            ]
        )

        video_time = t[
            ::video_stride
        ]

        video_cop = cop[
            ::video_stride
        ]

        (
            self.fig_support,
            self.support_animation,
        ) = (
            create_support_polygon_animation(
                video_polygons,
                video_time,
                video_cop,
            )
        )

        # =====================================================
        # 5. Output directory
        # =====================================================

        output_dir = Path(
            "logs/standing_eval"
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # =====================================================
        # 6. COP position + COP CSV
        # =====================================================

        cop_position_path = (
            output_dir
            / "cop_position.png"
        )

        plot_cop_position(
            cop_history=cop,
            time_history=t,
            support_polygons=(
                support_polygons
            ),
            impulse_active=(
                impulse_active
            ),
            output_path=(
                cop_position_path
            ),
        )

        # =====================================================
        # 7. File CSV tổng hợp cũ
        # =====================================================

        data = np.column_stack(
            (
                t,
                q,
                tau,
                cop,
                total_normal_force,
            )
        )

        header = [
            "time_s"
        ]

        header += [
            (
                f"{name}"
                f"_position_rad"
            )
            for name
            in self.joint_names
        ]

        header += [
            (
                f"{name}"
                f"_torque_Nm"
            )
            for name
            in self.joint_names
        ]

        header += [
            "cop_x_m",
            "cop_y_m",
            "total_normal_force_N",
        ]

        csv_path = (
            output_dir
            / "joint_data.csv"
        )

        np.savetxt(
            csv_path,
            data,
            delimiter=",",
            header=",".join(
                header
            ),
            comments="",
        )

        print(
            f"Joint data saved to: "
            f"{csv_path}"
        )

        # =====================================================
        # 8. CSV Joint Position cho MATLAB
        # =====================================================

        position_data = (
            np.column_stack(
                (
                    t,
                    q,
                    impulse_active.astype(
                        np.int8
                    ),
                )
            )
        )

        position_header = [
            "time_s"
        ]

        position_header += [
            (
                f"{name}"
                f"_position_rad"
            )
            for name
            in self.joint_names
        ]

        position_header += [
            "impulse_active"
        ]

        position_csv_path = (
            output_dir
            / "joint_position_data.csv"
        )

        np.savetxt(
            position_csv_path,
            position_data,
            delimiter=",",
            header=",".join(
                position_header
            ),
            comments="",
        )

        print(
            f"Joint position data "
            f"saved to: "
            f"{position_csv_path}"
        )

        # =====================================================
        # 9. CSV Joint Torque cho MATLAB
        # =====================================================

        torque_data = (
            np.column_stack(
                (
                    t,
                    tau,
                    impulse_active.astype(
                        np.int8
                    ),
                )
            )
        )

        torque_header = [
            "time_s"
        ]

        torque_header += [
            (
                f"{name}"
                f"_torque_Nm"
            )
            for name
            in self.joint_names
        ]

        torque_header += [
            "impulse_active"
        ]

        torque_csv_path = (
            output_dir
            / "joint_torque_data.csv"
        )

        np.savetxt(
            torque_csv_path,
            torque_data,
            delimiter=",",
            header=",".join(
                torque_header
            ),
            comments="",
        )

        print(
            f"Joint torque data "
            f"saved to: "
            f"{torque_csv_path}"
        )

        # =====================================================
        # 10. Plot joint position
        # =====================================================

        (
            self.fig_position,
            axes,
        ) = plt.subplots(
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

            self.add_impulse_regions(
                ax,
                t,
                impulse_active,
            )

            ax.set_title(
                name
            )

            ax.set_xlabel(
                "Time [s]"
            )

            ax.set_ylabel(
                "Position [rad]"
            )

            ax.grid(
                True
            )

            ax.legend()

        self.fig_position.savefig(
            (
                output_dir
                / "joint_position.png"
            ),
            dpi=150,
        )

        plt.close(
            self.fig_position
        )

        # =====================================================
        # 11. Plot joint torque
        # =====================================================

        (
            self.fig_torque,
            axes,
        ) = plt.subplots(
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

            self.add_impulse_regions(
                ax,
                t,
                impulse_active,
            )

            ax.set_title(
                name
            )

            ax.set_xlabel(
                "Time [s]"
            )

            ax.set_ylabel(
                "Torque [N.m]"
            )

            ax.grid(
                True
            )

            ax.legend()

        self.fig_torque.savefig(
            (
                output_dir
                / "joint_torque.png"
            ),
            dpi=150,
        )

        plt.close(
            self.fig_torque
        )

    # =========================================================
    # Physics hook
    # =========================================================

    def install_physics_hook(
        self,
    ):

        self._original_scene_update = (
            self.env.scene.update
        )

        def update_and_record(
            dt,
        ):

            self._original_scene_update(
                dt
            )

            self.record()

        self.env.scene.update = (
            update_and_record
        )

    def remove_physics_hook(
        self,
    ):

        if hasattr(
            self,
            "_original_scene_update",
        ):

            self.env.scene.update = (
                self._original_scene_update
            )

    def add_contact_visualization(
        self,
        visualizer,
        min_normal_force=1.0,
        force_scale=0.002,
    ):

        # =====================================================
        # 1. Lấy contact force hiện tại
        # =====================================================

        mjwarp.contact_force(
            self.env.sim.wp_model,
            self.env.sim.wp_data,
            self.contact_ids_wp,
            False,
            self.contact_force_wp,
        )

        ncon = int(
            self.env.sim.data.nacon[
                self.env_idx
            ].item()
        )

        if ncon <= 0:
            return

        contact_pos = (
            self.env.sim.data.contact.pos[
                :ncon
            ]
            .detach()
            .cpu()
            .numpy()
        )

        contact_geom = (
            self.env.sim.data.contact.geom[
                :ncon
            ]
            .detach()
            .cpu()
            .numpy()
        )

        contact_frame = (
            self.env.sim.data.contact.frame[
                :ncon
            ]
            .detach()
            .cpu()
            .numpy()
            .reshape(
                ncon,
                3,
                3,
            )
        )

        normal_force = (
            self.contact_force_torch[
                :ncon,
                0,
            ]
            .detach()
            .cpu()
            .numpy()
        )

        # =====================================================
        # 2. Vẽ contact bàn chân <-> terrain
        # =====================================================

        for j in range(ncon):

            fn = float(
                normal_force[j]
            )

            if fn <= min_normal_force:
                continue

            geom0 = int(
                contact_geom[
                    j,
                    0,
                ]
            )

            geom1 = int(
                contact_geom[
                    j,
                    1,
                ]
            )

            # MuJoCo:
            # contact normal hướng từ geom0 -> geom1
            normal = contact_frame[
                j,
                0,
                :
            ]

            # ---------------------------------------------
            # Xác định lực phản lực tác dụng lên bàn chân
            # ---------------------------------------------

            if geom0 == self.floor_geom_id:

                foot_geom = geom1

                force_direction = normal

            elif geom1 == self.floor_geom_id:

                foot_geom = geom0

                force_direction = -normal

            else:
                continue

            if foot_geom < 0:
                continue

            body_id = int(
                self.env.sim.mj_model.geom_bodyid[
                    foot_geom
                ]
            )

            if body_id not in (
                self.left_foot_body_id,
                self.right_foot_body_id,
            ):
                continue

            # =================================================
            # Contact point
            # =================================================

            point = contact_pos[
                j
            ]

            visualizer.add_sphere(
                center=point,
                radius=0.007,
                color=(
                    1.0,
                    0.0,
                    0.0,
                    1.0,
                ),
            )

            # =================================================
            # Normal contact force
            # =================================================

            force_end = (
                point
                + force_direction
                * fn
                * force_scale
            )

            visualizer.add_arrow(
                start=point,
                end=force_end,
                color=(
                    0.0,
                    1.0,
                    0.0,
                    1.0,
                ),
                width=0.006,
            )

    # =========================================================
    # Vùng xung lực trên đồ thị
    # =========================================================

    def add_impulse_regions(
        self,
        ax,
        t,
        impulse_active,
    ):

        active = (
            impulse_active.astype(
                np.int8
            )
        )

        padded = np.pad(
            active,
            (1, 1),
            constant_values=0,
        )

        changes = np.diff(
            padded
        )

        starts = np.where(
            changes == 1
        )[0]

        ends = (
            np.where(
                changes == -1
            )[0]
            - 1
        )

        for i, (
            start,
            end,
        ) in enumerate(
            zip(
                starts,
                ends,
            )
        ):

            ax.axvspan(
                t[start],
                t[end],
                color="red",
                alpha=0.30,
                label=(
                    "External impulse"
                    if i == 0
                    else None
                ),
            )


# =============================================================
# Gắn logger vào Native Viewer
# =============================================================

def with_joint_plots(
    base_viewer_class,
):

    class JointPlotViewer(
        base_viewer_class
    ):

        def setup(
            self,
        ):

            super().setup()

            self._joint_plotter = (
                JointPlotter(
                    env=self.env.unwrapped,
                    env_idx=self.env_idx,
                )
            )

        def run(
            self,
            num_steps=None,
            catch_sigint=True,
        ):

            super().run(
                num_steps=int(
                    8.0
                    / self.env
                    .unwrapped
                    .step_dt
                ),
                catch_sigint=(
                    catch_sigint
                ),
            )

        # =====================================================
        # Mỗi action step
        # =====================================================

        def _execute_step(
            self,
        ):

            success = (
                super()
                ._execute_step()
            )

            if success:

                self._joint_plotter.record()

                if (
                    self._joint_plotter
                    .sample_count
                    >=
                    self._joint_plotter
                    .max_samples
                ):

                    self._joint_plotter.finalize()

            return success

        # =====================================================
        # Close
        # =====================================================

        def close(
            self,
        ):

            if (
                self._joint_plotter
                .fig_position
                is not None
            ):

                plt.close(
                    self._joint_plotter
                    .fig_position
                )

            if (
                self._joint_plotter
                .fig_torque
                is not None
            ):

                plt.close(
                    self._joint_plotter
                    .fig_torque
                )

            if (
                self._joint_plotter
                .fig_support
                is not None
            ):

                plt.close(
                    self._joint_plotter
                    .fig_support
                )

            super().close()

    return JointPlotViewer