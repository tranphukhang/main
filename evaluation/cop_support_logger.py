from pathlib import Path

import mujoco_warp as mjwarp
import numpy as np
import torch
import warp as wp

from evaluation.cop import compute_cop
from evaluation.support_polygon import (
    compute_support_polygons,
    create_support_polygon_animation,
)
from evaluation.stability_timeseries import (
    create_stability_timeseries_animation,
    create_stability_velocity_animation,
)


class CopSupportLogger:

    def __init__(
        self,
        env,
        output_dir,
        env_idx=0,
        min_normal_force=1.0,
    ):
        self.env = env
        self.env_idx = env_idx
        self.output_dir = Path(output_dir)
        self.min_normal_force = min_normal_force

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.physics_dt = env.physics_dt

        self.max_samples = int(
            np.ceil(
                env.cfg.episode_length_s
                / self.physics_dt
            )
        )

        self.sample_count = 0
        self.finalized = False

        device = env.device

        # =====================================================
        # Contact buffers
        # =====================================================

        self.max_contacts = (
            env.sim.data.contact.pos.shape[0]
        )

        contact_dtype = (
            env.sim.data.contact.pos.dtype
        )

        geom_dtype = (
            env.sim.data.contact.geom.dtype
        )

        nacon_dtype = (
            env.sim.data.nacon.dtype
        )

        self.contact_pos_buffer = torch.empty(
            (
                self.max_samples,
                self.max_contacts,
                3,
            ),
            device=device,
            dtype=contact_dtype,
        )

        self.contact_geom_buffer = torch.empty(
            (
                self.max_samples,
                self.max_contacts,
                2,
            ),
            device=device,
            dtype=geom_dtype,
        )

        self.normal_force_buffer = torch.empty(
            (
                self.max_samples,
                self.max_contacts,
            ),
            device=device,
            dtype=contact_dtype,
        )

        self.nacon_buffer = torch.empty(
            self.max_samples,
            device=device,
            dtype=nacon_dtype,
        )

        # =====================================================
        # COM toàn robot
        # =====================================================

        self.robot = env.scene["robot"]

        self.root_body_id = (
            self.robot.indexing.root_body_id
        )

        self.com_pos_buffer = torch.empty(
            (
                self.max_samples,
                3,
            ),
            device=device,
            dtype=contact_dtype,
        )

        # =====================================================
        # Warp contact-force buffer
        # =====================================================

        self.contact_ids_wp = wp.array(
            np.arange(
                self.max_contacts,
                dtype=np.int32,
            ),
            dtype=wp.int32,
            device=env.sim.wp_device,
        )

        self.contact_force_wp = wp.zeros(
            self.max_contacts,
            dtype=wp.spatial_vector,
            device=env.sim.wp_device,
        )

        self.contact_force_torch = wp.to_torch(
            self.contact_force_wp
        )

    # =========================================================
    # Record tại mỗi physics substep
    # =========================================================

    def record(self):

        if self.sample_count >= self.max_samples:
            return

        i = self.sample_count

        mjwarp.contact_force(
            self.env.sim.wp_model,
            self.env.sim.wp_data,
            self.contact_ids_wp,
            False,
            self.contact_force_wp,
        )

        self.contact_pos_buffer[i].copy_(
            self.env.sim.data.contact.pos
        )

        self.contact_geom_buffer[i].copy_(
            self.env.sim.data.contact.geom
        )

        self.normal_force_buffer[i].copy_(
            self.contact_force_torch[:, 0]
        )

        self.nacon_buffer[i].copy_(
            self.env.sim.data.nacon[
                self.env_idx
            ]
        )

        # Hình chiếu COM toàn robot lên mặt phẳng XY
        self.com_pos_buffer[i].copy_(
            self.env.sim.data.subtree_com[
                self.env_idx,
                self.root_body_id,
                :3,
            ]
        )

        self.sample_count += 1

    # =========================================================
    # Hook vào physics loop
    # =========================================================

    def install_physics_hook(self):

        self._original_scene_update = (
            self.env.scene.update
        )

        def update_and_record(dt):

            self._original_scene_update(dt)
            self.record()

        self.env.scene.update = update_and_record

    def remove_physics_hook(self):

        if hasattr(
            self,
            "_original_scene_update",
        ):
            self.env.scene.update = (
                self._original_scene_update
            )

    # =========================================================
    # Xử lý và xuất video
    # =========================================================

    def finalize(self):

        if self.finalized:
            return

        if self.sample_count == 0:
            print("Không thu được dữ liệu contact.")
            return

        self.finalized = True
        n = self.sample_count

        print(
            f"Contact samples collected: {n}"
        )

        contact_pos = (
            self.contact_pos_buffer[:n]
            .detach()
            .cpu()
            .numpy()
        )

        contact_geom = (
            self.contact_geom_buffer[:n]
            .detach()
            .cpu()
            .numpy()
        )

        normal_force = (
            self.normal_force_buffer[:n]
            .detach()
            .cpu()
            .numpy()
        )

        nacon = (
            self.nacon_buffer[:n]
            .detach()
            .cpu()
            .numpy()
        )

        com_pos = (
            self.com_pos_buffer[:n]
            .detach()
            .cpu()
            .numpy()
        )

        time_history = (
            np.arange(1, n + 1)
            * self.physics_dt
        )

        # =====================================================
        # COP
        # =====================================================

        cop_history, _ = compute_cop(
            self.env.sim.mj_model,
            contact_pos,
            contact_geom,
            normal_force,
            nacon,
            min_normal_force=self.min_normal_force,
        )

        # =====================================================
        # Support polygon
        # =====================================================

        support_polygons = (
            compute_support_polygons(
                self.env.sim.mj_model,
                contact_pos,
                contact_geom,
                normal_force,
                nacon,
                min_normal_force=(
                    self.min_normal_force
                ),
            )
        )

        valid_cop = np.all(
            np.isfinite(cop_history),
            axis=1,
        )

        valid_polygon = np.array(
            [
                len(polygon) >= 3
                for polygon in support_polygons
            ]
        )

        print(
            f"Valid COP: "
            f"{np.sum(valid_cop)}/{n}"
        )

        print(
            f"Valid support polygon: "
            f"{np.sum(valid_polygon)}/{n}"
        )

        # =====================================================
        # Downsample 2000 Hz -> 50 Hz
        # =====================================================

        video_stride = int(
            round(
                self.env.step_dt
                / self.env.physics_dt
            )
        )

        video_polygons = (
            support_polygons[::video_stride]
        )

        video_cop = (
            cop_history[::video_stride]
        )

        video_com_pos = (
            com_pos[::video_stride]
        )

        video_com = video_com_pos[:, :2]

        video_time = (
            time_history[::video_stride]
        )

        # =====================================================
        # Capture Point theo Linear Inverted Pendulum Model
        # =====================================================

        num_video_samples = len(video_com)

        if num_video_samples >= 3:
            com_velocity_xy = np.gradient(
                video_com,
                self.env.step_dt,
                axis=0,
                edge_order=2,
            )
        elif num_video_samples == 2:
            com_velocity_xy = np.gradient(
                video_com,
                self.env.step_dt,
                axis=0,
            )
        else:
            com_velocity_xy = np.zeros_like(
                video_com
            )

        # Terrain của task balance nằm tại z = 0
        ground_height = 0.0

        # LIPM giả thiết chiều cao COM không đổi
        # Lấy chiều cao COM trung bình trong 1 giây đầu
        initial_duration = 1.0

        initial_samples = min(
            int(
                round(
                    initial_duration
                    / self.env.step_dt
                )
            ),
            len(video_com_pos),
        )

        com_height = float(
            np.mean(
                video_com_pos[
                    :initial_samples,
                    2,
                ]
            )
            - ground_height
        )

        if com_height <= 0.0:
            raise ValueError(
                f"Invalid COM height: {com_height}"
            )

        gravity = float(
            np.linalg.norm(
                self.env.sim.mj_model.opt.gravity
            )
        )

        omega_0 = np.sqrt(
            gravity / com_height
        )

        video_capture_point = (
            video_com
            + com_velocity_xy / omega_0
        )

        if num_video_samples >= 3:
            capture_point_velocity_xy = (
                np.gradient(
                    video_capture_point,
                    self.env.step_dt,
                    axis=0,
                    edge_order=2,
                )
            )
        elif num_video_samples == 2:
            capture_point_velocity_xy = (
                np.gradient(
                    video_capture_point,
                    self.env.step_dt,
                    axis=0,
                )
            )
        else:
            capture_point_velocity_xy = (
                np.zeros_like(
                    video_capture_point
                )
            )

        print(
            f"LIPM reference COM height "
            f"(mean of first "
            f"{initial_duration:.1f} s): "
            f"{com_height:.4f} m"
        )

        print(
            f"LIPM natural frequency: "
            f"{omega_0:.4f} rad/s"
        )

        video_fps = int(
            round(1.0 / self.env.step_dt)
        )

        output_path = (
            self.output_dir
            / "support_polygon.mp4"
        )

        create_support_polygon_animation(
            polygons=video_polygons,
            time_history=video_time,
            cop_history=video_cop,
            com_history=video_com,
            capture_point_history=(
                video_capture_point
            ),
            output_path=output_path,
            fps=video_fps,
        )

        timeseries_output_path = (
            self.output_dir
            / "stability_timeseries.mp4"
        )

        create_stability_timeseries_animation(
            time_history=video_time,
            com_history=video_com,
            cop_history=video_cop,
            capture_point_history=(
                video_capture_point
            ),
            output_path=(
                timeseries_output_path
            ),
            fps=video_fps,
        )

        velocity_output_path = (
            self.output_dir
            / "stability_velocity.mp4"
        )

        create_stability_velocity_animation(
            time_history=video_time,
            com_velocity_history=com_velocity_xy,
            capture_point_velocity_history=(
                capture_point_velocity_xy
            ),
            output_path=velocity_output_path,
            fps=video_fps,
        )