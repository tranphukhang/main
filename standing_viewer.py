from mjlab.viewer.viser.viewer import ViserPlayViewer

from robot_cfg import ACTUATED_JOINTS


class StandingViserViewer(ViserPlayViewer):

    def setup(self):
        super().setup()

        env = self.env.unwrapped
        robot = env.scene["robot"]

        # Lấy đúng 8 joint chủ động theo thứ tự ACTUATED_JOINTS.
        joint_ids, joint_names = robot.find_joints(
            ACTUATED_JOINTS,
            preserve_order=True,
        )

        self._monitor_joint_ids = joint_ids
        self._monitor_joint_names = joint_names

        # -----------------------------------------------------
        # Custom GUI
        # -----------------------------------------------------

        with self._server.gui.add_folder("Joint Monitor"):
            self._joint_monitor_html = self._server.gui.add_html("")

        self._update_joint_monitor()

    def sync_env_to_viewer(self):
        # Giữ toàn bộ chức năng Viser mặc định.
        super().sync_env_to_viewer()

        # Viewer chạy khoảng 60 Hz.
        # Update bảng khoảng 10 Hz là đủ để quan sát.
        if self._counter % 6 == 0:
            self._update_joint_monitor()

    def _update_joint_monitor(self):

        env = self.env.unwrapped
        robot = env.scene["robot"]

        env_idx = self._scene.env_idx
        joint_ids = self._monitor_joint_ids

        # -----------------------------------------------------
        # Read data
        # -----------------------------------------------------

        q_target = (
            robot.data.joint_pos_target[env_idx, joint_ids]
            .detach()
            .cpu()
            .tolist()
        )

        q_actual = (
            robot.data.joint_pos[env_idx, joint_ids]
            .detach()
            .cpu()
            .tolist()
        )

        torque = (
            robot.data.qfrc_actuator[env_idx, joint_ids]
            .detach()
            .cpu()
            .tolist()
        )

        # -----------------------------------------------------
        # Build HTML table
        # -----------------------------------------------------

        rows = []

        for name, qt, qa, tau in zip(
            self._monitor_joint_names,
            q_target,
            q_actual,
            torque,
        ):
            rows.append(
                f"""
                <tr>
                    <td>{name}</td>
                    <td style="text-align:right">{qt:+.4f}</td>
                    <td style="text-align:right">{qa:+.4f}</td>
                    <td style="text-align:right">{tau:+.3f}</td>
                </tr>
                """
            )

        self._joint_monitor_html.content = f"""
        <div style="font-size:0.85em;">
            <table style="
                width:100%;
                border-collapse:collapse;
                font-family:monospace;
            ">
                <thead>
                    <tr>
                        <th style="text-align:left">Joint</th>
                        <th style="text-align:right">q target [rad]</th>
                        <th style="text-align:right">q actual [rad]</th>
                        <th style="text-align:right">Torque [N·m]</th>
                    </tr>
                </thead>

                <tbody>
                    {''.join(rows)}
                </tbody>
            </table>
        </div>
        """