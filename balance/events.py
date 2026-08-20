from __future__ import annotations

import torch

from mjlab.envs import mdp


class BalanceBodyImpulse(
    mdp.apply_body_impulse
):
    """
    Body impulse dành cho bài toán balance recovery.

    Giữ nguyên toàn bộ lifecycle của mdp.apply_body_impulse:

        cooldown
            ->
        trigger
            ->
        sustain
            ->
        expire

    Phần bổ sung:

    Khi một impulse vừa được trigger, lưu lại pose của
    root/base ngay TRƯỚC khi lực đẩy bắt đầu.

    Reference gồm:

        position:
            x_ref, y_ref, z_ref

        orientation:
            quaternion q_ref

    Joint configuration q KHÔNG được lưu làm reference.
    """

    def __init__(
        self,
        cfg,
        env,
    ):

        # -----------------------------------------------------
        # Khởi tạo native body impulse
        # -----------------------------------------------------

        super().__init__(
            cfg,
            env,
        )

        self._env = env

        # -----------------------------------------------------
        # Reference position trước push
        # -----------------------------------------------------

        self.pre_push_pos_w = torch.zeros(
            env.num_envs,
            3,
            device=env.device,
            dtype=torch.float32,
        )

        # -----------------------------------------------------
        # Reference orientation trước push
        # Quaternion convention: (w, x, y, z)
        # -----------------------------------------------------

        self.pre_push_quat_w = torch.zeros(
            env.num_envs,
            4,
            device=env.device,
            dtype=torch.float32,
        )

        # Quaternion identity mặc định
        self.pre_push_quat_w[:, 0] = 1.0

        # -----------------------------------------------------
        # Reference đã tồn tại hay chưa
        # -----------------------------------------------------

        self.pre_push_pose_valid = torch.zeros(
            env.num_envs,
            device=env.device,
            dtype=torch.bool,
        )

        # -----------------------------------------------------
        # Expose buffer cho observation / reward
        # -----------------------------------------------------

        env._pre_push_pos_w = (
            self.pre_push_pos_w
        )

        env._pre_push_quat_w = (
            self.pre_push_quat_w
        )

        env._pre_push_pose_valid = (
            self.pre_push_pose_valid
        )

        # _active được tạo bởi mdp.apply_body_impulse
        # True  = force đang tác động
        # False = không có force
        env._push_active = self._active

    def __call__(
        self,
        env,
        env_ids,
        **kwargs,
    ):

        # =====================================================
        # 1. Lưu pose hiện tại TRƯỚC khi impulse event chạy
        # =====================================================

        root_pose_before = (
            self._asset.data.root_link_pose_w.clone()
        )

        # Trạng thái impulse trước event
        was_active = self._active.clone()

        # =====================================================
        # 2. Chạy native impulse lifecycle
        # =====================================================

        super().__call__(
            env,
            env_ids,
            **kwargs,
        )

        # =====================================================
        # 3. Phát hiện rising edge:
        #
        # False -> True
        #
        # nghĩa là impulse vừa được trigger
        # =====================================================

        just_triggered = (
            self._active
            & (~was_active)
        )

        if not just_triggered.any():
            return

        trigger_ids = (
            just_triggered
            .nonzero(as_tuple=False)
            .squeeze(-1)
        )

        # =====================================================
        # 4. Lưu PRE-PUSH pose
        # =====================================================

        self.pre_push_pos_w[
            trigger_ids
        ] = root_pose_before[
            trigger_ids,
            0:3,
        ]

        self.pre_push_quat_w[
            trigger_ids
        ] = root_pose_before[
            trigger_ids,
            3:7,
        ]

        self.pre_push_pose_valid[
            trigger_ids
        ] = True

    def reset(
        self,
        env_ids=None,
    ):

        # -----------------------------------------------------
        # Reset native impulse state
        # -----------------------------------------------------

        super().reset(
            env_ids=env_ids,
        )

        if env_ids is None:
            env_ids = slice(None)

        # -----------------------------------------------------
        # Episode mới:
        # chưa có pre-push reference
        # -----------------------------------------------------

        self.pre_push_pose_valid[
            env_ids
        ] = False

        self.pre_push_pos_w[
            env_ids
        ] = 0.0

        self.pre_push_quat_w[
            env_ids
        ] = 0.0

        self.pre_push_quat_w[
            env_ids,
            0
        ] = 1.0