import mujoco

from mjlab.viewer import NativeMujocoViewer


class CopEvalNativeViewer(NativeMujocoViewer):

    def setup(self):
        # Giữ nguyên toàn bộ setup của mjlab
        super().setup()

        # Hiển thị contact point
        self.viewer.opt.flags[
            mujoco.mjtVisFlag.mjVIS_CONTACTPOINT
        ] = 1

        # Hiển thị contact force
        self.viewer.opt.flags[
            mujoco.mjtVisFlag.mjVIS_CONTACTFORCE
        ] = 1