from __future__ import annotations

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from mjlab.utils.wrappers import VideoRecorder


class TensorBoardVideoRecorder(VideoRecorder):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        # video_folder = <run_dir>/videos/train
        self.writer = SummaryWriter(
            log_dir=str(self.video_folder.parents[1]),
        )

    def _finish_recording(self) -> None:

        if self.current_video_frames:

            frames = []

            for frame in self.current_video_frames:

                frame = np.asarray(frame)

                if frame.dtype != np.uint8:
                    frame = (
                        np.clip(frame, 0.0, 1.0) * 255.0
                    ).astype(np.uint8)

                frames.append(frame)

            video = (
                torch.from_numpy(np.stack(frames))
                .permute(0, 3, 1, 2)
                .unsqueeze(0)
            )

            fps = self._wrapped_env.metadata.get(
                "render_fps",
                30,
            )

            self.writer.add_video(
                "Train/video",
                video,
                global_step=self.step_count,
                fps=fps,
            )

            self.writer.flush()

        # Giữ nguyên ghi MP4 và reset recorder của MjLab.
        super()._finish_recording()

    def close(self) -> None:

        try:
            super().close()

        finally:
            self.writer.close()