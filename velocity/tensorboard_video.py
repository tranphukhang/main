from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

from torch.utils.tensorboard import SummaryWriter
from tensorboard.compat.proto.summary_pb2 import Summary

from mjlab.utils.wrappers import VideoRecorder


class TensorBoardVideoRecorder(VideoRecorder):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # video_folder = <run_dir>/videos/train
        run_dir = Path(self.video_folder).parents[1]

        self._tb_writer = SummaryWriter(
            log_dir=str(run_dir),
            filename_suffix=".video",
        )

    def _finish_recording(self):

        # Copy frames trước vì VideoRecorder gốc sẽ clear chúng.
        frames = list(self.current_video_frames)

        # Vẫn cho mjlab lưu MP4 bình thường.
        super()._finish_recording()

        if not frames:
            return

        images = []

        for frame in frames:
            frame = np.asarray(frame)

            if frame.dtype != np.uint8:
                frame = (
                    np.clip(frame, 0.0, 1.0) * 255.0
                ).astype(np.uint8)

            images.append(
                Image.fromarray(frame).convert("RGB")
            )

        fps = self._wrapped_env.metadata.get(
            "render_fps",
            30,
        )

        duration_ms = int(1000 / fps)

        buffer = BytesIO()

        images[0].save(
            buffer,
            format="GIF",
            save_all=True,
            append_images=images[1:],
            duration=duration_ms,
            loop=0,
            optimize=False,
        )

        gif_bytes = buffer.getvalue()
        buffer.close()

        height, width = frames[0].shape[:2]

        summary = Summary(
            value=[
                Summary.Value(
                    tag="Training/Rollout",
                    image=Summary.Image(
                        height=height,
                        width=width,
                        colorspace=3,
                        encoded_image_string=gif_bytes,
                    ),
                )
            ]
        )

        writer = self._tb_writer._get_file_writer()

        writer.add_summary(
            summary,
            global_step=self.step_count,
        )

        self._tb_writer.flush()

    def close(self):
        super().close()

        self._tb_writer.flush()
        self._tb_writer.close()