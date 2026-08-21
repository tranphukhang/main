from pathlib import Path
from datetime import datetime


def create_log_directory(
    root="balance/logs",
):
    """
    Tạo thư mục:

    balance/logs/
        YYYY-MM-DD_HH-MM-SS/

    """

    timestamp = datetime.now().strftime(
        "%Y-%m-%d_%H-%M-%S"
    )

    log_dir = (
        Path(root)
        / timestamp
    )

    video_dir = (
        log_dir
        / "video"
    )

    video_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    return video_dir



def create_video_path(
    video_dir,
    filename="balance.mp4",
):
    """
    Trả về đường dẫn file video
    """

    return (
        Path(video_dir)
        / filename
    )