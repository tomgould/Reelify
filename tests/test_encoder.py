import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest

from reelify.encoder import encode
from reelify.speed_map import Segment


def _make_synthetic_mp4(path: Path, duration_secs: int = 3, fps: int = 10) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 320, 240
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    total_frames = duration_secs * fps
    for i in range(total_frames):
        gray = np.full((height, width), (i * 17) % 256, dtype=np.uint8)
        frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        writer.write(frame)
    writer.release()


@pytest.fixture
def synthetic_mp4(tmp_path: Path) -> Path:
    path = tmp_path / "input.mp4"
    _make_synthetic_mp4(path)
    return path


def test_encode_produces_output(tmp_path: Path, synthetic_mp4: Path) -> None:
    output = tmp_path / "output.mp4"
    segments = [Segment(start_frame=0, end_frame=30, speed=1.0)]
    encode(synthetic_mp4, output, segments, fps=10.0, width=320, height=240)
    assert output.exists()
    assert output.stat().st_size > 1024


def test_encode_skips_idle_segments(tmp_path: Path, synthetic_mp4: Path) -> None:
    output = tmp_path / "output.mp4"
    segments = [
        Segment(start_frame=0, end_frame=10, speed=0.0),
        Segment(start_frame=10, end_frame=20, speed=1.0),
        Segment(start_frame=20, end_frame=30, speed=0.0),
    ]
    encode(synthetic_mp4, output, segments, fps=10.0, width=320, height=240)
    assert output.exists()
    assert output.stat().st_size > 1024


def test_encode_with_speed(tmp_path: Path, synthetic_mp4: Path) -> None:
    output = tmp_path / "output.mp4"
    segments = [Segment(start_frame=0, end_frame=30, speed=2.0)]
    encode(synthetic_mp4, output, segments, fps=10.0, width=320, height=240)
    assert output.exists()
    assert output.stat().st_size > 1024


def test_invalid_segments_all_idle(tmp_path: Path, synthetic_mp4: Path) -> None:
    output = tmp_path / "output.mp4"
    segments = [Segment(start_frame=0, end_frame=30, speed=0.0)]
    with pytest.raises(ValueError, match="No active segments to encode"):
        encode(synthetic_mp4, output, segments, fps=10.0, width=320, height=240)


@pytest.mark.skipif(
    subprocess.run(["ffmpeg", "-version"], capture_output=True).returncode != 0,
    reason="requires ffmpeg",
)
def test_encode_webm_input(tmp_path: Path) -> None:
    webm_path = tmp_path / "test.webm"
    output = tmp_path / "output.mp4"
    cmd = [
        "ffmpeg",
        "-y",
        "-f", "lavfi",
        "-i", "testsrc=duration=2:size=320x240:rate=10",
        "-c:v", "libvpx-vp9",
        "-b:v", "0",
        "-crf", "30",
        str(webm_path),
    ]
    result = subprocess.run(cmd, capture_output=True, check=False)
    if result.returncode != 0:
        pytest.skip("requires ffmpeg webm support")

    segments = [Segment(start_frame=0, end_frame=20, speed=1.0)]
    encode(webm_path, output, segments, fps=10.0, width=320, height=240)
    assert output.exists()
    assert output.stat().st_size > 1024
