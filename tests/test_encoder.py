import random
import subprocess
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

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


@patch("reelify.encoder.subprocess.run")
def test_encode_parallel_order(mock_run: MagicMock, tmp_path: Path) -> None:
    input_path = tmp_path / "input.mp4"
    input_path.write_text("fake")

    segment_calls: list[list[str]] = []
    concat_lines: list[str] = []

    def side_effect(cmd: list[str], **kwargs: object) -> MagicMock:
        result = MagicMock()
        result.returncode = 0
        result.stderr = b""
        result.stdout = b""

        if cmd[0] == "ffprobe":
            result.stdout = b"audio\n"
            return result

        if "-f" in cmd and "concat" in cmd:
            output_path = Path(cmd[-1])
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text("fake_output")
            concat_list_path = Path(cmd[cmd.index("-i") + 1])
            if concat_list_path.exists():
                concat_lines.extend(concat_list_path.read_text().strip().split("\n"))
            return result

        # Segment encoding
        segment_calls.append(cmd)
        time.sleep(random.uniform(0.01, 0.03))
        segment_file = Path(cmd[-1])
        segment_file.parent.mkdir(parents=True, exist_ok=True)
        segment_file.write_text("fake_segment")
        return result

    mock_run.side_effect = side_effect

    segments = [
        Segment(start_frame=0, end_frame=10, speed=1.0),
        Segment(start_frame=10, end_frame=20, speed=2.0),
        Segment(start_frame=20, end_frame=30, speed=1.0),
    ]
    output = tmp_path / "output.mp4"

    progress_calls: list[tuple[int, int]] = []
    def progress_callback(done: int, total: int) -> None:
        progress_calls.append((done, total))

    encode(input_path, output, segments, fps=10.0, width=320, height=240, progress_callback=progress_callback)

    assert output.exists()
    assert len(segment_calls) == 3

    # Verify concat list order
    assert len(concat_lines) == 3
    for i, line in enumerate(concat_lines):
        assert f"seg_{i:04d}.mp4" in line

    assert progress_calls == [(1, 3), (2, 3), (3, 3)]
