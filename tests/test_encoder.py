import math
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


# --- Frame-accuracy regression tests ---------------------------------------
#
# Each frame of the numbered clip shows its own index as eight black or white
# stripes, one bit each, so a decoded output frame says which source frame it
# came from.

_BIT_WIDTH = 40
_NUMBERED_FPS = 10.0


def _make_numbered_clip(path: Path, frames: int = 60) -> None:
    lum = f"255*mod(floor(N/pow(2\\,floor(X/{_BIT_WIDTH})))\\,2)"
    cmd = [
        "ffmpeg", "-v", "error", "-y",
        "-f", "lavfi",
        "-i", f"color=black:s={_BIT_WIDTH * 8}x32:r={_NUMBERED_FPS:g}:d={frames / _NUMBERED_FPS:g}",
        "-vf", f"format=gray,geq=lum='{lum}'",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        str(path),
    ]
    subprocess.run(cmd, capture_output=True, check=True)


def _frame_numbers(path: Path) -> list[int]:
    width, height = _BIT_WIDTH * 8, 32
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "gray", "-"],
        capture_output=True, check=True,
    ).stdout
    frames = np.frombuffer(raw, dtype=np.uint8).reshape(-1, height, width)
    numbers = []
    for frame in frames:
        value = 0
        for bit in range(8):
            stripe = frame[:, bit * _BIT_WIDTH + 10:(bit + 1) * _BIT_WIDTH - 10]
            if stripe.mean() > 128:
                value |= 1 << bit
        numbers.append(value)
    return numbers


def _duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout
    return float(out)


@pytest.fixture
def numbered_mp4(tmp_path: Path) -> Path:
    path = tmp_path / "numbered.mp4"
    _make_numbered_clip(path)
    return path


def test_numbered_clip_decodes(numbered_mp4: Path) -> None:
    assert _frame_numbers(numbered_mp4) == list(range(60))


def test_plain_cut_keeps_exact_frames(tmp_path: Path, numbered_mp4: Path) -> None:
    output = tmp_path / "output.mp4"
    segments = [Segment(start_frame=7, end_frame=23, speed=1.0)]
    encode(numbered_mp4, output, segments, fps=_NUMBERED_FPS, width=320, height=32)
    assert _frame_numbers(output) == list(range(7, 23))


@pytest.mark.parametrize(
    ("start", "end", "speed", "expected"),
    [
        (20, 40, 2.0, list(range(20, 40, 2))),
        (13, 47, 3.0, list(range(13, 47, 3))),
        (1, 59, 8.0, list(range(1, 59, 8))),
    ],
)
def test_speed_up_comes_from_its_own_frames(
    tmp_path: Path, numbered_mp4: Path, start: int, end: int, speed: float, expected: list[int]
) -> None:
    output = tmp_path / "output.mp4"
    segments = [Segment(start_frame=start, end_frame=end, speed=speed)]
    encode(numbered_mp4, output, segments, fps=_NUMBERED_FPS, width=320, height=32)
    assert _frame_numbers(output) == expected
    assert _duration(output) == pytest.approx(len(expected) / _NUMBERED_FPS, abs=0.01)


def test_speed_up_at_fractional_speed(tmp_path: Path, numbered_mp4: Path) -> None:
    output = tmp_path / "output.mp4"
    segments = [Segment(start_frame=14, end_frame=46, speed=2.83)]
    encode(numbered_mp4, output, segments, fps=_NUMBERED_FPS, width=320, height=32)
    numbers = _frame_numbers(output)
    assert numbers[0] == 14
    assert all(14 <= n < 46 for n in numbers)
    assert numbers == sorted(set(numbers))
    assert len(numbers) == math.ceil((46 - 14) / 2.83)


def test_cuts_and_speed_ups_join_in_order(tmp_path: Path, numbered_mp4: Path) -> None:
    output = tmp_path / "output.mp4"
    segments = [
        Segment(start_frame=0, end_frame=10, speed=0.0),
        Segment(start_frame=10, end_frame=20, speed=1.0),
        Segment(start_frame=20, end_frame=30, speed=0.0),
        Segment(start_frame=30, end_frame=54, speed=3.0),
        Segment(start_frame=54, end_frame=60, speed=0.0),
    ]
    encode(numbered_mp4, output, segments, fps=_NUMBERED_FPS, width=320, height=32)
    assert _frame_numbers(output) == list(range(10, 20)) + list(range(30, 54, 3))
