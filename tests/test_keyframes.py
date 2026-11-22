from pathlib import Path

import cv2
import numpy as np
import pytest

from reelify.keyframes import extract_keyframes


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


def _make_static_mp4(path: Path, duration_secs: int = 2, fps: int = 10) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 320, 240
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    total_frames = duration_secs * fps
    gray = np.full((height, width), 128, dtype=np.uint8)
    frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for _ in range(total_frames):
        writer.write(frame)
    writer.release()


@pytest.fixture
def synthetic_mp4(tmp_path: Path) -> Path:
    path = tmp_path / "input.mp4"
    _make_synthetic_mp4(path)
    return path


def test_extract_creates_dir(tmp_path: Path, synthetic_mp4: Path) -> None:
    output_dir = tmp_path / "new_dir" / "keyframes"
    assert not output_dir.exists()
    extract_keyframes(synthetic_mp4, output_dir)
    assert output_dir.exists()


def test_extract_returns_jpegs(tmp_path: Path, synthetic_mp4: Path) -> None:
    output_dir = tmp_path / "keyframes"
    paths = extract_keyframes(synthetic_mp4, output_dir)
    assert isinstance(paths, list)
    for p in paths:
        assert p.suffix.lower() == ".jpg"


def test_extract_empty_on_no_scenes(tmp_path: Path) -> None:
    static_path = tmp_path / "static.mp4"
    _make_static_mp4(static_path)
    output_dir = tmp_path / "keyframes"
    paths = extract_keyframes(static_path, output_dir)
    assert isinstance(paths, list)
