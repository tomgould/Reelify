from pathlib import Path

import cv2
import numpy as np
import pytest

from reelify.analyser import analyse, AnalysisResult


def _make_test_video(path: Path, frames: list[np.ndarray], fps: float = 30.0) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError("Failed to open VideoWriter")
    for frame in frames:
        writer.write(frame)
    writer.release()


@pytest.fixture
def sample_video(tmp_path: Path) -> Path:
    black = np.zeros((100, 100, 3), dtype=np.uint8)
    white = np.full((100, 100, 3), 255, dtype=np.uint8)
    frames = [black, black, white, white]
    video_path = tmp_path / "test.mp4"
    _make_test_video(video_path, frames)
    return video_path


def test_analyse_returns_result(sample_video: Path) -> None:
    result = analyse(sample_video, sample_every=1)
    assert isinstance(result, AnalysisResult)
    assert len(result.scores) == 4
    assert result.fps == pytest.approx(30.0, abs=1.0)
    assert result.total_frames == 4
    assert result.sample_every == 1
    assert result.width == 100
    assert result.height == 100


def test_first_score_is_zero(sample_video: Path) -> None:
    result = analyse(sample_video, sample_every=1)
    assert result.scores[0] == 0.0


def test_idle_frames_low_score(sample_video: Path) -> None:
    result = analyse(sample_video, sample_every=1)
    assert result.scores[1] < 0.01


def test_active_frames_high_score(sample_video: Path) -> None:
    result = analyse(sample_video, sample_every=1)
    assert result.scores[2] > 0.9


def test_invalid_path_raises() -> None:
    with pytest.raises(ValueError):
        analyse(Path("nonexistent.mp4"))
