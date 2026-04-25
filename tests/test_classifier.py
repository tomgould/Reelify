import pytest
from reelify.analyser import AnalysisResult
from reelify.classifier import Chunk, classify


def make_result(scores: list[float], fps: float = 30.0, sample_every: int = 1, total_frames: int | None = None) -> AnalysisResult:
    if total_frames is None:
        total_frames = len(scores) * sample_every
    return AnalysisResult(
        scores=scores,
        fps=fps,
        total_frames=total_frames,
        sample_every=sample_every,
        width=1920,
        height=1080,
    )


def test_all_idle() -> None:
    result = make_result([0.1, 0.1, 0.1])
    chunks = classify(result, idle_threshold=0.5)
    assert chunks == [Chunk(start_frame=0, end_frame=3, active=False)]


def test_all_active() -> None:
    result = make_result([0.9, 0.9, 0.9])
    chunks = classify(result, idle_threshold=0.5)
    assert chunks == [Chunk(start_frame=0, end_frame=3, active=True)]


def test_margin_extends_active() -> None:
    fps = 30.0
    sample_every = 1
    scores = [0.1] * 10 + [0.9] + [0.1] * 10
    result = make_result(scores, fps=fps, sample_every=sample_every)
    margin_secs = 0.3
    margin_frames = round(margin_secs * fps)  # 9
    chunks = classify(result, idle_threshold=0.5, margin_secs=margin_secs)

    assert len(chunks) == 3
    assert chunks[1].active is True
    assert chunks[1].start_frame == max(0, 10 - margin_frames)
    assert chunks[1].end_frame == min(21, 10 + margin_frames + 1)


def test_adjacent_active_merged() -> None:
    fps = 30.0
    sample_every = 1
    scores = [0.1] * 5 + [0.9] + [0.1] * 5 + [0.9] + [0.1] * 5
    result = make_result(scores, fps=fps, sample_every=sample_every)
    margin_secs = 0.3
    margin_frames = round(margin_secs * fps)  # 9
    chunks = classify(result, idle_threshold=0.5, margin_secs=margin_secs)

    assert len(chunks) == 1
    assert chunks[0].active is True
    assert chunks[0].start_frame == 0
    assert chunks[0].end_frame == 17


def test_chunks_cover_all_frames() -> None:
    result = make_result([0.1, 0.9, 0.1, 0.9, 0.1], fps=30.0, sample_every=1, total_frames=5)
    chunks = classify(result, idle_threshold=0.5, margin_secs=0.0)
    total = sum(chunk.end_frame - chunk.start_frame for chunk in chunks)
    assert total == result.total_frames
