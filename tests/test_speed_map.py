import math
import pytest
from reelify.analyser import AnalysisResult
from reelify.classifier import Chunk
from reelify.speed_map import IDLE_SPEED, ACTIVE_SPEED, MAX_SPEED, Segment, build_speed_map, segment_duration_secs


def make_result(total_frames: int, fps: float = 30.0) -> AnalysisResult:
    return AnalysisResult(
        scores=[],
        fps=fps,
        total_frames=total_frames,
        sample_every=1,
        width=1920,
        height=1080,
    )


def test_idle_chunks_cut() -> None:
    result = make_result(total_frames=30)
    chunks = [Chunk(start_frame=0, end_frame=30, active=False)]
    segments = build_speed_map(chunks, result, max_duration_secs=10.0)
    assert segments == [Segment(start_frame=0, end_frame=30, speed=IDLE_SPEED)]


def test_active_chunks_normal() -> None:
    result = make_result(total_frames=30)
    chunks = [Chunk(start_frame=0, end_frame=30, active=True)]
    segments = build_speed_map(chunks, result, max_duration_secs=10.0)
    assert segments == [Segment(start_frame=0, end_frame=30, speed=ACTIVE_SPEED)]


def test_speed_scaled_when_over_budget() -> None:
    fps = 30.0
    result = make_result(total_frames=300, fps=fps)  # 10 seconds of frames
    chunks = [Chunk(start_frame=0, end_frame=300, active=True)]
    max_duration_secs = 5.0
    segments = build_speed_map(chunks, result, max_duration_secs=max_duration_secs)
    assert len(segments) == 1
    assert segments[0].speed == pytest.approx(2.0)


def test_speed_capped_at_8x() -> None:
    fps = 30.0
    result = make_result(total_frames=3000, fps=fps)  # 100 seconds of frames
    chunks = [Chunk(start_frame=0, end_frame=3000, active=True)]
    max_duration_secs = 5.0
    segments = build_speed_map(chunks, result, max_duration_secs=max_duration_secs)
    assert len(segments) == 1
    assert segments[0].speed == MAX_SPEED


def test_total_output_duration_respected() -> None:
    fps = 30.0
    result = make_result(total_frames=600, fps=fps)  # 20 seconds of frames
    chunks = [Chunk(start_frame=0, end_frame=600, active=True)]
    max_duration_secs = 5.0
    segments = build_speed_map(chunks, result, max_duration_secs=max_duration_secs)
    total_output = sum(segment_duration_secs(seg, fps) for seg in segments)
    assert total_output <= max_duration_secs + 1e-9
