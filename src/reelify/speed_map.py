from dataclasses import dataclass
from reelify.analyser import AnalysisResult
from reelify.classifier import Chunk


IDLE_SPEED = 0.0
ACTIVE_SPEED = 1.0
MAX_SPEED = 8.0


@dataclass
class Segment:
    start_frame: int
    end_frame: int
    speed: float


def segment_duration_secs(seg: Segment, fps: float) -> float:
    if seg.speed <= 0.0:
        return 0.0
    return (seg.end_frame - seg.start_frame) / fps / seg.speed


def build_speed_map(
    chunks: list[Chunk],
    result: AnalysisResult,
    max_duration_secs: float,
) -> list[Segment]:
    segments = [
        Segment(start_frame=chunk.start_frame, end_frame=chunk.end_frame, speed=ACTIVE_SPEED if chunk.active else IDLE_SPEED)
        for chunk in chunks
    ]

    total_output = sum(segment_duration_secs(seg, result.fps) for seg in segments)

    if total_output > max_duration_secs:
        total_active_secs = sum(
            segment_duration_secs(seg, result.fps) for seg in segments if seg.speed > 0.0
        )
        required_speed = total_active_secs / max_duration_secs
        scaled_speed = min(required_speed, MAX_SPEED)
        for seg in segments:
            if seg.speed > 0.0:
                seg.speed = scaled_speed

    return segments
