from dataclasses import dataclass
from reelify.analyser import AnalysisResult


@dataclass
class Chunk:
    start_frame: int
    end_frame: int
    active: bool


def classify(result: AnalysisResult, idle_threshold: float, margin_secs: float = 0.3) -> list[Chunk]:
    if result.total_frames <= 0:
        return []

    margin_frames = round(margin_secs * result.fps)

    labels = [False] * result.total_frames

    for sample_idx, score in enumerate(result.scores):
        orig_start = sample_idx * result.sample_every
        orig_end = (sample_idx + 1) * result.sample_every
        active = score >= idle_threshold
        for f in range(orig_start, min(orig_end, result.total_frames)):
            labels[f] = active

    padded = labels[:]
    for f in range(result.total_frames):
        if labels[f]:
            start = max(0, f - margin_frames)
            end = min(result.total_frames, f + margin_frames + 1)
            for g in range(start, end):
                padded[g] = True

    chunks: list[Chunk] = []
    current_start = 0
    current_label = padded[0]

    for f in range(1, result.total_frames):
        if padded[f] != current_label:
            chunks.append(Chunk(start_frame=current_start, end_frame=f, active=current_label))
            current_start = f
            current_label = padded[f]

    chunks.append(Chunk(start_frame=current_start, end_frame=result.total_frames, active=current_label))

    return chunks
