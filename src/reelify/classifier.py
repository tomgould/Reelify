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

    # Build padded via a single dilation: any frame within margin_frames of an active frame is padded.
    # O(n) two-pass: forward sweep marks runs starting at each active frame,
    # backward sweep marks runs ending at each active frame.
    padded = [False] * result.total_frames
    # Forward pass
    streak = 0
    for f in range(result.total_frames):
        if labels[f]:
            streak = margin_frames + 1
        if streak > 0:
            padded[f] = True
            streak -= 1
    # Backward pass
    streak = 0
    for f in range(result.total_frames - 1, -1, -1):
        if labels[f]:
            streak = margin_frames + 1
        if streak > 0:
            padded[f] = True
            streak -= 1

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
