import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class AnalysisResult:
    scores: list[float]
    fps: float
    total_frames: int
    sample_every: int
    width: int
    height: int


def _probe_duration(video_path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v", "quiet",
            "-show_entries", "format=duration",
            "-of", "default=nw=1:nk=1",
            str(video_path),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise ValueError(f"ffprobe failed for {video_path}")
    duration = float(result.stdout.strip())
    if duration <= 0:
        raise ValueError(f"ffprobe returned invalid duration for {video_path}")
    return duration


def analyse(
    video_path: Path,
    sample_every: int = 4,
    progress_callback: Callable[[int, int], None] | None = None,
) -> AnalysisResult:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    estimated_total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    has_estimated_total = estimated_total > 0
    estimated_samples = int(estimated_total / sample_every) if has_estimated_total else 0

    scores: list[float] = []
    prev_gray: np.ndarray | None = None
    frame_index = 0
    actual_frame_count = 0
    sampled_so_far = 0

    while True:
        if frame_index % sample_every != 0:
            ret = cap.grab()
            if not ret:
                break
            frame_index += 1
            actual_frame_count += 1
            continue

        ret, frame = cap.read()
        if not ret:
            break

        actual_frame_count += 1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is None:
            scores.append(0.0)
        else:
            mad = np.mean(np.abs(gray.astype(np.float32) - prev_gray.astype(np.float32)))
            scores.append(mad / 255.0)
        prev_gray = gray
        frame_index += 1
        sampled_so_far += 1

        if progress_callback and sampled_so_far % 50 == 0 and has_estimated_total:
            progress_callback(sampled_so_far, estimated_samples)

    cap.release()

    try:
        duration = _probe_duration(video_path)
        fps = actual_frame_count / duration
        total_frames = actual_frame_count
    except Exception:
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    return AnalysisResult(
        scores=scores,
        fps=fps,
        total_frames=total_frames,
        sample_every=sample_every,
        width=width,
        height=height,
    )
