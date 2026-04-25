import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class AnalysisResult:
    scores: list[float]
    fps: float
    total_frames: int
    sample_every: int
    width: int
    height: int


def _probe(video_path: Path) -> tuple[float, float, int, int]:
    """Return (duration, fps, width, height) via ffprobe."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "quiet",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,r_frame_rate:format=duration",
            "-of", "csv=p=0",
            str(video_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    lines = [l.strip() for l in result.stdout.strip().splitlines() if l.strip()]
    width = height = fps_val = duration = 0.0
    for line in lines:
        parts = line.split(",")
        if len(parts) == 3:
            # stream line: width,height,r_frame_rate
            width, height = int(parts[0]), int(parts[1])
            num, den = parts[2].split("/")
            fps_val = float(num) / float(den)
        elif len(parts) == 1:
            try:
                duration = float(parts[0])
            except ValueError:
                pass
    return duration, fps_val, int(width), int(height)


_THUMB_WIDTH = 320


def analyse(
    video_path: Path,
    sample_every: int = 4,
    progress_callback: Callable[[int, int], None] | None = None,
) -> AnalysisResult:
    duration, native_fps, width, height = _probe(video_path)
    if native_fps <= 0:
        native_fps = 30.0

    thumb_height = int(height * _THUMB_WIDTH / width) if width > 0 else _THUMB_WIDTH
    sample_fps = native_fps / sample_every
    estimated_samples = max(1, int(duration * sample_fps))

    # Use ffmpeg to decode only the sampled frames at thumbnail resolution.
    # This avoids reading every intermediate frame entirely.
    cmd = [
        "ffmpeg", "-i", str(video_path),
        "-vf", f"fps={sample_fps},scale={_THUMB_WIDTH}:{thumb_height}",
        "-f", "rawvideo", "-pix_fmt", "gray",
        "-",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    assert proc.stdout is not None

    frame_bytes = _THUMB_WIDTH * thumb_height
    scores: list[float] = []
    prev_gray: np.ndarray | None = None
    sampled = 0

    while True:
        raw = proc.stdout.read(frame_bytes)
        if len(raw) < frame_bytes:
            break
        gray = np.frombuffer(raw, dtype=np.uint8).reshape((thumb_height, _THUMB_WIDTH))
        if prev_gray is None:
            scores.append(0.0)
        else:
            mad = np.mean(np.abs(gray.astype(np.float32) - prev_gray.astype(np.float32)))
            scores.append(mad / 255.0)
        prev_gray = gray
        sampled += 1
        if progress_callback and sampled % 10 == 0:
            progress_callback(sampled, estimated_samples)

    proc.stdout.close()
    proc.wait()

    total_frames = int(sampled * sample_every)
    fps = native_fps

    return AnalysisResult(
        scores=scores,
        fps=fps,
        total_frames=total_frames,
        sample_every=sample_every,
        width=width,
        height=height,
    )
