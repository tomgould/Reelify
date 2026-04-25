"""LLM-vision enrichment for Reelify segments."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from reelify.analyser import AnalysisResult
from reelify.classifier import Chunk
from reelify.vision.provider import VisionProvider

# Keywords used for activity scoring.  Active keywords push score toward 1.0;
# idle keywords push score toward 0.0.
_ACTIVE_KEYWORDS = frozenset(
    {"typing", "coding", "terminal", "clicking", "scrolling", "writing", "editing",
     "running", "building", "compiling", "debugging", "browsing", "selecting",
     "dragging", "drawing", "recording"}
)
_IDLE_KEYWORDS = frozenset(
    {"idle", "blank", "empty", "static", "loading", "waiting", "paused",
     "screen saver", "screensaver", "desktop", "nothing", "still"}
)


@dataclass
class SegmentMeta:
    start_frame: int
    end_frame: int
    active: bool
    caption: str
    score: float  # 0.0–1.0 interestingness


@dataclass
class VideoMetadata:
    source_path: str
    fps: float
    duration_secs: float
    segments: list[SegmentMeta]
    keyframes: list[str]   # relative paths to keyframe images
    provider_used: str
    created_at: str        # ISO8601


@dataclass
class EnrichmentResult:
    captions: list[str]
    scores: list[float]
    metadata: VideoMetadata


def _score_caption(caption: str) -> float:
    """Return a 0–1 interestingness score based on keyword presence."""
    lower = caption.lower()
    words = set(lower.split())
    # Check multi-word idle phrases first
    has_idle_phrase = any(phrase in lower for phrase in ("screen saver", "screensaver"))
    active_hits = sum(1 for kw in _ACTIVE_KEYWORDS if kw in words)
    idle_hits = sum(1 for kw in _IDLE_KEYWORDS if kw in words) + (1 if has_idle_phrase else 0)

    if active_hits == 0 and idle_hits == 0:
        return 0.5  # neutral — no strong signal

    total = active_hits + idle_hits
    return active_hits / total


def _nearest_chunk_index(frame: int, chunks: list[Chunk]) -> int:
    """Return the index of the chunk whose midpoint is nearest to *frame*."""
    best_idx = 0
    best_dist = float("inf")
    for i, chunk in enumerate(chunks):
        mid = (chunk.start_frame + chunk.end_frame) / 2
        dist = abs(mid - frame)
        if dist < best_dist:
            best_dist = dist
            best_idx = i
    return best_idx


def _keyframes_for_chunk(chunk: Chunk, keyframe_paths: list[Path]) -> list[Path]:
    """Return keyframes whose approximate source frame falls within *chunk*."""
    matched: list[Path] = []
    for kf_path in keyframe_paths:
        matched.append(kf_path)  # without frame metadata we can't filter precisely
    return matched


def _combine_captions(captions: list[str]) -> str:
    """Combine multiple captions into a single description."""
    unique = list(dict.fromkeys(c.strip() for c in captions if c.strip()))
    return " ".join(unique) if unique else ""


def enrich(
    video_path: Path,
    keyframe_paths: list[Path],
    chunks: list[Chunk],
    result: AnalysisResult,
    provider: VisionProvider,
    scoring: str = "fast",  # "fast" | "deep"
    progress_callback: Callable[[int, int], None] | None = None,
) -> EnrichmentResult:
    """Enrich a set of video chunks with LLM-generated captions and scores.

    Args:
        video_path: Path to the source video (used for metadata only).
        keyframe_paths: Ordered list of keyframe image paths.
        chunks: Classified chunks from :func:`reelify.classifier.classify`.
        result: Analysis result from :func:`reelify.analyser.analyse`.
        provider: Initialised :class:`VisionProvider` to use.
        scoring: ``'fast'`` → one caption per keyframe;
            ``'deep'`` → up to 3 frames per chunk (first, middle, last).

    Returns:
        :class:`EnrichmentResult` with captions, scores, and full metadata.
    """
    if not keyframe_paths:
        # No keyframes — return empty enrichment
        segments_meta = [
            SegmentMeta(
                start_frame=chunk.start_frame,
                end_frame=chunk.end_frame,
                active=bool(chunk.active),
                caption="",
                score=0.5,
            )
            for chunk in chunks
        ]
        return EnrichmentResult(
            captions=[],
            scores=[],
            metadata=VideoMetadata(
                source_path=str(video_path),
                fps=result.fps,
                duration_secs=result.total_frames / result.fps if result.fps else 0.0,
                segments=segments_meta,
                keyframes=[],
                provider_used=provider.name,
                created_at=datetime.now(timezone.utc).isoformat(),
            ),
        )

    all_captions: list[str] = []
    all_scores: list[float] = []

    if scoring == "fast":
        for idx, kf_path in enumerate(keyframe_paths):
            caption = provider.describe_frame(kf_path)
            all_captions.append(caption)
            all_scores.append(_score_caption(caption))
            if progress_callback:
                progress_callback(idx + 1, len(keyframe_paths))
    else:
        # deep: assign keyframes to chunks, then describe up to 3 per chunk
        # Group keyframes by nearest chunk
        chunk_to_keyframes: dict[int, list[Path]] = {i: [] for i in range(len(chunks))}
        # Evenly spread keyframes across chunks by index if no frame metadata
        if chunks:
            step = max(1, len(keyframe_paths) // len(chunks)) if len(chunks) < len(keyframe_paths) else 1
            for kf_idx, kf_path in enumerate(keyframe_paths):
                chunk_idx = min(kf_idx // step, len(chunks) - 1)
                chunk_to_keyframes[chunk_idx].append(kf_path)

        for chunk_idx in range(len(chunks)):
            kfs = chunk_to_keyframes[chunk_idx]
            if not kfs:
                all_captions.append("")
                all_scores.append(0.5)
                continue

            # first, middle, last — deduplicated
            indices = {0, len(kfs) // 2, len(kfs) - 1}
            selected = [kfs[i] for i in sorted(indices)]

            captions = [provider.describe_frame(kf) for kf in selected]
            combined = _combine_captions(captions)
            score = _score_caption(combined) if combined else 0.5
            all_captions.append(combined)
            all_scores.append(score)

    # Build SegmentMeta — align captions to chunks
    segments_meta: list[SegmentMeta] = []
    for i, chunk in enumerate(chunks):
        caption = all_captions[i] if i < len(all_captions) else ""
        score = all_scores[i] if i < len(all_scores) else 0.5
        segments_meta.append(
            SegmentMeta(
                start_frame=chunk.start_frame,
                end_frame=chunk.end_frame,
                active=bool(chunk.active),
                caption=caption,
                score=score,
            )
        )

    duration_secs = result.total_frames / result.fps if result.fps else 0.0

    metadata = VideoMetadata(
        source_path=str(video_path),
        fps=result.fps,
        duration_secs=duration_secs,
        segments=segments_meta,
        keyframes=[str(kf) for kf in keyframe_paths],
        provider_used=provider.name,
        created_at=datetime.now(timezone.utc).isoformat(),
    )

    return EnrichmentResult(
        captions=all_captions,
        scores=all_scores,
        metadata=metadata,
    )
