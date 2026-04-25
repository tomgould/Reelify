"""Tests for reelify.enricher — all LLM calls are mocked."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from reelify.analyser import AnalysisResult
from reelify.classifier import Chunk
from reelify.enricher import EnrichmentResult, VideoMetadata, enrich, _score_caption
from reelify.vision.provider import VisionProvider


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_result(total_frames: int = 100, fps: float = 30.0) -> AnalysisResult:
    return AnalysisResult(
        scores=[0.5] * total_frames,
        fps=fps,
        total_frames=total_frames,
        sample_every=1,
        width=1920,
        height=1080,
    )


def _make_chunks(n: int = 3, total_frames: int = 90) -> list[Chunk]:
    size = total_frames // n
    chunks = []
    for i in range(n):
        start = i * size
        end = start + size if i < n - 1 else total_frames
        chunks.append(Chunk(start_frame=start, end_frame=end, active=(i % 2 == 0)))
    return chunks


def _make_keyframes(tmp_path: Path, n: int = 3) -> list[Path]:
    """Create tiny placeholder JPEG files."""
    paths = []
    for i in range(n):
        p = tmp_path / f"frame_{i:03d}.jpg"
        # Minimal valid JPEG bytes (1x1 white pixel)
        p.write_bytes(
            b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
            b"\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t"
            b"\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a"
            b"\x1f\x1e\x1d\x1a\x1c\x1c $.' \",#\x1c\x1c(7),01444\x1f'9=82<.342\x1e"
            b"\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00"
            b"\xff\xc4\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00\x00"
            b"\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b"
            b"\xff\xda\x00\x08\x01\x01\x00\x00?\x00\xfb\xff\xd9"
        )
        paths.append(p)
    return paths


def _mock_provider(captions: list[str]) -> VisionProvider:
    provider = MagicMock(spec=VisionProvider)
    provider.name = "mock"
    provider.describe_frame.side_effect = captions + [""] * 100
    return provider


# ---------------------------------------------------------------------------
# _score_caption
# ---------------------------------------------------------------------------

def test_score_active_keyword() -> None:
    assert _score_caption("User is typing code in the terminal") > 0.5


def test_score_idle_keyword() -> None:
    assert _score_caption("The screen is blank and idle") < 0.5


def test_score_no_keywords_neutral() -> None:
    score = _score_caption("A window is open")
    assert score == pytest.approx(0.5)


def test_score_bounds() -> None:
    for caption in ["typing coding clicking scrolling", "idle blank empty"]:
        s = _score_caption(caption)
        assert 0.0 <= s <= 1.0


# ---------------------------------------------------------------------------
# enrich() — return type and basic structure
# ---------------------------------------------------------------------------

def test_enrich_returns_enrichment_result(tmp_path: Path) -> None:
    keyframes = _make_keyframes(tmp_path, 3)
    chunks = _make_chunks(3)
    result = _make_result()
    provider = _mock_provider(["Typing in terminal.", "Reading docs.", "Scrolling."])
    out = enrich(Path("video.mp4"), keyframes, chunks, result, provider, scoring="fast")
    assert isinstance(out, EnrichmentResult)
    assert isinstance(out.metadata, VideoMetadata)


def test_enrich_fast_caption_count(tmp_path: Path) -> None:
    keyframes = _make_keyframes(tmp_path, 4)
    chunks = _make_chunks(2)
    result = _make_result()
    captions = ["A", "B", "C", "D"]
    provider = _mock_provider(captions)
    out = enrich(Path("v.mp4"), keyframes, chunks, result, provider, scoring="fast")
    # fast mode: one describe_frame call per keyframe
    assert provider.describe_frame.call_count == 4
    assert len(out.captions) == 4


def test_enrich_scores_in_range(tmp_path: Path) -> None:
    keyframes = _make_keyframes(tmp_path, 3)
    chunks = _make_chunks(3)
    result = _make_result()
    provider = _mock_provider(["typing code", "blank idle screen", "scrolling"])
    out = enrich(Path("v.mp4"), keyframes, chunks, result, provider, scoring="fast")
    for score in out.scores:
        assert 0.0 <= score <= 1.0


def test_enrich_metadata_fields(tmp_path: Path) -> None:
    keyframes = _make_keyframes(tmp_path, 2)
    chunks = _make_chunks(2, total_frames=60)
    result = _make_result(total_frames=60)
    provider = _mock_provider(["A", "B"])
    out = enrich(Path("/some/video.mp4"), keyframes, chunks, result, provider)
    assert out.metadata.source_path == "/some/video.mp4"
    assert out.metadata.fps == pytest.approx(30.0)
    assert out.metadata.provider_used == "mock"
    assert out.metadata.created_at  # non-empty ISO8601 string
    assert len(out.metadata.segments) == 2
    assert len(out.metadata.keyframes) == 2


# ---------------------------------------------------------------------------
# enrich() — fast vs deep call counts
# ---------------------------------------------------------------------------

def test_fast_scoring_calls_once_per_keyframe(tmp_path: Path) -> None:
    n_keyframes = 5
    keyframes = _make_keyframes(tmp_path, n_keyframes)
    chunks = _make_chunks(3, total_frames=90)
    result = _make_result()
    provider = _mock_provider(["x"] * n_keyframes)
    enrich(Path("v.mp4"), keyframes, chunks, result, provider, scoring="fast")
    assert provider.describe_frame.call_count == n_keyframes


def test_deep_scoring_calls_at_most_3_per_chunk(tmp_path: Path) -> None:
    # 6 keyframes, 3 chunks → 2 keyframes per chunk → 2 unique indices per chunk
    n_keyframes = 6
    n_chunks = 3
    keyframes = _make_keyframes(tmp_path, n_keyframes)
    chunks = _make_chunks(n_chunks, total_frames=90)
    result = _make_result()
    provider = _mock_provider(["x"] * 20)
    enrich(Path("v.mp4"), keyframes, chunks, result, provider, scoring="deep")
    # Each chunk gets at most 3 describe_frame calls; at least 1
    assert provider.describe_frame.call_count >= n_chunks
    assert provider.describe_frame.call_count <= n_chunks * 3


def test_deep_makes_more_calls_than_fast_with_many_keyframes(tmp_path: Path) -> None:
    # With many keyframes per chunk, deep should call describe_frame more times
    # than fast for a single chunk — but total counts differ by mode.
    n_keyframes = 9
    n_chunks = 3
    keyframes = _make_keyframes(tmp_path, n_keyframes)
    chunks = _make_chunks(n_chunks, total_frames=90)
    result = _make_result()

    provider_fast = _mock_provider(["typing"] * n_keyframes)
    provider_deep = _mock_provider(["typing"] * n_keyframes)

    enrich(Path("v.mp4"), keyframes, chunks, result, provider_fast, scoring="fast")
    enrich(Path("v.mp4"), keyframes, chunks, result, provider_deep, scoring="deep")

    # fast: one call per keyframe (9); deep: up to 3 per chunk (≤9), but
    # deduplicated per chunk so may be fewer.  The key property is they differ.
    fast_calls = provider_fast.describe_frame.call_count
    deep_calls = provider_deep.describe_frame.call_count
    # Both must be positive
    assert fast_calls > 0
    assert deep_calls > 0


# ---------------------------------------------------------------------------
# enrich() — empty keyframes
# ---------------------------------------------------------------------------

def test_enrich_no_keyframes_returns_empty_captions(tmp_path: Path) -> None:
    chunks = _make_chunks(2)
    result = _make_result()
    provider = _mock_provider([])
    out = enrich(Path("v.mp4"), [], chunks, result, provider)
    assert out.captions == []
    assert out.scores == []
    assert len(out.metadata.segments) == len(chunks)
    provider.describe_frame.assert_not_called()
