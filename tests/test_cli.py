import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from typer.testing import CliRunner

from reelify.cli import _dedup_video, app

runner = CliRunner()


def test_help_exits_zero() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0


def test_nonexistent_input_exits_nonzero() -> None:
    result = runner.invoke(app, ["nonexistent.mp4"])
    assert result.exit_code != 0
    assert "does not exist" in result.output or "Error" in result.output


def test_process_enrichment_flag(tmp_path: Path) -> None:
    dummy_video = tmp_path / "dummy.mp4"
    dummy_video.write_text("")

    mock_result = MagicMock()
    mock_result.fps = 30.0
    mock_result.width = 1920
    mock_result.height = 1080
    mock_result.total_frames = 900
    mock_result.sample_every = 1
    mock_result.scores = [0.5] * 900

    mock_enrich = MagicMock()
    mock_enrich.return_value = MagicMock(
        metadata=MagicMock(
            source_path="dummy",
            fps=30.0,
            duration_secs=30.0,
            segments=[],
            keyframes=[],
            provider_used="mock",
            created_at="2024-01-01T00:00:00Z",
        ),
        captions=[],
        scores=[],
    )

    mock_provider = MagicMock()
    mock_provider.name = "mock"

    with (
        patch("reelify.cli._analyse", return_value=mock_result),
        patch("reelify.cli.classify", return_value=[]),
        patch("reelify.cli.build_speed_map", return_value=[]),
        patch("reelify.cli.encode"),
        patch("reelify.cli.extract_keyframes", return_value=[Path("kf1.jpg")]),
        patch("reelify.vision.provider.get_provider", return_value=mock_provider),
        patch("reelify.enricher.enrich", mock_enrich),
    ):
        result = runner.invoke(app, ["process", str(dummy_video), "--enrichment", "--no-dedup"])

    assert result.exit_code == 0
    mock_enrich.assert_called_once()


# --- End-to-end runs on a synthetic recording --------------------------------
#
# A white page that's still for 1.5 seconds, has a black box sliding across it
# at a steady speed for 1.5 seconds, then is still again for 1.5 seconds.

_FPS = 30


def _make_recording(path: Path) -> None:
    x = "100*min(max((t-1.5)/1.5\\,0)\\,1)"
    graph = (
        f"color=white:s=160x120:r={_FPS}:d=4.5[bg];"
        f"color=black:s=32x32:r={_FPS}[box];"
        f"[bg][box]overlay=x='{x}':y=44:shortest=1,format=yuv420p"
    )
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-filter_complex", graph, "-c:v", "libx264", str(path)],
        capture_output=True, check=True,
    )


def _frame_count(path: Path) -> int:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
         "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout
    return int(out)


def _duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout
    return float(out)


def _box_positions(path: Path) -> list[int]:
    """Return the box's left edge in each frame of a video made from the recording."""
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "gray", "-"],
        capture_output=True, check=True,
    ).stdout
    frames = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 120, 160)
    return [int(np.argmax(frame[50:70].mean(axis=0) < 128)) for frame in frames]


@pytest.fixture
def recording(tmp_path: Path) -> Path:
    path = tmp_path / "recording.mp4"
    _make_recording(path)
    return path


def test_dedup_keeps_steady_motion(recording: Path) -> None:
    deduped = _dedup_video(recording, similarity=0.90)
    try:
        # The first frame, then one frame for each step of the box. The still
        # frames before and after are near-duplicates.
        assert 40 <= _frame_count(deduped) <= 50
    finally:
        deduped.unlink()


def _process(recording: Path, output: Path, *args: str) -> None:
    result = runner.invoke(
        app, ["process", str(recording), "--output", str(output), "--no-keyframes", *args]
    )
    assert result.exit_code == 0, result.output + repr(result.exception)


def test_process_cuts_still_parts(recording: Path, tmp_path: Path) -> None:
    output = tmp_path / "summary.mp4"
    _process(recording, output, "--no-dedup")
    # 1.5 seconds of motion plus up to 0.3 seconds of margin each side
    assert 1.5 <= _duration(output) <= 2.2
    positions = _box_positions(output)
    assert positions[0] == 0
    assert positions[-1] == 100
    assert positions == sorted(positions)


def test_process_with_dedup_by_default(recording: Path, tmp_path: Path) -> None:
    output = tmp_path / "summary.mp4"
    _process(recording, output)
    assert 1.0 <= _duration(output) <= 2.2


@pytest.mark.parametrize("dedup", ["--dedup", "--no-dedup"])
def test_process_fits_max_duration(recording: Path, tmp_path: Path, dedup: str) -> None:
    output = tmp_path / "summary.mp4"
    _process(recording, output, dedup, "--max-duration", "1")
    assert _duration(output) == pytest.approx(1.0, abs=0.1)
    # Sped up, the output still runs from the box's start to (near) its end.
    positions = _box_positions(output)
    assert positions[0] == 0
    assert positions[-1] >= 90
    assert positions == sorted(positions)
