from pathlib import Path
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from reelify.cli import app

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
