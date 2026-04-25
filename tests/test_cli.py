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
