from pathlib import Path
from typing import Optional

import typer

from reelify.pipeline import ReelifyConfig, run

app = typer.Typer(help="Turn long screen recordings into concise summary videos.")


@app.command()
def process(
    input_path: Path = typer.Argument(..., exists=True, readable=True, help="Path to the input video file."),
    output: Optional[Path] = typer.Option(None, help="Output video path. Defaults to <input_stem>_summary.mp4."),
    max_duration: int = typer.Option(120, help="Target output duration in seconds."),
    idle_threshold: float = typer.Option(0.02, help="Motion threshold (0-1) for classifying idle frames."),
    keyframes: bool = typer.Option(False, help="Extract representative keyframes from scene changes."),
    subtitles: bool = typer.Option(False, help="Burn in subtitles (Phase 2)."),
) -> None:
    if output is None:
        output = input_path.parent / f"{input_path.stem}_summary.mp4"

    config = ReelifyConfig(
        max_duration=max_duration,
        idle_threshold=idle_threshold,
        keyframes=keyframes,
        subtitles=subtitles,
    )

    typer.echo(f"Input:   {input_path}")
    typer.echo(f"Output:  {output}")
    typer.echo(f"Config:  max_duration={config.max_duration}, idle_threshold={config.idle_threshold}, keyframes={config.keyframes}, subtitles={config.subtitles}")

    run(input_path, output, config)
