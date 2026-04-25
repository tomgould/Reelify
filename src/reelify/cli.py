import json
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import typer

from reelify.analyser import analyse
from reelify.classifier import classify
from reelify.encoder import encode
from reelify.keyframes import extract_keyframes
from reelify.pipeline import ReelifyConfig
from reelify.speed_map import build_speed_map

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

    typer.echo("  Step 1: Analysing frames…")

    def _analysis_progress(sampled: int, total: int) -> None:
        if total > 0:
            pct = int((sampled / total) * 100)
            typer.echo(f"\r    {pct}%", nl=False)

    result = analyse(input_path, progress_callback=_analysis_progress)
    typer.echo()

    typer.echo("  Step 2: Classifying segments…")
    chunks = classify(result, config.idle_threshold)

    typer.echo("  Step 3: Building speed map…")
    segments = build_speed_map(chunks, result, float(config.max_duration))

    typer.echo("  Step 4: Encoding video…")

    def _encode_progress(idx: int, total: int) -> None:
        typer.echo(f"  Encoding segment {idx}/{total}…")

    encode(
        input_path,
        output,
        segments,
        result.fps,
        result.width,
        result.height,
        progress_callback=_encode_progress,
    )

    keyframe_paths: list[Path] = []
    if config.keyframes or config.enrichment:
        keyframe_dir = output.parent / f"{output.stem}_keyframes"
        keyframe_paths = extract_keyframes(input_path, keyframe_dir)

    if config.enrichment:
        from reelify.vision.provider import get_provider
        from reelify.enricher import enrich
        vision = get_provider(config.provider)
        typer.echo(f"  Step 5: Enriching with vision (provider={config.provider}, scoring={config.scoring}) …")
        enrichment_result = enrich(
            input_path,
            keyframe_paths,
            chunks,
            result,
            vision,
            config.scoring,
        )
        if config.metadata:
            meta_path = output.with_suffix(".json")
            meta_dict = asdict(enrichment_result.metadata)
            meta_path.write_text(json.dumps(meta_dict, indent=2))

    typer.echo("Done.")


@app.command()
def analyse(
    input_path: Path = typer.Argument(..., exists=True, readable=True, help="Path to the input video file."),
    provider: str = typer.Option("local", help="Vision provider: local (default) | api (requires REELIFY_PRO=1) | auto."),
    scoring: str = typer.Option("fast", help="Scoring mode: fast | deep."),
    output: Optional[Path] = typer.Option(None, help="Output JSON path. Defaults to <input_stem>_analysis.json."),
) -> None:
    """Analyse a screen recording with LLM vision enrichment and write JSON metadata."""
    from reelify.analyser import analyse as run_analyse
    from reelify.classifier import classify
    from reelify.keyframes import extract_keyframes
    from reelify.enricher import enrich
    from reelify.vision.provider import get_provider

    if output is None:
        output = input_path.parent / f"{input_path.stem}_analysis.json"

    typer.echo(f"Analysing {input_path} …")

    typer.echo("  Step 1/4: Reading video frames …")

    def _analysis_progress(sampled: int, total: int) -> None:
        if total > 0:
            pct = int((sampled / total) * 100)
            typer.echo(f"\r    {pct}%", nl=False)

    result = run_analyse(input_path, progress_callback=_analysis_progress)
    typer.echo()
    typer.echo(f"    Done — {result.total_frames} frames @ {result.fps:.1f}fps ({result.total_frames / result.fps / 60:.1f} min)")

    typer.echo("  Step 2/4: Classifying segments …")
    chunks = classify(result, idle_threshold=0.02)
    typer.echo(f"    Done — {len(chunks)} segments found")

    keyframe_dir = input_path.parent / f"{input_path.stem}_keyframes"
    typer.echo("  Step 3/4: Extracting keyframes …")
    keyframe_paths = extract_keyframes(input_path, keyframe_dir)
    typer.echo(f"    Done — {len(keyframe_paths)} keyframes")

    vision = get_provider(provider)
    typer.echo(f"  Step 4/4: Enriching with vision (provider={provider}, scoring={scoring}) …")
    enrichment_result = enrich(
        input_path,
        keyframe_paths,
        chunks,
        result,
        vision,
        scoring,
        progress_callback=lambda i, n: typer.echo(f"    Captioning keyframe {i}/{n} …"),
    )

    meta_dict = asdict(enrichment_result.metadata)
    output.write_text(json.dumps(meta_dict, indent=2))

    duration = enrichment_result.metadata.duration_secs
    seg_count = len(enrichment_result.metadata.segments)
    typer.echo(f"Duration:  {duration:.1f}s")
    typer.echo(f"Segments:  {seg_count}")

    # Top 3 captions by score
    scored = sorted(
        zip(enrichment_result.scores, enrichment_result.captions),
        key=lambda x: x[0],
        reverse=True,
    )
    typer.echo("Top captions:")
    for score, caption in scored[:3]:
        if caption:
            typer.echo(f"  [{score:.2f}] {caption}")

    typer.echo(f"Written to {output}")
    typer.echo("Done.")
