import json
import subprocess
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import typer

from reelify.analyser import analyse as _analyse
from reelify.classifier import classify
from reelify.encoder import encode
from reelify.keyframes import extract_keyframes
from reelify.pipeline import ReelifyConfig
from reelify.speed_map import build_speed_map

app = typer.Typer(help="Turn long screen recordings into concise summary videos.")


_PRESETS: dict[str, dict] = {
    "cli": dict(
        dedup=True,
        dedup_similarity=0.95,
        frametime=3.0,
        idle_threshold=0.005,
        keyframes=True,
        max_duration=0,
    ),
}


def _dedup_video(input_path: Path, similarity: float = 0.98, frametime: float = 0.0) -> Path:
    """Return a temp file with near-duplicate frames removed via ffmpeg scene filter.

    frametime > 0 holds each unique frame for that many seconds in the output.
    """
    threshold = round(1.0 - similarity, 4)
    ntf = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    ntf.close()
    tmp = Path(ntf.name)

    if frametime > 0:
        # Space selected frames frametime seconds apart; output at matching fps
        out_fps = round(1.0 / frametime, 6)
        vf = (
            f"select='gt(scene,{threshold})',"
            f"setpts=N*{frametime}/TB,"
            f"scale=trunc(iw/2)*2:trunc(ih/2)*2"
        )
        extra = ["-r", str(out_fps)]
    else:
        vf = (
            f"select='gt(scene,{threshold})',"
            f"setpts=N/FRAME_RATE/TB,"
            f"scale=trunc(iw/2)*2:trunc(ih/2)*2"
        )
        extra = ["-vsync", "vfr"]

    cmd = [
        "ffmpeg", "-y", "-i", str(input_path),
        "-vf", vf,
        *extra,
        "-c:v", "libx264", "-preset", "ultrafast",
        "-c:a", "copy",
        str(tmp),
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg dedup failed:\n{result.stderr.decode()}")
    return tmp


@app.command()
def process(
    input_path: Path = typer.Argument(..., exists=True, readable=True, help="Path to the input video file."),
    output: Optional[Path] = typer.Option(None, help="Output video path. Defaults to <input_stem>_summary.mp4."),
    max_duration: int = typer.Option(0, help="Target output duration in seconds. 0 = no limit."),
    idle_threshold: float = typer.Option(0.02, help="Motion threshold (0-1) for classifying idle frames."),
    keyframes: bool = typer.Option(True, help="Extract representative keyframes from scene changes."),
    subtitles: bool = typer.Option(False, help="Burn in subtitles (Phase 2)."),
    enrichment: bool = typer.Option(False, "--enrichment/--no-enrichment", help="Enable LLM vision enrichment."),
    dedup: bool = typer.Option(True, "--dedup/--no-dedup", help="Drop near-duplicate frames before analysis (default: on)."),
    dedup_similarity: float = typer.Option(0.90, "--dedup-similarity", help="Similarity threshold for --dedup (0–1, default 0.90)."),
    frametime: float = typer.Option(0.0, "--frametime", help="Hold each unique deduped frame for N seconds. 0 = keep original timing."),
    preset: Optional[str] = typer.Option(None, "--preset", help=f"Apply a named preset. Available: {', '.join(_PRESETS)}."),
) -> None:
    if preset is not None:
        if preset not in _PRESETS:
            typer.echo(f"Unknown preset '{preset}'. Available: {', '.join(_PRESETS)}", err=True)
            raise typer.Exit(1)
        p = _PRESETS[preset]
        dedup = p.get("dedup", dedup)
        dedup_similarity = p.get("dedup_similarity", dedup_similarity)
        frametime = p.get("frametime", frametime)
        idle_threshold = p.get("idle_threshold", idle_threshold)
        keyframes = p.get("keyframes", keyframes)
        if max_duration == 0:
            max_duration = p.get("max_duration", max_duration)

    if output is None:
        output = input_path.parent / f"{input_path.stem}_summary.mp4"

    effective_max = max_duration if max_duration > 0 else 86400

    config = ReelifyConfig(
        max_duration=effective_max,
        idle_threshold=idle_threshold,
        keyframes=keyframes,
        subtitles=subtitles,
        enrichment=enrichment,
    )

    typer.echo(f"Input:   {input_path}")
    typer.echo(f"Output:  {output}")
    max_label = f"{max_duration}s" if max_duration > 0 else "unlimited"
    typer.echo(f"Config:  max_duration={max_label}, idle_threshold={config.idle_threshold}, keyframes={config.keyframes}, subtitles={config.subtitles}, enrichment={config.enrichment}")

    analyse_path = input_path
    dedup_tmp: Optional[Path] = None
    if dedup:
        ft_label = f", frametime={frametime}s" if frametime > 0 else ""
        typer.echo(f"  Step 0: Deduplicating frames (similarity≥{dedup_similarity:.0%}{ft_label})…")
        dedup_tmp = _dedup_video(input_path, dedup_similarity, frametime)
        analyse_path = dedup_tmp
        typer.echo(f"    → {dedup_tmp}")

    typer.echo("  Step 1: Analysing frames…")

    def _analysis_progress(sampled: int, total: int) -> None:
        if total > 0:
            pct = int((sampled / total) * 100)
            typer.echo(f"\r    {pct}%", nl=False)

    result = _analyse(analyse_path, progress_callback=_analysis_progress)
    typer.echo()

    typer.echo("  Step 2: Classifying segments…")
    chunks = classify(result, config.idle_threshold)

    typer.echo("  Step 3: Building speed map…")
    segments = build_speed_map(chunks, result, float(config.max_duration))

    typer.echo("  Step 4: Encoding video…")

    def _encode_progress(idx: int, total: int) -> None:
        typer.echo(f"  Encoding segment {idx}/{total}…")

    encode(
        analyse_path,
        output,
        segments,
        result.fps,
        result.width,
        result.height,
        progress_callback=_encode_progress,
    )

    if config.subtitles:
        from reelify.subtitles import extract_audio, transcribe, write_srt, burn_subtitles
        typer.echo("  Step 5: Transcribing audio (Whisper)…")
        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = Path(tmpdir) / "audio.wav"
            extract_audio(analyse_path, wav_path)
            sub_segments = transcribe(wav_path)
            srt_path = output.with_suffix(".srt")
            write_srt(sub_segments, srt_path)
            typer.echo("  Step 6: Burning subtitles…")
            burn_subtitles(output, srt_path, output)

    keyframe_paths: list[Path] = []
    if config.keyframes or config.enrichment:
        keyframe_dir = output.parent / f"{output.stem}_keyframes"
        keyframe_paths = extract_keyframes(analyse_path, keyframe_dir)

    if config.enrichment:
        from reelify.vision.provider import get_provider
        from reelify.enricher import enrich
        vision = get_provider(config.provider)
        typer.echo(f"  Step 7: Enriching with vision (provider={config.provider}, scoring={config.scoring}) …")
        enrichment_result = enrich(
            analyse_path,
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

    if dedup_tmp and dedup_tmp.exists():
        dedup_tmp.unlink()

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
