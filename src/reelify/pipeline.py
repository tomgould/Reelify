import json
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, asdict
from pathlib import Path

from reelify.analyser import analyse
from reelify.classifier import classify
from reelify.speed_map import build_speed_map
from reelify.encoder import encode
from reelify.keyframes import extract_keyframes


@dataclass
class ReelifyConfig:
    max_duration: int
    idle_threshold: float
    keyframes: bool
    subtitles: bool
    enrichment: bool = False
    metadata: bool = False
    scoring: str = "fast"
    provider: str = "auto"
    captions: bool = False


def run(
    input_path: Path,
    output_path: Path,
    config: ReelifyConfig,
    progress_callback: Callable[[int, int], None] | None = None,
) -> None:
    result = analyse(input_path, progress_callback=progress_callback)
    chunks = classify(result, config.idle_threshold)
    segments = build_speed_map(chunks, result, float(config.max_duration))
    encode(input_path, output_path, segments, result.fps, result.width, result.height)

    if config.subtitles:
        from reelify.subtitles import extract_audio, transcribe, write_srt, burn_subtitles
        print("  Transcribing audio (Whisper)…")
        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = Path(tmpdir) / "audio.wav"
            extract_audio(input_path, wav_path)
            segments = transcribe(wav_path)
            srt_path = output_path.with_suffix(".srt")
            write_srt(segments, srt_path)
            print("  Burning subtitles…")
            burn_subtitles(output_path, srt_path, output_path)

    keyframe_paths: list[Path] = []
    if config.keyframes or config.enrichment:
        keyframe_dir = output_path.parent / f"{output_path.stem}_keyframes"
        keyframe_paths = extract_keyframes(input_path, keyframe_dir)

    if config.enrichment:
        from reelify.vision.provider import get_provider
        from reelify.enricher import enrich
        vision = get_provider(config.provider)
        enrichment_result = enrich(
            input_path, keyframe_paths, chunks, result, vision, config.scoring
        )
        if config.metadata:
            meta_path = output_path.with_suffix(".json")
            meta_dict = asdict(enrichment_result.metadata)
            meta_path.write_text(json.dumps(meta_dict, indent=2))
