from dataclasses import dataclass
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


def run(input_path: Path, output_path: Path, config: ReelifyConfig) -> None:
    result = analyse(input_path)
    chunks = classify(result, config.idle_threshold)
    segments = build_speed_map(chunks, result, float(config.max_duration))
    encode(input_path, output_path, segments, result.fps, result.width, result.height)
    if config.keyframes:
        keyframe_dir = output_path.parent / f"{output_path.stem}_keyframes"
        extract_keyframes(input_path, keyframe_dir)
