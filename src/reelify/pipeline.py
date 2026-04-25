from dataclasses import dataclass
from pathlib import Path


@dataclass
class ReelifyConfig:
    max_duration: int
    idle_threshold: float
    keyframes: bool
    subtitles: bool


def run(input_path: Path, output_path: Path, config: ReelifyConfig) -> None:
    raise NotImplementedError("Pipeline not yet implemented")
