from pathlib import Path
import subprocess
import tempfile

from reelify.speed_map import Segment


def _atempo_chain(speed: float) -> str:
    filters: list[str] = []
    remaining = speed
    while remaining > 2.0:
        filters.append("atempo=2.0")
        remaining /= 2.0
    filters.append(f"atempo={remaining:.4f}")
    return ",".join(filters)


def _build_segment_command(
    input_path: Path,
    output_path: Path,
    start_frame: int,
    end_frame: int,
    speed: float,
    fps: float,
) -> list[str]:
    start_time = start_frame / fps
    duration = (end_frame - start_frame) / fps

    cmd = [
        "ffmpeg",
        "-y",
        "-ss", str(start_time),
        "-i", str(input_path),
        "-t", str(duration),
    ]

    if speed != 1.0:
        cmd.extend(["-vf", f"setpts=PTS/{speed}"])
        cmd.extend(["-af", _atempo_chain(speed)])

    cmd.extend([
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-c:a", "aac",
        "-b:a", "128k",
        "-pix_fmt", "yuv420p",
        str(output_path),
    ])
    return cmd


def encode(
    input_path: Path,
    output_path: Path,
    segments: list[Segment],
    fps: float,
    width: int,
    height: int,
) -> None:
    active_segments = [seg for seg in segments if seg.speed != 0.0]
    if not active_segments:
        raise ValueError("No active segments to encode")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        segment_files: list[Path] = []

        for idx, seg in enumerate(active_segments):
            segment_file = tmp_path / f"seg_{idx:04d}.mp4"
            cmd = _build_segment_command(
                input_path,
                segment_file,
                seg.start_frame,
                seg.end_frame,
                seg.speed,
                fps,
            )
            result = subprocess.run(cmd, capture_output=True, check=False)
            if result.returncode != 0:
                raise RuntimeError(
                    f"FFmpeg segment encoding failed: {result.stderr.decode('utf-8', errors='replace')}"
                )
            segment_files.append(segment_file)

        concat_list = tmp_path / "concat_list.txt"
        with concat_list.open("w") as f:
            for seg_file in segment_files:
                f.write(f"file '{seg_file.resolve()}'\n")

        concat_cmd = [
            "ffmpeg",
            "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(concat_list),
            "-c", "copy",
            str(output_path),
        ]
        result = subprocess.run(concat_cmd, capture_output=True, check=False)
        if result.returncode != 0:
            raise RuntimeError(
                f"FFmpeg concat failed: {result.stderr.decode('utf-8', errors='replace')}"
            )
