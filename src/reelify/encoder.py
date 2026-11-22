import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, Future
from pathlib import Path
import subprocess
import tempfile

from reelify.speed_map import Segment


def _has_audio(input_path: Path) -> bool:
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-select_streams", "a:0",
         "-show_entries", "stream=codec_type", "-of", "default=nw=1:nk=1", str(input_path)],
        capture_output=True, check=False,
    )
    return result.returncode == 0 and result.stdout.strip() != b""


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
    audio: bool,
) -> list[str]:
    # Seek and limit on the input side (-ss and -t before -i), so both are
    # measured on the source's timeline before setpts changes it. As output
    # options they were measured after setpts, which cut sped-up segments from
    # the wrong part of the video. Every segment is re-encoded, so FFmpeg
    # decodes from the keyframe before the seek point and drops frames until it
    # gets there, which makes the cut frame-accurate. Both ends sit halfway
    # between two frames, so rounding can't pull in a frame from either side.
    half_frame = 0.5 / fps
    start_time = max(0.0, start_frame / fps - half_frame)
    duration = (end_frame - start_frame) / fps - half_frame

    cmd = [
        "ffmpeg",
        "-y",
        "-ss", f"{start_time:.6f}",
        "-t", f"{duration:.6f}",
        "-i", str(input_path),
    ]

    # Restart the segment's clock at zero and divide it by the speed. fps then
    # keeps the source frame rate: round=up keeps the segment's first frame,
    # then the last frame at or before each output frame's time, so N frames
    # at speed S become ceil(N / S) frames. The scale rounds odd dimensions
    # down to even, as libx264 requires.
    vf = (
        f"setpts=(PTS-STARTPTS)/{speed},"
        f"fps={fps}:round=up,"
        "scale=trunc(iw/2)*2:trunc(ih/2)*2"
    )
    cmd.extend(["-vf", vf])
    if audio and speed != 1.0:
        cmd.extend(["-af", _atempo_chain(speed)])

    cmd.extend(["-c:v", "libx264", "-preset", "fast", "-crf", "23", "-pix_fmt", "yuv420p"])

    if audio:
        cmd.extend(["-c:a", "aac", "-b:a", "128k"])
    else:
        cmd.extend(["-an"])

    cmd.append(str(output_path))
    return cmd


def encode(
    input_path: Path,
    output_path: Path,
    segments: list[Segment],
    fps: float,
    width: int,
    height: int,
    progress_callback: Callable[[int, int], None] | None = None,
) -> None:
    active_segments = [seg for seg in segments if seg.speed != 0.0]
    if not active_segments:
        raise ValueError("No active segments to encode")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    audio = _has_audio(input_path)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        segment_files: list[Path] = []

        with ThreadPoolExecutor(max_workers=os.cpu_count()) as pool:
            futures: list[tuple[Future, Path]] = []
            for idx, seg in enumerate(active_segments):
                segment_file = tmp_path / f"seg_{idx:04d}.mp4"
                cmd = _build_segment_command(
                    input_path,
                    segment_file,
                    seg.start_frame,
                    seg.end_frame,
                    seg.speed,
                    fps,
                    audio,
                )
                future = pool.submit(subprocess.run, cmd, capture_output=True, check=False)
                futures.append((future, segment_file))
                segment_files.append(segment_file)

            completed = 0
            for future, segment_file in futures:
                result = future.result()
                if result.returncode != 0:
                    raise RuntimeError(
                        f"FFmpeg segment encoding failed: {result.stderr.decode('utf-8', errors='replace')}"
                    )
                completed += 1
                if progress_callback:
                    progress_callback(completed, len(active_segments))

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
