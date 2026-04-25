"""Whisper-based subtitle extraction for Reelify."""
from pathlib import Path
import subprocess
import tempfile


def extract_audio(video_path: Path, output_path: Path) -> Path:
    """Extract audio as 16kHz mono WAV using FFmpeg."""
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"FFmpeg audio extraction failed: {result.stderr.decode('utf-8', errors='replace')}"
        )
    return output_path


def transcribe(audio_path: Path, model_name: str = "base") -> list[dict]:
    """Run Whisper and return list of {"start": float, "end": float, "text": str}.
    Raises ImportError with helpful message if openai-whisper not installed."""
    try:
        import whisper
    except ImportError as exc:
        raise RuntimeError(
            "openai-whisper not installed — run: pip install 'reelify[subtitles]'"
        ) from exc

    model = whisper.load_model(model_name)
    result = model.transcribe(str(audio_path))
    return [
        {"start": seg["start"], "end": seg["end"], "text": seg["text"].strip()}
        for seg in result["segments"]
    ]


def _format_srt_time(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def segments_to_srt(segments: list[dict]) -> str:
    """Convert segment list to SRT format string."""
    lines: list[str] = []
    for i, seg in enumerate(segments, start=1):
        start = _format_srt_time(seg["start"])
        end = _format_srt_time(seg["end"])
        lines.append(f"{i}")
        lines.append(f"{start} --> {end}")
        lines.append(seg["text"])
        lines.append("")
    return "\n".join(lines)


def write_srt(segments: list[dict], output_path: Path) -> Path:
    """Write SRT file, return path."""
    output_path.write_text(segments_to_srt(segments), encoding="utf-8")
    return output_path


def _run_ffmpeg_burn(input_path: Path, srt_path: Path, output_path: Path) -> None:
    cmd = [
        "ffmpeg", "-y", "-i", str(input_path),
        "-vf", f"subtitles={str(srt_path)}",
        "-c:a", "copy",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"FFmpeg subtitle burn failed: {result.stderr.decode('utf-8', errors='replace')}"
        )


def burn_subtitles(video_path: Path, srt_path: Path, output_path: Path) -> None:
    """Burn SRT subtitles into video using FFmpeg subtitles= filter."""
    if video_path == output_path:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_output = Path(tmpdir) / "output.mp4"
            _run_ffmpeg_burn(video_path, srt_path, tmp_output)
            tmp_output.replace(output_path)
    else:
        _run_ffmpeg_burn(video_path, srt_path, output_path)
