# Reelify

Turns a long screen recording into a ~2 minute summary video.

## What it does

1. Removes idle gaps (no mouse/keyboard/visual change)
2. Speeds up low-activity sections (smooth timelapse, not hard cuts)
3. Extracts the most visually distinct keyframes
4. Optionally burns in subtitles (window titles, visible text via Whisper)
5. Outputs a compressed summary MP4 + optional JSON activity map

## Stack

| Concern | Tool |
|---|---|
| Idle gap removal | OpenCV frame diff + input event hooks |
| Scene detection / keyframes | PySceneDetect |
| Timelapse / speedup / concat | FFmpeg (`setpts`, `concat` demuxer) |
| Subtitles | FFmpeg `drawtext` / SRT burn-in |
| Audio transcription (optional) | Whisper (local) |
| CLI | Typer |

## Requirements

- Python 3.10+
- FFmpeg (system install)
- Linux (Ubuntu 22.04+)

## Usage

```bash
pip install -e .
reelify input.mp4 --output summary.mp4
```

## Development

```bash
python3 -m pytest tests/ -q --tb=short
```
