# Reelify

Reelify turns long screen recordings into concise summary videos using motion analysis and optional LLM vision enrichment.

## Requirements

- Python 3.10+
- FFmpeg (system install)
- LM Studio (optional, for local vision)

## Install

```bash
pip install -e .
```

> **Experimental**: Reelify was built as an exploratory project to see how far a fully local pipeline could go. It works well for personal workflows, but comes with no stability guarantees.

## What it does

The pipeline runs in five stages:

1. **Analyse**: compute per-frame motion scores with OpenCV
2. **Classify**: group frames into idle/active chunks with margin padding
3. **Speed-map**: drop idle chunks and, with `--max-duration`, speed up the rest evenly to fit (at most 8×)
4. **Encode**: assemble segments via FFmpeg concat demuxer + `setpts`
5. **Enrich (optional)**: extract keyframes and caption them with a local or API vision model

## Usage

### `reelify process`, compress a recording

```bash
reelify process input.webm
```

Produces `input_summary.mp4` with idle gaps removed. Nothing is sped up unless you set a target length:

```bash
reelify process input.webm --max-duration 120
```

If what's left after the cuts is longer than the target, every kept segment is sped up by the same amount, to at most 8×.

### `reelify analyse`, analyse with LLM vision, output JSON

```bash
reelify analyse input.webm --provider local --scoring fast
```

Produces `input_analysis.json` containing segment metadata, keyframe captions, and activity scores.

## Stack

| Concern | Tool |
|---|---|
| Frame diff / motion analysis | OpenCV |
| Keyframe / scene detection | PySceneDetect |
| Encode, speed-up, concat | FFmpeg |
| CLI framework | Typer |
| Local vision (default) | LM Studio + Qwen2.5-VL-7B |
| API vision (optional) | Google Gemini Flash (requires `REELIFY_PRO=1`) |
| Image preparation | Pillow |
| LM Studio HTTP client | requests |

## Vision providers

Reelify supports two vision provider modes for the `analyse` command:

**Local (default)**: runs against a local LM Studio instance serving a vision model such as Qwen2.5-VL-7B. LM Studio must be running on `http://localhost:1234` before you invoke the command.

**API**: uses Google Gemini Flash. Requires setting `REELIFY_PRO=1` and providing a `GOOGLE_API_KEY`:

```bash
REELIFY_PRO=1 GOOGLE_API_KEY=your_key reelify analyse input.webm --provider api
```

## Development & tests

```bash
python3 -m pytest tests/ -q
```

## Status

🧪 **Experimental**: No stability guarantees. APIs, defaults, and behaviour may change between commits.
