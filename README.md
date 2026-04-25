# Reelify

Reelify turns long screen recordings into concise summary videos using motion analysis and optional LLM vision enrichment.

> **Experimental** — Reelify was built as an exploratory project to see how far a fully local pipeline could go. It works well for personal workflows, but comes with no stability guarantees.

## What it does

The pipeline runs in five stages:

1. **Analyse** — compute per-frame motion scores with OpenCV
2. **Classify** — group frames into idle/active chunks with margin padding
3. **Speed-map** — assign each chunk a target speed (cut, 8×, 3×, 1×)
4. **Encode** — assemble segments via FFmpeg concat demuxer + `setpts`
5. **Enrich (optional)** — extract keyframes and caption them with a local or API vision model

## Usage

### `reelify process` — compress a recording

```bash
reelify process input.webm
```

Produces `input_summary.mp4` with idle gaps removed and low-activity sections sped up.

### `reelify analyse` — analyse with LLM vision, output JSON

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

## Requirements

- Python 3.10+
- FFmpeg (system install)
- LM Studio (optional, for local vision)

## Install

```bash
pip install -e .
```

## Vision providers

Reelify supports two vision provider modes for the `analyse` command:

**Local (default)** — Runs against a local LM Studio instance serving a vision model such as Qwen2.5-VL-7B. LM Studio must be running on `http://localhost:1234` before you invoke the command.

**API** — Uses Google Gemini Flash. Requires setting `REELIFY_PRO=1` and providing a `GOOGLE_API_KEY`:

```bash
REELIFY_PRO=1 GOOGLE_API_KEY=your_key reelify analyse input.webm --provider api
```

## Development & tests

```bash
python3 -m pytest tests/ -q
```

## Status

🧪 **Experimental** — No stability guarantees. APIs, defaults, and behaviour may change between commits.
