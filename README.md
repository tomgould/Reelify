# Reelify

Give Reelify an hour of screen recording and it hands back the minutes where the screen actually changes. A target length is optional, and keyframe captions can come from a vision model on your own machine.

## Requirements

- Python 3.10 or later
- FFmpeg and ffprobe on the `PATH`
- LM Studio serving a vision model, only for captions

## Install

```bash
pip install -e .
# with the optional subtitles extra
pip install -e '.[subtitles]'
```

## Shorten a recording

```bash
reelify process input.webm
```

This writes `input_summary.mp4` with the idle gaps removed. Nothing is sped up unless you give a target length in seconds:

```bash
reelify process input.webm --max-duration 120
```

If what's left after the cuts is longer than the target, every kept segment is sped up by the same amount, to at most 8x.

Other options for `process`:

- `--no-dedup` keeps repeated frames; by default FFmpeg's `mpdecimate` drops them first (`--dedup-similarity` sets how alike two frames must be).
- `--frametime N` holds each frame left after deduplication for N seconds.
- `--preset cli` suits terminal recordings: deduplication at 0.95 similarity, three seconds per frame and an idle threshold of 0.005.
- `--subtitles` transcribes the speech and burns the subtitles into the output (needs the `subtitles` extra).
- `--enrichment` captions the keyframes with a vision model and writes them beside the output video as `<output_stem>.json` (for example `summary.json`), in the same format as `analyse`. `--provider` and `--scoring` work as they do for `analyse`.

## Describe a recording as JSON

```bash
reelify analyse input.webm --provider local --scoring fast
```

This writes `input_analysis.json` with the segments, their activity scores and a caption for each keyframe. `--scoring deep` captions up to three frames per segment instead of one per keyframe.

## How it works

Every fourth frame is decoded as a 320-pixel greyscale thumbnail and compared with the previous one by mean absolute difference. Samples under the idle threshold (0.02 by default) mark idle frames; everything within 0.3 seconds of motion counts as active. Idle runs are dropped, the active runs share one speed when a target length needs it, and each run is re-encoded with `libx264` in parallel before the pieces are concatenated. Keyframes come from PySceneDetect's content detector.

[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) has the detail.

## Captions

By default captions come from LM Studio on `http://localhost:1234`, serving a vision model such as Qwen2.5-VL-7B, so the footage never leaves the machine. Start LM Studio before running `analyse`.

`--provider api` sends keyframes to Google's Gemini API instead. It needs `REELIFY_PRO=1` and a `GOOGLE_API_KEY`:

```bash
REELIFY_PRO=1 GOOGLE_API_KEY=your_key reelify analyse input.webm --provider api
```

`--provider auto` tries LM Studio first and falls back to the API when `REELIFY_PRO=1` is set.

## Tests

```bash
pip install -e '.[dev]'
python3 -m pytest tests/ -q
```

## Licence

MIT. See [LICENSE](LICENSE).
