# Roadmap

## Done — Phase 1

- Motion analysis with OpenCV frame diff
- Chunk classification (idle vs active) with margin padding
- Speed-map generation (cut / 3× / 1×)
- FFmpeg-based encode and concat pipeline
- Keyframe extraction via PySceneDetect
- Vision enrichment (local LM Studio + Gemini API)
- `analyse` subcommand with JSON output
- 54-test suite

## In progress / near-term

- Better deep-scoring prompts (currently returns OCR text sometimes)
- Whisper subtitle burn-in
- Audio-aware idle handling (preserve quiet audio instead of cutting)
  - Dead code cleanup: remove unused `_keyframes_for_chunk` from `enricher.py` (#6)
  - Parallel segment encoding in `encoder.py` (#9)
  - Extract helper in `subtitles.py` to deduplicate ffmpeg call (#10)
  - Replace `assert` with `RuntimeError` in `analyser.py` (#4)
  - Add optional `log` callable to `pipeline.py` (#13)
  - Rename single-letter variable `l` → `line` in `analyser.py` (#12)

## Bug-fix sprint (2026-04-25)

- **P0** — Fix `pipeline.py` variable shadowing (`segments` overwritten by subtitle dicts) (#1)
- **P0** — Wire up `--enrichment` CLI flag so enrichment branch is reachable (#5)
- **P1** — Eliminate TOCTOU race in `_dedup_video()` by replacing `mktemp()` with `NamedTemporaryFile` (#8)
- **P1** — Move `-ss` / `-t` after `-i` in encoder for frame-accurate seeking (#3)
- **P1** — Optimise `classifier.py` margin padding from O(n²) to O(n) (#2)

## Future ideas

- Adaptive speed tiers (1× / 3× / 8× / cut)
- Interactive threshold preview
- Web UI
- Export SRT alongside JSON
- Support for additional vision providers
