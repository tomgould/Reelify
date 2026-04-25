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

## Future ideas

- Adaptive speed tiers (1× / 3× / 8× / cut)
- Interactive threshold preview
- Web UI
- Export SRT alongside JSON
- Support for additional vision providers
