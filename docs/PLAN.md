Phase 1 MVP Plan
================

Architecture Overview
---------------------
The Reelify pipeline processes a long screen recording through five discrete stages to produce a compressed summary video.

    analyse -> classify -> speed-map -> encode -> (optional) keyframes

- **Analyse**: compute per-frame motion scores with OpenCV
- **Classify**: group frames into idle/active chunks with margin padding
- **Speed-map**: assign each chunk a target speed (cut, 8x, 3x, 1x)
- **Encode**: assemble segments via FFmpeg concat demuxer + setpts
- **Keyframes (optional)**: extract representative scene-change JPEGs

Stage 1: Activity Analysis
--------------------------
OpenCV reads the input video frame by frame. Adjacent frames are converted to grayscale and compared with a mean absolute difference (MAD). The MAD is normalised to 0–1 by dividing by 255. A frame is marked **idle** when MAD < `idle_threshold` (default 0.02), otherwise **active**. This mirrors auto-editor's `--edit motion:threshold=0.02` heuristic.

Stage 2: Chunk Classification
-----------------------------
Consecutive frames with the same activity label are merged into chunks. A margin of ~0.3 s is applied around every active chunk so that tiny idle gaps inside an active region do not produce hard cuts. After padding, adjacent chunks of the same type are merged again.

Resulting chunk types:
- **idle** — no meaningful motion for the entire chunk
- **active** — at least one frame exceeded the threshold (after padding)

Stage 3: Speed Map
------------------
Each chunk is assigned a playback speed:

| Chunk type | Speed | Notes |
|------------|-------|-------|
| idle       | cut   | Dropped entirely; if audio is present in a future version, use 8x instead |
| low-activity | 3x  | Not yet distinguished in Phase 1; active chunks below a second speed threshold could map here |
| active     | 1x    | Preserved at normal speed, ideally copied without re-encode |

The speed map is a list of `(start_time, end_time, speed)` tuples. The total duration after applying speeds must be <= `max_duration` (default 120 s). If the map exceeds the limit, speeds of the longest active chunks can be increased iteratively.

Stage 4: FFmpeg Encode
----------------------
FFmpeg encodes the final video using the concat demuxer. Each segment is described by a concat file entry that optionally applies the `setpts` filter for speed changes:

    file 'segment_0.mp4'
    inpoint 12.500
    outpoint 45.200
    ...

Active 1x segments are copied (`-c copy`) when possible to avoid generation loss. Speed-adjusted segments are re-encoded with a fast preset (`libx264` / `libx265`). The concat demuxer guarantees frame-accurate stitching.

Stage 5 (optional): Keyframe Extraction
---------------------------------------
When `--keyframes` is passed, PySceneDetect runs `ContentDetector` (and optionally `AdaptiveDetector`) over the original video to locate scene boundaries. One representative frame from each scene is saved as a JPEG in an adjacent folder named `{input_stem}_keyframes/`.

CLI Interface
-------------
Built with Typer:

    reelify input.mp4 [--output summary.mp4] [--max-duration 120] \
                      [--idle-threshold 0.02] [--keyframes] [--subtitles]

Parameters:
- `input_path` — source video (required)
- `--output` — destination path; defaults to `{input_stem}_summary.mp4`
- `--max-duration` — target output length in seconds
- `--idle-threshold` — motion threshold (0–1)
- `--keyframes` — enable scene-keyframe extraction
- `--subtitles` — reserved for Phase 2 ( Whisper subtitle burn-in )

Dependencies
------------
Runtime:
- `opencv-python` — frame reading and diff
- `scenedetect[opencv]` — scene boundary detection
- `typer[all]` — CLI framework
- `ffmpeg-python` — Pythonic FFmpeg wrapper for encode stage

Optional:
- `openai-whisper` — subtitle transcription (Phase 2)

Dev:
- `pytest`, `pytest-cov`

Phase 2 Ideas
-------------
- Integrate Whisper for automatic subtitle generation and burn-in
- Export a JSON activity map with per-second activity scores
- Interactive GUI (Tkinter or web-based) for threshold tuning and preview
- Adaptive speed map: use a second threshold to distinguish "low-activity" (3x) from "active" (1x)
- Audio-aware idle handling: preserve quiet audio sections instead of cutting them
