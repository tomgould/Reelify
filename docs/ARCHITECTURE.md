# Architecture

The Reelify pipeline processes a long screen recording through five discrete stages to produce a compressed summary video.

    analyse -> classify -> speed-map -> encode -> (optional) enrich

- **Analyse**: compute per-frame motion scores with OpenCV
- **Classify**: group frames into idle/active chunks with margin padding
- **Speed-map**: assign each chunk a target speed (cut, 8×, 3×, 1×)
- **Encode**: assemble segments via FFmpeg concat demuxer + `setpts`
- **Enrich (optional)**: extract representative scene-change JPEGs and caption them with an LLM vision provider

## Stage 1: Activity Analysis

OpenCV reads the input video frame by frame. Adjacent frames are converted to grayscale and compared with a mean absolute difference (MAD). The MAD is normalised to 0–1 by dividing by 255. A frame is marked **idle** when MAD < `idle_threshold` (default 0.02), otherwise **active**. This mirrors auto-editor's `--edit motion:threshold=0.02` heuristic.

## Stage 2: Chunk Classification

Consecutive frames with the same activity label are merged into chunks. A margin of ~0.3 s is applied around every active chunk so that tiny idle gaps inside an active region do not produce hard cuts. After padding, adjacent chunks of the same type are merged again.

Resulting chunk types:
- **idle** — no meaningful motion for the entire chunk
- **active** — at least one frame exceeded the threshold (after padding)

## Stage 3: Speed Map

Each chunk is assigned a playback speed:

| Chunk type | Speed | Notes |
|------------|-------|-------|
| idle       | cut   | Dropped entirely; if audio is present in a future version, use 8× instead |
| low-activity | 3×  | Not yet distinguished in current builds; active chunks below a second speed threshold could map here |
| active     | 1×    | Preserved at normal speed, ideally copied without re-encode |

The speed map is a list of `(start_time, end_time, speed)` tuples. The total duration after applying speeds must be <= `max_duration` (default 120 s). If the map exceeds the limit, speeds of the longest active chunks can be increased iteratively.

## Stage 4: FFmpeg Encode

FFmpeg encodes the final video using the concat demuxer. Each segment is described by a concat file entry that optionally applies the `setpts` filter for speed changes:

    file 'segment_0.mp4'
    inpoint 12.500
    outpoint 45.200
    ...

Active 1× segments are copied (`-c copy`) when possible to avoid generation loss. Speed-adjusted segments are re-encoded with a fast preset (`libx264` / `libx265`). The concat demuxer guarantees frame-accurate stitching.

## Stage 5 (optional): Keyframe Extraction & Vision Enrichment

When `analyse` is invoked (or `process --keyframes` is passed), PySceneDetect runs `ContentDetector` (and optionally `AdaptiveDetector`) over the original video to locate scene boundaries. One representative frame from each scene is saved as a JPEG in an adjacent folder named `{input_stem}_keyframes/`.

During enrichment, each keyframe is sent to a vision provider (local LM Studio or Gemini API) for a one-sentence caption. These captions, together with per-chunk activity scores, are written to a JSON metadata file.

## Supported Formats

- **Input**: MP4, MKV, WebM (any container OpenCV/FFmpeg can decode)
- **Output**: always MP4 (`libx264`), regardless of input container

## CLI Interface

Built with Typer. Two subcommands are exposed:

**`reelify process`** — run the full compression pipeline:

    reelify process input.webm [--output summary.mp4] [--max-duration 120] \
                               [--idle-threshold 0.02] [--keyframes] [--subtitles]

Parameters:
- `input_path` — source video (required; MP4, MKV, WebM)
- `--output` — destination path; defaults to `{input_stem}_summary.mp4`
- `--max-duration` — target output length in seconds
- `--idle-threshold` — motion threshold (0–1)
- `--keyframes` — enable scene-keyframe extraction
- `--subtitles` — reserved for future use (Whisper subtitle burn-in)

**`reelify analyse`** — run analysis + optional vision enrichment and emit JSON:

    reelify analyse input.webm [--provider local|api|auto] [--scoring fast|deep] \
                               [--output analysis.json]

Parameters:
- `input_path` — source video (required)
- `--provider` — vision provider: `local` (default), `api` (requires `REELIFY_PRO=1`), or `auto`
- `--scoring` — `fast` for quick captions, `deep` for more detailed analysis
- `--output` — destination path; defaults to `{input_stem}_analysis.json`

## Component Dependencies

Runtime:
- `opencv-python` — frame reading and diff
- `scenedetect[opencv]` — scene boundary detection
- `typer[all]` — CLI framework
- `ffmpeg-python` — Pythonic FFmpeg wrapper for encode stage
- `requests` — HTTP client for LM Studio local vision provider
- `google-genai` — Google Gemini API client
- `Pillow` — image resize and encoding before sending to vision providers

Optional:
- `openai-whisper` — subtitle transcription (future)

Dev:
- `pytest`, `pytest-cov`
