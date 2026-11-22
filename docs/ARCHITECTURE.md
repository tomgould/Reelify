# Architecture

`reelify process` takes a long screen recording through an optional deduplication step and four stages, with captions as an optional fifth:

    dedup -> analyse -> classify -> speed-map -> encode -> (optional) enrich

## Deduplication

On by default (`--no-dedup` turns it off). FFmpeg's `mpdecimate` filter compares each frame with the last one it kept, in 8x8 blocks, and drops it when the blocks barely differ. `--dedup-similarity` (0 to 1, default 0.90) sets the thresholds. With `--frametime N`, each kept frame is held for N seconds. The later stages work on this shorter temporary file, which is deleted at the end.

## Stage 1: Activity analysis

ffprobe reads the duration, frame rate and size. FFmpeg then decodes every fourth frame as a 320-pixel-wide greyscale thumbnail and pipes the raw pixels out. Each thumbnail is compared with the one before by mean absolute difference (MAD), normalised to 0 to 1 by dividing by 255. A sample counts as active when its MAD is at least `idle_threshold` (default 0.02) and idle otherwise. This mirrors auto-editor's `--edit motion:threshold=0.02`.

## Stage 2: Chunk classification

Each sample's label applies to the four frames it stands for. Every frame within 0.3 seconds of an active frame is marked active too, in two linear passes (forwards, then backwards), so short idle gaps inside activity don't produce hard cuts. Runs of frames with the same label become chunks:

- idle: no meaningful motion anywhere in the chunk
- active: at least one frame over the threshold, after padding

## Stage 3: Speed map

| Chunk type | Speed | Notes |
|------------|-------|-------|
| idle       | cut   | Dropped entirely |
| active     | 1x    | Normal speed, unless a target length applies |

The speed map is a list of `Segment(start_frame, end_frame, speed)`. With `--max-duration`, if the kept segments add up to more than the target, all of them get the same speed, `total / target`, capped at 8x, so a very long recording can still come out over the target. Without it, nothing is sped up.

## Stage 4: Encode

Each kept segment is encoded on its own, several at once, then the pieces are joined with the concat demuxer (`-c copy`). A segment is cut with `-ss` and `-t` before `-i`, so both are measured on the source's timeline; because every segment is re-encoded with `libx264`, FFmpeg makes that seek frame-accurate. The video filter is `setpts=(PTS-STARTPTS)/speed,fps=<source fps>:round=up`, which keeps the segment's first frame and the source frame rate, so N frames at speed S become ceil(N / S) frames. Sound is retimed with a chain of `atempo` filters of at most 2x each.

## Subtitles (optional)

With `--subtitles` and the `subtitles` extra installed, the audio is extracted, transcribed locally, written as an `.srt` beside the output and burned into the video.

## Stage 5 (optional): Keyframes and captions

PySceneDetect's `ContentDetector` finds the scene boundaries, and one frame from each scene is saved as a JPEG in a folder beside the output (`{output_stem}_keyframes/` for `process`, `{input_stem}_keyframes/` for `analyse`). `process` saves keyframes by default (`--no-keyframes` turns this off).

With `--enrichment`, or always in `analyse`, each keyframe goes to a vision provider for a one-sentence caption: LM Studio on the same machine by default, or Google's Gemini API with `REELIFY_PRO=1`. Both commands write the captions, each matched to its nearest chunk, and the per-chunk activity scores to a JSON file in the same format: `{input_stem}_analysis.json` for `analyse`, `{output_stem}.json` beside the output video for `process --enrichment`.

## Formats

- Input: anything FFmpeg can decode, such as MP4, MKV and WebM
- Output: always MP4 (`libx264`)

## Commands

The CLI is built with Typer.

`reelify process` runs the compression pipeline:

    reelify process input.webm [--output summary.mp4] [--max-duration 120] \
                               [--idle-threshold 0.02] [--no-keyframes] [--subtitles] \
                               [--enrichment] [--provider local|api|auto] [--scoring fast|deep] \
                               [--no-dedup] [--dedup-similarity 0.90] \
                               [--frametime 0] [--preset cli]

- `input_path`: source video (required)
- `--output`: destination path; defaults to `{input_stem}_summary.mp4`
- `--max-duration`: target output length in seconds; 0, the default, means no target
- `--idle-threshold`: motion threshold (0 to 1)
- `--keyframes / --no-keyframes`: save scene keyframes (on by default)
- `--subtitles`: transcribe and burn in subtitles
- `--enrichment / --no-enrichment`: caption the keyframes and write `{output_stem}.json` beside the output video
- `--provider`, `--scoring`: as for `analyse`; used with `--enrichment`
- `--dedup / --no-dedup`, `--dedup-similarity`, `--frametime`: see Deduplication
- `--preset cli`: settings for terminal recordings (0.95 similarity, 3 second frame time, 0.005 idle threshold)

`reelify analyse` runs analysis and captioning and writes JSON:

    reelify analyse input.webm [--provider local|api|auto] [--scoring fast|deep] \
                               [--output analysis.json]

- `input_path`: source video (required)
- `--provider`: `local` (default), `api` (needs `REELIFY_PRO=1`) or `auto` (local first, then the API when `REELIFY_PRO=1`)
- `--scoring`: `fast` captions each keyframe once; `deep` captions up to three frames per chunk (first, middle and last)
- `--output`: destination path; defaults to `{input_stem}_analysis.json`

## Dependencies

Runtime:
- `numpy`: frame differences (installed with OpenCV)
- `scenedetect[opencv]` and `opencv-python`: scene boundary detection and keyframe images
- `typer[all]`: CLI framework
- `ffmpeg-python`: FFmpeg wrapper
- `requests`: HTTP client for LM Studio
- `google-genai`: client for the Gemini API
- `Pillow`: resizes and encodes images before they go to a vision provider

Optional:
- `subtitles` extra: speech transcription for `--subtitles`

Dev:
- `pytest`, `pytest-cov`
