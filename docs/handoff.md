# Reelify — Session Handoff

**Working directory:** `/home/tg/Reelify`  
**Date:** 2026-04-25

---

## What is Reelify?

A Python CLI tool that takes a long screen recording (e.g. 1 hour MP4) and produces a ~2 minute summary video by:

1. Removing/compressing idle gaps (no mouse/keyboard/visual change)
2. Speeding up low-activity sections (smooth timelapse feel, not hard cuts)
3. Extracting the most visually distinct keyframes
4. Annotating the output with subtitles (window titles, app names, visible text)
5. Outputting a compressed summary MP4 + optional JSON activity map

---

## What's been done this session

- [x] Project named: **Reelify**
- [x] Directory created: `/home/tg/Reelify` with `git init`
- [x] Trawl reference pool created: **Reelify - Reference** (`mullet-pools/reelify-reference`)
- [x] Three reference repos cloned and symlinked into the pool:
  - `WyattBlue/auto-editor` — idle/motionless gap removal
  - `Breakthrough/PySceneDetect` — scene boundary + keyframe detection
  - `collingreen/chronolapse` — screenshot timelapse pipeline
- [x] Indexing triggered (background) — check with: `mullet pool health reelify-reference`

---

## First thing to do in the new session

1. Verify indexing completed:
   ```
   mullet pool health reelify-reference
   ```

2. Then kick off planning — ask Claude to:
   > "The Reelify reference pool is indexed. Use it to research how auto-editor handles motion detection, how PySceneDetect extracts keyframes, and how chronolapse structures its pipeline. Then produce a PLAN.md for Reelify Phase 1 MVP."

3. Work in `/home/tg/Reelify`

---

## Key decisions already made

- **Language:** Python (CLI-first, Click or Typer)
- **Platform:** Linux (Ubuntu 22.04+)
- **Core deps:** FFmpeg (system), OpenCV, PySceneDetect, optional: auto-editor, Whisper
- **No cloud API required** — runs fully locally
- **Input:** MP4/MKV screen recording
- **Output:** summary MP4 + optional JSON metadata

---

## Tech stack (rough)

| Concern | Tool |
|---|---|
| Idle gap removal | OpenCV frame diff + input event hooks |
| Scene detection / keyframes | PySceneDetect |
| Timelapse / speedup / concat | FFmpeg (`setpts`, `concat` demuxer) |
| Subtitles | FFmpeg `drawtext` / SRT burn-in |
| Audio transcription (optional) | Whisper (local) |
| CLI | Typer or Click |

---

## Reference pool

**Pool name:** `reelify-reference`  
**Search via:** `mcp__mullet__trawl_search_pool` with `pool_name: reelify-reference`  
**Repos indexed:**
- auto-editor (`/home/tg/BrightSites/www/github_auditer/data/WyattBlue/public/auto-editor`)
- PySceneDetect (`/home/tg/BrightSites/www/github_auditer/data/Breakthrough/public/PySceneDetect`)
- chronolapse (`/home/tg/BrightSites/www/github_auditer/data/collingreen/public/chronolapse`)
