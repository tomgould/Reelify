# TODO

Tasks ordered by priority. Each has loop mode, files, plan, and acceptance criteria.

## P0 — Critical bugs (fix before any other work)

### [LOOP: Micro] Bug: `pipeline.py` variable shadowing (#1)
- **Files:** `src/reelify/pipeline.py`, `tests/test_cli.py`
- **Plan:**
  1. In `pipeline.py`, rename the inner `segments` variable in the subtitles block to `sub_segments`.
  2. In `tests/test_cli.py`, add a test that calls `pipeline.run()` with `subtitles=True` on a stub video and verifies it does not crash or produce wrong output.
- **Acceptance:** The `pipeline.py` subtitles block no longer shadows the outer `segments` list, and the new test passes.
- **Status:** [ ] todo

### [LOOP: Mini] Bug: `cli.py` enrichment flag unreachable (#5)
- **Files:** `src/reelify/cli.py`, `tests/test_cli.py`
- **Plan:**
  1. Add `--enrichment/--no-enrichment` Typer option to the `process()` command.
  2. Wire the option through to `ReelifyConfig(enrichment=enrichment, ...)`.
  3. Add a CLI test that invokes `process` with `--enrichment` and verifies enrichment is triggered (mock the vision provider).
- **Acceptance:** The `--enrichment` flag is available in the CLI, reaches `ReelifyConfig`, and the new test confirms the enrichment branch executes.
- **Status:** [ ] todo

## P1 — High priority

### [LOOP: Micro] Bug: `encoder.py` TOCTOU `mktemp` race (#8)
- **Files:** `src/reelify/cli.py` (`_dedup_video()`)
- **Plan:**
  1. Replace `tempfile.mktemp()` with `tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)`.
  2. Close the temporary file immediately, then use `.name` as the path for ffmpeg.
- **Acceptance:** No use of `mktemp()` remains in `_dedup_video()`; the temporary file is created atomically before ffmpeg writes to it.
- **Status:** [ ] todo

### [LOOP: Micro] Bug: `encoder.py` frame-seek accuracy (#3)
- **Files:** `src/reelify/encoder.py`
- **Plan:**
  1. In `_build_segment_command()`, move `-ss` and `-t` to after `-i` (input seeking → accurate but slower).
  2. Add a comment explaining the trade-off (fast seek before `-i` can land on non-keyframes vs accurate seek after `-i`).
- **Acceptance:** `-ss` and `-t` appear after `-i` in the generated ffmpeg command, and segments start on the correct frame without grey/corrupt first frames.
- **Status:** [ ] todo

### [LOOP: Mini] Perf: `classifier.py` O(n²) margin-padding (#2)
- **Files:** `src/reelify/classifier.py`, `tests/test_classifier.py`
- **Plan:**
  1. Replace the O(n × margin_frames) padding loop with a two-pass O(n) approach (forward pass + backward pass) or a sliding-window / cumulative-sum dilation.
  2. Ensure existing `tests/test_classifier.py` stays green.
  3. Add a performance test with a 54,000-frame input that completes in <1s.
- **Acceptance:** Existing tests pass, and the new performance test runs in under one second.
- **Status:** [ ] todo

## P2 — Medium priority

### [LOOP: Micro] Dead code: `enricher.py` unused `_keyframes_for_chunk` (#6)
- **Files:** `src/reelify/enricher.py`
- **Plan:**
  1. Delete the `_keyframes_for_chunk` function entirely.
  2. Remove any now-unused imports.
- **Acceptance:** `_keyframes_for_chunk` no longer exists in `enricher.py`, and all tests still pass.
- **Status:** [ ] todo

### [LOOP: Mini] Perf: `encoder.py` sequential segment encoding (#9)
- **Files:** `src/reelify/encoder.py`, `tests/test_encoder.py`
- **Plan:**
  1. Use `concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count())` to fan out `_build_segment_command` + `subprocess.run` calls.
  2. Collect results in order.
  3. Preserve the `progress_callback` (fire it as each future completes).
- **Acceptance:** Segments are encoded in parallel, results remain in the correct order, progress callbacks fire per completed segment, and all encoder tests pass.
- **Status:** [ ] todo

### [LOOP: Micro] Code quality: `subtitles.py` duplicated ffmpeg call (#10)
- **Files:** `src/reelify/subtitles.py`
- **Plan:**
  1. Extract a private helper `_run_ffmpeg_burn(input_path, srt_path, output_path)` that wraps the ffmpeg invocation.
  2. Call the helper from both branches of `if video_path == output_path` in `burn_subtitles`.
- **Acceptance:** The ffmpeg invocation appears exactly once in `subtitles.py`, and subtitle burn-in tests remain green.
- **Status:** [ ] todo

## P3 — Low priority / polish

### [LOOP: Micro] Code quality: `analyser.py` assert → RuntimeError (#4)
- **Files:** `src/reelify/analyser.py`
- **Plan:**
  1. Replace `assert proc.stdout is not None` with `if proc.stdout is None: raise RuntimeError("ffmpeg stdout is None")`.
- **Acceptance:** No `assert` remains for the stdout check; `RuntimeError` is raised on the same condition, and tests still pass.
- **Status:** [ ] todo

### [LOOP: Micro] Code quality: `pipeline.py` print → log callable (#13)
- **Files:** `src/reelify/pipeline.py`
- **Plan:**
  1. Add optional `log: Callable[[str], None] = lambda _: None` parameter to `run()`.
  2. Replace the two `print(...)` calls with `log(...)`.
- **Acceptance:** `pipeline.run()` accepts a `log` callable, prints nothing by default, and all pipeline tests pass.
- **Status:** [ ] todo

### [LOOP: Micro] Code quality: `analyser.py` variable name `l` (#12)
- **Files:** `src/reelify/analyser.py`
- **Plan:**
  1. Rename `l` to `line` in the list comprehension on line 33.
- **Acceptance:** The variable is renamed, and all analyser tests pass.
- **Status:** [ ] todo

## Completed

(empty initially)
