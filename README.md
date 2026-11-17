# Apex Highlight

TimePin sidecar plugin for Apex Legends highlight detection. The plugin ports the apex-highlight-clipper (AHC) HUD recognition pipeline: ultrawide HUD layout detection, dual-model digit reading, combat-event extraction (shoot / hit / kill / downed), downed-protection intervals, and combat-outcome trimming (own result, squad wipe, champion, teammate spectate tail). Output is a versioned `Mark[]` list the host merges into the timeline.

## Protocol and host contract

- Protocol version `1.0.0`, JSON-RPC 2.0 over UTF-8 JSON Lines (`main.py`).
- Methods: `initialize`, `run`, `cancel`, `shutdown`; `progress` and `log` notifications.
- `cancel` is cooperative: a dedicated stdin reader accepts it while `run` executes, the recognition loop checks the cancellation event between frames/batches, and `run` returns partial marks/logs.
- `frames.extract` is a reverse host-service call. `output_format` is explicitly `png`, `delivery` is `files`, extraction windows are bounded to at most 60 seconds or 128 samples, and temporary frames live only under `outputDir`.
- The plugin never decodes video itself; every pixel arrives as a host-extracted PNG inside a scratch dir that is removed after the request completes (deletion is skipped if the request was cancelled/failed — the host may still be writing).

### Known host-contract differences (verified against TimePin `beta` `fbc56579`)

- TimePin sends host services as `hostServices["frames.extract"]`; `docs/frame-service.md` says `framesExtract`. The plugin accepts both and selects the matching request/response field naming (`camel` for the Rust host, `snake` for the documented service).
- Current Rust wire uses `videoPath`, `ptsSec`, `totalFrames`, `imageBase64`; the documented Python client uses `video_path`, `pts_sec`, `total_frames`, `image_base64`.
- TimePin manifest validation uses lowercase capabilities. This manifest uses `videoframes` and `filewrite`; camelCase values such as `fileWrite` are rejected by the host.
- `parametersSchema` is the properties map directly, not `{type:"object",properties:{...}}`.
- The TimePin manifest validator rejects relative dependency paths. `timepin_vision` therefore uses a pinned immutable GitHub archive URL in `plugin.json`/`requirements.txt`, targeting the current `beta` commit `fbc56579a7dac5ed40d4cd8cff29e27c8acb1e99`; use a stable release artifact instead when TimePin publishes one.
- A cancelled frame request only abandons the plugin's wait; it does not prove host FFmpeg stopped. The plugin does not delete a scratch directory until the request completes.

## Layout

```text
plugin.json                  TimePin manifest
main.py                      JSON-RPC stdio entry point (warms cv2/numpy/ORT
                             on the main thread — native imports from a worker
                             thread can deadlock on Windows)
src/errors.py                shared RpcError / Cancelled
src/config.py                manifest and parameter validation
src/protocol.py              single-reader bidirectional transport
src/app.py                   lifecycle, layout probe, extraction loop,
                             HUD/outcome orchestration, cancellation
src/frames.py                frames.extract adapter (wire validation)
src/extract.py               batch windowing + host-scratch cleanup
src/layout.py                HUD layout detection (AHC anchored-hud-v2):
                             pillarbox/stretch/compact candidates, counter-marker
                             offset scoring, voice-icon name-row confirmation
src/hud.py                   per-frame HUD reads: ammo/damage/kills counters,
                             downed marker, inactive view, icon similarity
src/digits.py                HUD digit reader (AHC hud-digits-v1): primary
                             CTC ONNX + PP-OCRv6 rec check model agreement
src/ocr.py                   explicit-backend OCR engine factory
                             (cpu/dml/cuda, no silent CPU fallback)
src/stat_counter_fallback.py icon-anchored counter fallback + zero-glyph proof
src/events.py                pure event detection: shoot/hit/kill rules,
                             segment merging, combat classification, Mark adapter
src/downed.py                downed-protection intervals (terminal/recovery tail)
src/outcome.py               combat-outcome reader (AHC own-center-result-v4):
                             own result prefix, squad wipe/champion, teammate
                             spectate continuation, segment decision metadata
src/pipeline.py              finalize_marks orchestration
src/resources.py             sha256-verified resource loading (provenance.json)
resources/                   HUD digit model, profiles, marker/OCR templates,
                             combat-outcome assets, provenance.json
tests/                       unit, golden-parity, and subprocess protocol tests
requirements.txt             pinned TimePin SDK archive + onnxruntime
```

## Recognition resources

`resources/` ships the HUD digit model (`hud_digits_v1.onnx`, sha256 `b59b9f…f5f6`), the PP-OCRv6 rec check model (`models/ppocrv6_rec_small.onnx`, sha256 `6f327246…c484`), the ultrawide HUD profile, combat-outcome profile, and the full template set (counter/knockdown markers, statistics icons, voice icon, combat-outcome toast templates with masks). `resources/provenance.json` + `src/resources.py` verify every file by sha256 at load; a missing or corrupted resource raises `FileNotFoundError`/`RuntimeError` instead of silently degrading. See `resources/README.md` for source paths and the unresolved upstream redistribution-license caveat. The plugin never downloads models at runtime.

## Run flow

1. `initialize` — negotiate protocol + host services (`frames.extract`/`framesExtract` v1).
2. `run` — bounded windows of host-delivered frames:
   - Layout probe (`src/layout.detect_images`): picks the HUD profile among ultrawide/stretched/pillarbox/compact candidates by counter-marker score; `unconfirmed` falls back to the default candidate and may be upgraded by the voice-icon name row check.
   - Per batch (`src/hud.HudReader.read_batch`): ammo/damage/kills crops → digit reader (dual-model agreement) or icon-anchored fallback; downed marker score; inactive-view strip check. Each row feeds `src/outcome.OutcomeReader.observe` (toast prefix / squad-wipe / champion / spectate) before the scratch dir is cleaned.
   - `finalize_marks` (`src/pipeline`): downed intervals → event detection → segment merge/clip by `gap/min/max` params → combat classification (kept / review / rejected) → outcome decisions and counters → `Mark[]`.
3. Marks carry `metadata` with classification, decision, coverage, signals, counter basis, and outcome kind so the host UI can explain each segment.
4. `cancel`/`shutdown` stop the run cooperatively and return partial output.

Decode/model/backend failures degrade per-batch to timestamp skeleton rows and log lines (`Frame decode degraded`, `HUD read degraded`, `Layout probe frames undecodable`) rather than failing the whole run; missing models/templates/backends surface as actionable errors (`FileNotFoundError`, `RuntimeError` naming the provider) — there is no silent CPU fallback.

## Weapon recognition

`params.enable_weapon_detection` is a strict boolean, defaults to `true`, and can
be set to `false` to skip weapon OCR, catalog loading, learning, and weapon
metadata entirely. The existing HUD icon signature is still read for the combat
detector's weapon-swap guard.

Weapon identification combines native HUD name OCR, the packaged AHC icon
catalog, and per-output-directory learned icon samples. Observations are taken
before each extracted-frame batch is cleaned; only compact timestamped results
survive across batches. Combat marks summarize the **active** weapon, not both
carried slots; inactive/spectator frames do not contribute.

TimePin protocol v1 currently defines `Mark.metadata` as `dict[str, str]`, so
structured values use JSON strings (not nested wire arrays):

```json
{"weapons":"[\"R-99\",\"Flatline\"]","weapon_timeline":"[{\"t\":12.5,\"weapon\":\"R-99\",\"kind\":\"shoot\"},{\"t\":15.0,\"weapon\":\"Flatline\",\"kind\":\"shoot\"}]"}
```

Consumers use `JSON.parse(mark.metadata.weapons)` (and likewise for
`weapon_timeline`) to obtain the arrays requested by the feature. Times are
absolute seconds in the input video. Unrecognized weapons produce no invented
name.

Learning never changes the packaged catalog or source MP4. Samples are persisted
only as `outputDir/learned-weapons.json` and merged on the next `run` using that
same authorized output directory. A fresh output directory starts with the
packaged catalog alone; cross-task learning requires reusing/copying the learned
file under the host's explicit directory authorization. Existing symlinked
learning paths are rejected instead of following them outside the sandbox.

## Development

Python `>=3.10,<3.13` is required. From this repository:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements-test.txt
.venv\Scripts\python -m pytest tests/ -v
.venv\Scripts\python -m ruff check tests/baseline.py tests/test_baseline.py tests/test_real_video_baseline.py tests/test_parameter_matrix.py tests/test_hud.py tests/test_events.py tests/test_layout.py tests/test_weapon.py
```

The suite covers the protocol layer, host-contract differences, wire-style handling, cancellation, golden-parity checks against the AHC algorithms (layout candidates, digit prepare/decode/agreement, counter fallback, event/segment/down/outcome logic), and subprocess end-to-end runs with a mock host.

### Validation levels

- `tests/test_hud.py`, `test_events.py`, `test_layout.py`, `test_weapon.py`: deterministic synthetic crops/sequences and injected recognizers. These are not real-footage accuracy measurements.
- `tests/test_parameter_matrix.py`: 18 visible cases across `{keyframes,image2} x {cpu,dml,cuda} x {1,2,4}` through initialize/run and the host-file contract. Metadata-stub frames deliberately avoid inference; separate real-session checks enforce explicit provider availability. Mocked GPU requests do **not** prove GPU recognition.
- `.github/workflows/tests.yml`: Windows CPU runners on Python 3.10/3.11/3.12 install the actual SDK and test dependencies; no private footage, GUI, or physical GPU acceptance.
- `tests/baseline.py`: opt-in real-video run versus an independent original AHC `analysis.json`. It reports all-Mark counts and strict `<0.5s` start/end errors, preserves mismatches and returns failure on them, rejects unrelated/stale oracles, and checks input integrity. See [private fixture acquisition and baseline commands](tests/fixtures/README.md).

Real recordings/oracles are not bundled or available for this validation delivery. Real Apex parity (both aspect ratios and combat/weapon scenarios), physical GPU inference, and Rust/Tauri host/UI behavior remain **unverified**. The optional generated-video FFmpeg check proves extraction/lifecycle only, not recognition accuracy.

Installation of the immutable SDK archive requires network access and permissions to read the TimePin repository. For offline/local development, install the actual SDK source with `python -m pip install "<TimePin-checkout>/plugins/_shared/timepin_vision"`; do not use `../...` as a manifest dependency. `onnxruntime` selects the CPU package by default; `backend=dml/cuda` requires the matching onnxruntime build — an unavailable provider raises an explicit `RuntimeError`, never a silent CPU fallback.

## Lifecycle examples

Each object below is one line on stdin. stdout contains protocol messages only, and all lines are flushed immediately.

```json
{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"1.0.0","capabilities":{"hostServices":{"frames.extract":"1.0.0"}}}}
{"jsonrpc":"2.0","id":2,"method":"run","params":{"taskId":"task-1","video":{"path":"C:/videos/中文视频.mp4","duration":120.0},"params":{"extract_mode":"image2","sample_fps":2},"outputDir":"C:/output/task-1"}}
```

`video.path` and `outputDir` must be existing absolute paths; the output directory is authorized and created by TimePin. The plugin emits `frames.extract` requests with its own integer id. The host must answer each request before `run` can complete; merely piping the two lines above without a bidirectional host is not a full smoke test. Progress includes host fields `taskId/progress/message/marksSoFar` plus versioned `percent/stage` for diagnostics.

```json
{"jsonrpc":"2.0","id":3,"method":"cancel","params":{"taskId":"task-1"}}
{"jsonrpc":"2.0","id":4,"method":"shutdown","params":{}}
```

Cancellation is per task; omitted `taskId` means the active run. Wrong-task cancellation has no effect. A second concurrent run or initialize is rejected. Both requests and notifications are supported; notifications never receive responses. JSON-RPC batch messages are deliberately unsupported (`-32600`).

## Limitations / integration acceptance

- The plugin consumes host-extracted PNG frames only; it never opens the MP4, and never writes video/chapter/clip output — only the `Mark[]` response.
- The recognition logic targets AHC's `anchored-hud-v2` ultrawide layout; other aspect ratios work through the pillarbox/stretch/compact candidates but a layout that never confirms falls back with `layout_status: unconfirmed` and reduced accuracy.
- Unit and subprocess mock-host tests exercise the real sidecar, reverse RPC, UTF-8 paths, cancellation, errors, shutdown, and golden parity of the ported algorithms. They do **not** prove Tauri UI discovery, actual FFmpeg extraction, real footage accuracy, or embedded-runtime packaging.
- TimePin currently serializes `run` and `cancel` behind the same process mutex (`src-tauri/src/plugins/manager.rs`), so UI cancellation may not reach this independently cancellable sidecar until run ends. Fixing that is a host task, outside this plugin PR.
- If extraction is cancelled, times out, or fails before returning, the private `apex-frames-*` directory is retained to avoid racing a still-writing host. Remove it under the authorized output directory once host extraction has stopped. Successfully analyzed batches are cleaned immediately.
- The pinned TimePin SDK is available in source, but embedded-runtime/venv installation and authenticated/offline packaging remain host/deployment acceptance work.
