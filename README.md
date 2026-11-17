# Apex Highlight

TimePin sidecar plugin for Apex Legends highlight detection. This repository currently implements the versioned plugin protocol skeleton and host-owned frame extraction integration; event detection is intentionally stubbed until the next milestone.

## Protocol and host contract

- Protocol version `1.0.0`, JSON-RPC 2.0 over UTF-8 JSON Lines (`main.py`).
- Methods: `initialize`, `run`, `cancel`, `shutdown`; `progress` and `log` notifications.
- `cancel` is cooperative: a dedicated stdin reader accepts it while `run` executes, `analyze` checks the cancellation event, and `run` returns partial marks/logs.
- `frames.extract` is a reverse host-service call. `output_format` is explicitly `png`, `delivery` is `files`, extraction windows are bounded to at most 60 seconds or 128 samples, and temporary frames live only under `outputDir`.

### Known host-contract differences (verified against TimePin `beta` `fbc56579`)

- TimePin sends host services as `hostServices["frames.extract"]`; `docs/frame-service.md` says `framesExtract`. The plugin accepts both and selects the matching request/response field naming (`camel` for the Rust host, `snake` for the documented service).
- Current Rust wire uses `videoPath`, `ptsSec`, `totalFrames`, `imageBase64`; the documented Python client uses `video_path`, `pts_sec`, `total_frames`, `image_base64`.
- TimePin manifest validation uses lowercase capabilities. This manifest uses `videoframes` and `filewrite`; camelCase values such as `fileWrite` are rejected by the host.
- `parametersSchema` is the properties map directly, not `{type:"object",properties:{...}}`.
- The TimePin manifest validator rejects relative dependency paths. `timepin_vision` therefore uses a pinned immutable GitHub archive URL in `plugin.json`/`requirements.txt`, targeting the current `beta` commit `fbc56579a7dac5ed40d4cd8cff29e27c8acb1e99`; use a stable release artifact instead when TimePin publishes one.
- A cancelled frame request only abandons the plugin's wait; it does not prove host FFmpeg stopped. The skeleton does not delete a scratch directory until the request completes.

## Layout

```text
plugin.json          TimePin manifest
main.py              JSON-RPC stdio entry point
src/errors.py        shared RpcError / Cancelled
src/config.py        manifest and parameter validation
src/protocol.py      single-reader bidirectional transport (implemented in TPIN-7)
src/app.py           lifecycle, path checks, extraction windows, cancellation
src/frames.py        frames.extract adapter (implemented in TPIN-7)
src/pipeline.py      cancellable analyze traversal (implemented in TPIN-7)
src/events.py        event-to-Mark adapter (implemented in TPIN-7)
resources/           HUD model, profile, and marker templates
tests/               unit and end-to-end protocol tests
requirements.txt     pinned TimePin SDK archive + onnxruntime
```

## Recognition resources

`resources/` contains the task-requested HUD digit model, the single available ultrawide profile, and its two marker templates, copied byte-for-byte. See `resources/README.md` for exact source paths, hashes, and the unresolved upstream redistribution-license caveat. Profile template paths are relative to `resources/`. This milestone never loads the model or templates and does not download secondary models.

## Development

Python `>=3.10,<3.13` is required. From this repository:

```powershell
python -m unittest discover -s tests -v
python -m compileall -q main.py src tests
ruff check main.py src tests
ruff format --check main.py src tests
```

The protocol stub and its tests use only Python's standard library. Recognition dependencies are declared but intentionally not imported during handshake/stub analysis. For the later recognition implementation:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

Installation of the immutable SDK archive requires network access and permissions to read the TimePin repository. It has not been verified in an isolated plugin venv here. For offline/local development, install the actual SDK source with `python -m pip install "<TimePin-checkout>/plugins/_shared/timepin_vision"`; do not use `../...` as a manifest dependency. `onnxruntime` currently selects the CPU package; `backend=dml/cuda` is accepted/reserved only and does not enable GPU inference in this milestone.

## Lifecycle examples

Each object below is one line on stdin. stdout contains protocol messages only, and all lines are flushed immediately.

```json
{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"1.0.0","capabilities":{"hostServices":{"frames.extract":"1.0.0"}}}}
{"jsonrpc":"2.0","id":2,"method":"run","params":{"taskId":"task-1","video":{"path":"C:/videos/中文视频.mp4","duration":120.0},"params":{"extract_mode":"image2","sample_fps":2},"outputDir":"C:/output/task-1"}}
```

`video.path` and `outputDir` must be existing absolute paths; the output directory is authorized and created by TimePin. The plugin emits a `frames.extract` request with its own integer id. The host must answer this request before normal `run` completion; merely piping the two lines above without a bidirectional host is not a full smoke test. Progress includes host fields `taskId/progress/message/marksSoFar` plus versioned `percent/stage` for diagnostics. No detections are fabricated: successful runs return `marks:[]`, `evidencePaths:[]`, and an explicit stub log.

```json
{"jsonrpc":"2.0","id":3,"method":"cancel","params":{"taskId":"task-1"}}
{"jsonrpc":"2.0","id":4,"method":"shutdown","params":{}}
```

Cancellation is per task; omitted `taskId` means the active run. Wrong-task cancellation has no effect. A second concurrent run or initialize is rejected. Both requests and notifications are supported; notifications never receive responses. JSON-RPC batch messages are deliberately unsupported (`-32600`).

## Limitations / integration acceptance

- This is not an Apex recognition implementation: segment parameters are validated/reserved, and no MP4, chapter, or clip output is ever modified/generated.
- Unit and subprocess mock-host tests exercise the real sidecar, reverse RPC, UTF-8 paths, cancellation, errors, and shutdown. They do **not** prove Tauri UI discovery, actual FFmpeg extraction, SDK installation, or recognition accuracy.
- TimePin currently serializes `run` and `cancel` behind the same process mutex (`src-tauri/src/plugins/manager.rs`), so UI cancellation may not reach this independently cancellable sidecar until run ends. Fixing that is a host task, outside this plugin PR.
- If extraction is cancelled, times out, or fails before returning, the private `apex-frames-*` directory is retained to avoid racing a still-writing host. Remove it under the authorized output directory once host extraction has stopped. Successfully analyzed batches are cleaned immediately.
- The pinned TimePin SDK is available in source, but embedded-runtime/venv installation and authenticated/offline packaging remain host/deployment acceptance work.
