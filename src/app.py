"""Plugin lifecycle and bounded, host-owned frame extraction."""

from __future__ import annotations

import shutil
import tempfile
import threading
from functools import partial
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from .config import (
    PLUGIN_NAME,
    PROTOCOL_VERSION,
    load_manifest,
    parse_params,
    positive_number,
)
from .errors import Cancelled, RpcError
from .events import Mark
from .frames import FrameClient
from .pipeline import analyze

if TYPE_CHECKING:
    from .protocol import JsonRpcServer


@dataclass
class RunContext:
    task_id: str
    video_path: Path
    duration: float
    output_dir: Path
    params: dict[str, str | float]
    cancel_event: threading.Event = field(default_factory=threading.Event)


class Application:
    def __init__(self, server: JsonRpcServer, manifest_path: Path) -> None:
        self.server = server
        self.manifest = load_manifest(manifest_path)
        self.initialized = False
        self.wire_style: str | None = None
        self._active: RunContext | None = None
        self._lock = threading.Lock()

    def dispatch(self, method: str, params: dict[str, object]) -> object:
        if method == "initialize":
            return self.initialize(params)
        if method == "run":
            return self.run()
        if method == "cancel":
            task_id = params.get("taskId")
            if task_id is not None and (not isinstance(task_id, str) or not task_id):
                raise RpcError(-32602, "taskId must be a nonempty string")
            with self._lock:
                ctx = self._active
                matched = ctx is not None and (
                    task_id is None or task_id == ctx.task_id
                )
                if matched:
                    ctx.cancel_event.set()
            return {"cancelled": matched}
        if method == "shutdown":
            self.dispatch("cancel", {})
            return {}
        raise RpcError(-32601, f"Method not found: {method}")

    def initialize(self, params: dict[str, object]) -> dict[str, object]:
        version = params.get("protocolVersion", PROTOCOL_VERSION)
        if not isinstance(version, str) or version.split(".")[0] != "1":
            raise RpcError(-32602, "Unsupported host protocolVersion")
        capabilities = params.get("capabilities", {})
        if not isinstance(capabilities, dict):
            raise RpcError(-32602, "Host capabilities must be an object")
        services = capabilities.get("hostServices", {})
        if not isinstance(services, dict):
            raise RpcError(-32602, "hostServices must be an object")
        wire_style = None
        # Current Rust host uses a dotted key and camelCase frame fields.
        # The documented Python service uses framesExtract and snake_case.
        for key, style in (("frames.extract", "camel"), ("framesExtract", "snake")):
            service_version = services.get(key)
            if (
                isinstance(service_version, str)
                and service_version.split(".")[0] == "1"
            ):
                wire_style = style
                break
        self.wire_style = wire_style
        self.initialized = True
        return {
            "protocolVersion": PROTOCOL_VERSION,
            "pluginName": PLUGIN_NAME,
            "version": self.manifest["version"],
            "capabilities": {
                "progressEvents": True,
                "cancellation": True,
                "videoFrames": True,
                "fileWrite": True,
            },
        }

    def prepare_run(self, raw: dict[str, object]) -> None:
        """Called by the reader before starting a worker, so cancel cannot be lost."""
        if not self.initialized:
            raise RpcError(-32001, "initialize must be called before run")
        if self.wire_style is None:
            raise RpcError(-32010, "Host does not advertise frames.extract v1")
        task_id = raw.get("taskId")
        video = raw.get("video")
        output = raw.get("outputDir")
        if not isinstance(task_id, str) or not task_id:
            raise RpcError(-32602, "taskId must be a nonempty string")
        if (
            not isinstance(video, dict)
            or not isinstance(video.get("path"), str)
            or not video["path"]
        ):
            raise RpcError(-32602, "video.path must be a nonempty string")
        if not isinstance(output, str) or not output:
            raise RpcError(-32602, "outputDir must be a nonempty string")
        duration = positive_number(video.get("duration"), "video.duration")
        parameters = parse_params(
            raw.get("params", {}), self.manifest["parametersSchema"]
        )
        video_path = Path(video["path"])
        output_dir = Path(output)
        if not video_path.is_absolute() or not output_dir.is_absolute():
            raise RpcError(-32602, "video.path and outputDir must be absolute paths")
        try:
            video_path = video_path.resolve(strict=True)
            output_dir = output_dir.resolve(strict=True)
            if not video_path.is_file() or not output_dir.is_dir():
                raise OSError(
                    "Expected an existing video file and authorized output directory"
                )
        except (OSError, ValueError) as exc:
            raise RpcError(-32602, f"Invalid run paths: {exc}") from exc
        with self._lock:
            if self._active is not None:
                raise RpcError(-32001, "A run is already active")
            self._active = RunContext(
                task_id, video_path, duration, output_dir, parameters
            )

    def _progress(
        self, ctx: RunContext, fraction: float, stage: str, marks: int = 0
    ) -> None:
        fraction = min(1.0, max(0.0, fraction))
        self.server.notify(
            "progress",
            {
                "protocolVersion": PROTOCOL_VERSION,
                "taskId": ctx.task_id,
                "progress": fraction,
                "message": stage,
                "marksSoFar": marks,
                "percent": fraction * 100,
                "stage": stage,
            },
        )

    def run(self) -> dict[str, object]:
        with self._lock:
            ctx = self._active
        if ctx is None:
            raise RpcError(-32001, "run was not prepared")
        marks: list[Mark] = []
        logs = [
            "Protocol skeleton: event detection is not implemented; no marks are generated."
        ]
        processed = 0
        fraction = 0.0
        try:
            self.server.notify(
                "log",
                {
                    "protocolVersion": PROTOCOL_VERSION,
                    "taskId": ctx.task_id,
                    "level": "info",
                    "message": logs[0],
                },
            )
            client = FrameClient(
                lambda method, params: self.server.request(
                    method, params, ctx.cancel_event
                ),
                wire_style=self.wire_style,
            )
            sample_fps = float(ctx.params["sample_fps"])
            # Stay comfortably below the host's 512-frame limit per request.
            window = min(60.0, 128.0 / sample_fps)
            start = 0.0
            while start < ctx.duration:
                if ctx.cancel_event.is_set():
                    raise Cancelled()
                end = min(ctx.duration, start + window)
                if end <= start:
                    raise RpcError(
                        -32602,
                        "Video duration is too large to advance extraction windows",
                    )
                self._progress(ctx, fraction, "extracting", len(marks))
                scratch = Path(
                    tempfile.mkdtemp(prefix="apex-frames-", dir=ctx.output_dir)
                )
                extraction_complete = False
                try:
                    frames = client.extract(
                        str(ctx.video_path),
                        time_range=(start, end),
                        scratch_dir=str(scratch),
                        mode=str(ctx.params["extract_mode"]),
                        sample_fps=sample_fps,
                    )
                    extraction_complete = True

                    def progress(
                        done: int, total: int, *, base: float, stop: float
                    ) -> None:
                        nonlocal fraction
                        fraction = (
                            base + (stop - base) * (done / total if total else 1)
                        ) / ctx.duration
                        self._progress(ctx, fraction, "analyzing", len(marks))

                    marks.extend(
                        analyze(
                            frames,
                            ctx.params,
                            cancel_event=ctx.cancel_event,
                            on_progress=partial(progress, base=start, stop=end),
                        )
                    )
                    processed += len(frames)
                finally:
                    # A cancelled/timed-out reverse RPC does NOT stop host FFmpeg.
                    # Do not delete its output while it may still be writing.
                    if (
                        extraction_complete
                        and scratch.resolve() == scratch
                        and not scratch.is_symlink()
                    ):
                        shutil.rmtree(scratch)
                    else:
                        logs.append(
                            f"Extraction scratch retained for host cleanup: {scratch}"
                        )
                start = end
                fraction = end / ctx.duration
            if ctx.cancel_event.is_set():
                raise Cancelled()
            logs.append(
                f"Processed {processed} frames; backend={ctx.params['backend']} (stub, unused)."
            )
            self._progress(ctx, 1.0, "complete", len(marks))
        except Cancelled:
            logs.append(
                "Run cancelled; returning partial marks. Host extraction may still be finishing."
            )
            self._progress(ctx, fraction, "cancelled", len(marks))
        except RpcError:
            raise
        except OSError as exc:
            raise RpcError(-32000, f"Frame scratch IO failed: {exc}") from exc
        finally:
            with self._lock:
                self._active = None
        return {
            "protocolVersion": PROTOCOL_VERSION,
            "pluginName": PLUGIN_NAME,
            "marks": marks,
            "logs": logs,
            "evidencePaths": [],
        }
