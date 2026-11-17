"""Plugin lifecycle and bounded, host-owned frame extraction."""

from __future__ import annotations

import shutil
import tempfile
import threading
from collections.abc import Sequence
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
from .extract import FrameExtractor
from .frames import FrameClient
from .pipeline import finalize_marks
from .resources import load_json

if TYPE_CHECKING:
    from .protocol import JsonRpcServer


@dataclass
class RunContext:
    task_id: str
    video_path: Path
    duration: float
    output_dir: Path
    params: dict[str, str | float | bool]
    sample_points: list[float]
    cancel_event: threading.Event = field(default_factory=threading.Event)
    profile_layout_version: str | None = None
    logs: list[str] = field(default_factory=list)
    _observed_rows: list[dict[str, object]] = field(default_factory=list)


class Application:
    def __init__(self, server: JsonRpcServer, manifest_path: Path) -> None:
        self.server = server
        self.manifest = load_manifest(manifest_path)
        self.initialized = False
        self.wire_style: str | None = None
        self._active: RunContext | None = None
        self._lock = threading.Lock()

    def _detect_layout(
        self,
        ctx: RunContext,
        extractor,
        logs: list[str],
    ) -> dict[str, object] | None:
        """Probe sample points for the HUD layout.

        Returns ``None`` when every extraction request was cancelled —
        callers then treat the whole run as cancelled. A decode/OCR
        failure propagates so the caller can decide whether layout
        failure is fatal for this backend.
        """
        scratch = Path(tempfile.mkdtemp(prefix="apex-layout-", dir=ctx.output_dir))
        extraction_complete = False
        try:
            self._progress(ctx, 0.0, "detecting layout")
            layout_start, layout_end = self._layout_window(
                ctx.duration, ctx.sample_points
            )
            try:
                layout_frames = extractor.client.extract(
                    str(ctx.video_path),
                    time_range=(layout_start, layout_end),
                    scratch_dir=str(scratch),
                    mode=extractor.mode,
                    sample_fps=max(extractor.sample_fps, 2.0),
                )
            except Cancelled:
                return None
            extraction_complete = True
            selected = self._pick_layout_frames(layout_frames, ctx.sample_points)
            if not selected:
                return None
            try:
                images = extractor.load_images(selected)
            except (OSError, ValueError) as exc:
                # 主机只交付文件路径；桩帧/损坏帧不能触发整局失败，
                # 降级为未确认布局（下游 read_batch 仍按 profile 尝试）。
                logs.append(f"Layout probe frames undecodable: {exc}")
                return None
            from .layout import confirm_by_name_row, detect_images

            height, width = images[0].shape[:2]
            detected = detect_images(images, width, height)
            if not isinstance(detected, dict):
                raise RpcError(-32603, "Layout detection returned invalid data")
            # 未确认的候选用本人名称行语音图标复核一次（与上游 detect_video 顺序一致）。
            if detected.get("layout_status") == "unconfirmed":
                proof = confirm_by_name_row(images, detected)
                if proof:
                    detected.update(proof)
                    detected["layout_status"] = "confirmed"
                    detected["layout_basis"] = "self_name_row"
            ctx.profile_layout_version = str(
                detected.get("layout_version") or "anchored-hud-v2"
            )
            logs.append(
                f"HUD layout {ctx.profile_layout_version} "
                f"({detected.get('layout_status', 'unknown')}, "
                f"score={detected.get('layout_score', 'n/a')})"
            )
            return detected
        finally:
            if (
                extraction_complete
                and scratch.resolve() == scratch
                and not scratch.is_symlink()
            ):
                shutil.rmtree(scratch)
            else:
                logs.append(f"Layout scratch retained for host cleanup: {scratch}")

    def _read_hud(
        self,
        ctx: RunContext,
        profile: dict[str, object] | None,
        images: Sequence | None,
        times: Sequence[float],
    ) -> list[dict[str, object]]:
        """Read one HUD batch and feed outcome observation inline.

        AHC 的 combat outcome 观察与帧读取同步进行（observe 依赖整帧，
        必须在 scratch 清理前完成），因此这里直接持有并推进
        OutcomeReader。read_batch 返回的行同时是后续事件的唯一输入。

        返回的 row 写入两份时间戳：``t``（宿主拼写，HUD 行原有契约）
        与 ``time``（AHC 事件/倒地/结果模块统一的秒时间戳）。
        """
        if not isinstance(images, Sequence):
            images = None  # defensive: caller may pass Frame objects instead
        reader = getattr(ctx, "_hud_reader", None)
        if reader is None and profile is not None:
            from .hud import HudReader

            reader = HudReader(
                profile,
                backend=str(ctx.params["backend"]),
                cancel_event=ctx.cancel_event,
            )
            ctx._hud_reader = reader
        if reader is not None and images is not None:
            try:
                rows = reader.read_batch(images, times)
            except (OSError, ValueError, FileNotFoundError, RuntimeError) as exc:
                # 解码/模型/后端失败时降级为时间戳骨架，保留 run 完整性。
                ctx.logs.append(f"HUD read degraded: {exc}")
                rows = [
                    {"t": t, "time": t, "inactive_view": False}
                    for t in times
                ]
        else:
            # 未确认布局（或主机只交付元数据桩帧）时保留时间戳骨架。
            rows = [
                {"t": t, "time": t, "inactive_view": False}
                for t in times
            ]
        if not isinstance(rows, list):
            raise RpcError(-32603, "HUD reader returned invalid rows")
        for row in rows:
            row["time"] = row["t"]
            ctx._observed_rows.append(row)
        if reader is not None and images is not None:
            for image, row in zip(images, rows):
                self._outcome_reader(ctx).observe(image, row)
        return rows

    def _outcome_reader(self, ctx: RunContext):
        from .outcome import OutcomeReader

        outcome = getattr(ctx, "_outcome_reader", None)
        if outcome is None:
            # combat-outcome-profile 是独立资源，布局版本由 detect_images 写入。
            outcome_profile = load_json("combat-outcome-profile.json")
            outcome_profile["layout_version"] = ctx.profile_layout_version
            outcome = OutcomeReader(
                outcome_profile,
                backend=str(ctx.params["backend"]),
                cancel_event=ctx.cancel_event,
            )
            ctx._outcome_reader = outcome
        return outcome

    def _weapon_reader(self, ctx: RunContext, profile: dict[str, object]):
        """懒加载 WeaponReader，复用已检测的 HUD layout profile。"""
        from .weapon import WeaponReader

        reader = getattr(ctx, "_weapon_reader", None)
        if reader is None:
            reader = WeaponReader(
                profile,
                backend=str(ctx.params["backend"]),
                output_dir=ctx.output_dir,
                cancel_event=ctx.cancel_event,
            )
            ctx._weapon_reader = reader
        return reader

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
        sample_points = self._sample_points(duration)
        with self._lock:
            if self._active is not None:
                raise RpcError(-32001, "A run is already active")
            self._active = RunContext(
                task_id,
                video_path,
                duration,
                output_dir,
                parameters,
                sample_points,
            )

    @staticmethod
    def _sample_points(duration: float) -> list[float]:
        """AHC ``detect_video`` sample schedule, expressed in seconds."""
        points = [
            min(5.0, duration * 0.1),
            min(90.0, duration * 0.2),
            duration * 0.25,
            duration * 0.55,
        ]
        if duration >= 600:
            points.extend((duration * 0.8, duration * 0.95))
        if duration >= 3000:
            points.extend((duration * 0.05, duration * 0.4))
        # Keep only valid, distinct ordered offsets inside the video.
        return sorted({point for point in points if 0 <= point < duration})

    @staticmethod
    def _layout_window(
        duration: float, sample_points: list[float]
    ) -> tuple[float, float]:
        """One narrow extraction covering AHC's layout probe points."""
        # Pad around the nominal times so host sampling includes them.
        return max(0.0, min(sample_points) - 0.1), min(
            duration, max(sample_points) + 0.1
        )

    @staticmethod
    def _pick_layout_frames(frames: list, sample_points: list[float]) -> list:
        """Choose one extracted frame per layout sample point."""
        selected: dict[float, object] = {}
        for point in sample_points:
            if not frames:
                break
            nearest = min(frames, key=lambda frame: abs(frame.pts_sec - point))
            if nearest not in selected.values():
                selected[point] = nearest
        return sorted(selected.values(), key=lambda frame: frame.pts_sec)

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
        logs = ctx.logs
        processed = 0
        fraction = 0.0
        try:
            client = FrameClient(
                lambda method, params: self.server.request(
                    method, params, ctx.cancel_event
                ),
                wire_style=self.wire_style,
            )
            sample_fps = float(ctx.params["sample_fps"])
            extractor = FrameExtractor(
                client,
                ctx.video_path,
                str(ctx.params["extract_mode"]),
                sample_fps,
                ctx.output_dir,
            )
            rows: list[dict[str, object]] = []

            profile_data = self._detect_layout(ctx, extractor, logs)

            start = 0.0
            window_index = 0
            while start < ctx.duration:
                if ctx.cancel_event.is_set():
                    raise Cancelled()
                end = min(ctx.duration, start + extractor.batch_seconds)
                if end <= start:
                    raise RpcError(
                        -32602,
                        "Video duration is too large to advance extraction windows",
                    )
                for batch in extractor.iter_batches(
                    (start, end),
                    on_batch=lambda done, total, *, base=start, stop=end: (
                        self._progress(
                            ctx,
                            0.02
                            + 0.78
                            * (
                                (base + (stop - base) * (done / total if total else 1))
                                / ctx.duration
                            ),
                            "analyzing",
                            len(marks),
                        )
                    ),
                ):
                    if ctx.cancel_event.is_set():
                        raise Cancelled()
                    try:
                        images = extractor.load_images(batch)
                    except (OSError, ValueError) as exc:
                        # 主机可能交付元数据桩帧（测试 mock）或损坏帧；
                        # 降级为时间戳骨架行而不让整局失败。
                        logs.append(f"Frame decode degraded: {exc}")
                        images = None
                    batch_rows = self._read_hud(
                        ctx,
                        profile_data,
                        images,
                        [frame.pts_sec for frame in batch],
                    )
                    # 武器 OCR/图标比对必须在 scratch 清理前完成，结果持久化到行。
                    if (
                        profile_data is not None
                        and ctx.params.get("enable_weapon_detection") is not False
                        and images is not None
                    ):
                        weapon_reader = self._weapon_reader(ctx, profile_data)
                        times = [f.pts_sec for f in batch]
                        weapon_observations = weapon_reader.sample(
                            images, times=times, rows=batch_rows
                        )
                        for row, obs in zip(batch_rows, weapon_observations):
                            if obs.get("weapon_id"):
                                row["current_weapon"] = {
                                    "id": obs["weapon_id"],
                                    "name": obs["weapon_name"],
                                    "category": obs["weapon"].get("category") if obs.get("weapon") else None,
                                    "variant": obs.get("weapon_variant"),
                                }
                                row["weapon"] = obs["weapon_name"]
                                row["weapon_id"] = obs["weapon_id"]
                                row["weapon_confidence"] = obs.get("confidence", 0.0)
                                row["weapon_method"] = obs.get("weapon_method")
                            row["weapon_observation"] = obs
                    rows.extend(batch_rows)
                processed = len(rows)
                start = end
                fraction = 0.02 + 0.78 * (end / ctx.duration)
                window_index += 1

            if ctx.cancel_event.is_set():
                raise Cancelled()
            # 持久化学习样本到 outputDir/learned-weapons.json（仅当有新样本且未取消）
            weapon_reader = getattr(ctx, "_weapon_reader", None)
            if weapon_reader is not None:
                weapon_reader.save_learning()
            final = finalize_marks(
                rows,
                ctx.duration,
                sample_fps,
                outcome_reader=getattr(ctx, "_outcome_reader", None),
                params=ctx.params,
                cancel_event=ctx.cancel_event,
            )
            marks.extend(final["marks"])
            logs.extend(final["logs"])
            logs.append(
                f"Processed {processed} frames; backend={ctx.params['backend']}; "
                f"marks={len(marks)}."
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
