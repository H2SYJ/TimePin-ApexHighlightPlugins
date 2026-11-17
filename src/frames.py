"""Validated metadata-only access to the host's ``frames.extract`` service.

The host owns decoding and file creation. This client never reads image data,
imports a vision SDK, or writes to the video or scratch directory.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .errors import RpcError

PROTOCOL_VERSION = "1.0.0"
_SUPPORTED_MODES = ("image2", "keyframes")


@dataclass(frozen=True)
class Frame:
    index: int
    pts_sec: float
    path: Path


def _request_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RpcError(-32602, f"{name} must be a finite number")
    try:
        number = float(value)
    except OverflowError as exc:
        raise RpcError(-32602, f"{name} must be a finite number") from exc
    if not math.isfinite(number):
        raise RpcError(-32602, f"{name} must be a finite number")
    return number


def _invalid_result(message: str) -> RpcError:
    return RpcError(-32603, f"Invalid frames.extract result: {message}")


def _wire_field(data: dict[str, object], camel: str, snake: str) -> object:
    if camel in data and snake in data and data[camel] != data[snake]:
        raise _invalid_result(f"conflicting {camel}/{snake} fields")
    if camel in data:
        return data[camel]
    return data.get(snake)


class FrameClient:
    def __init__(
        self,
        send_request: Callable[[str, dict[str, object]], object],
        *,
        wire_style: str = "camel",
    ) -> None:
        if wire_style not in ("camel", "snake"):
            raise ValueError("wire_style must be 'camel' or 'snake'")
        if not callable(send_request):
            raise TypeError("send_request must be callable")
        self._send_request = send_request
        self._wire_style = wire_style

    def extract(
        self,
        video_path: str,
        *,
        time_range: tuple[float, float],
        scratch_dir: str,
        mode: str,
        sample_fps: float,
    ) -> list[Frame]:
        """Request file delivery and reject malformed or out-of-scratch results.

        ``wire_style`` changes only the request's video path field. Both known
        response field styles are accepted, without accepting conflicting aliases.
        Invalid arguments use JSON-RPC -32602; invalid host results use -32603.
        Host/transport exceptions, including cancellation, propagate unchanged.
        """
        for name, value in (("video_path", video_path), ("scratch_dir", scratch_dir)):
            if not isinstance(value, str) or not value or "\x00" in value:
                raise RpcError(-32602, f"{name} must be a nonempty path string")
        if mode not in _SUPPORTED_MODES:
            raise RpcError(-32602, "mode must be 'image2' or 'keyframes'")
        if not isinstance(time_range, tuple) or len(time_range) != 2:
            raise RpcError(-32602, "time_range must be a (start, end) tuple")
        start = _request_number(time_range[0], "time_range start")
        end = _request_number(time_range[1], "time_range end")
        if start < 0 or end <= start:
            raise RpcError(-32602, "time_range must satisfy 0 <= start < end")
        fps = _request_number(sample_fps, "sample_fps")
        if not 0 < fps <= 1000:
            raise RpcError(-32602, "sample_fps must satisfy 0 < sample_fps <= 1000")

        # Capture the authorized canonical boundary before calling the host.
        try:
            scratch_root = Path(scratch_dir).resolve(strict=True)
            if not scratch_root.is_dir():
                raise RpcError(-32602, "scratch_dir must be an existing directory")
        except (OSError, RuntimeError, ValueError) as exc:
            raise RpcError(-32602, "scratch_dir must be an existing directory") from exc

        video_key = "videoPath" if self._wire_style == "camel" else "video_path"
        params: dict[str, object] = {
            "protocolVersion": PROTOCOL_VERSION,
            video_key: video_path,
            "options": {
                "mode": mode,
                "time_range": [start, end],
                "sample_fps": fps,
                "output_format": "png",
                "scratch_dir": scratch_dir,
                "delivery": "files",
            },
        }
        result = self._send_request("frames.extract", params)
        if not isinstance(result, dict):
            raise _invalid_result("expected an object")
        version = result.get("protocolVersion")
        # Like the host, accept numeric minor/patch updates for major version 1.
        if (
            not isinstance(version, str)
            or re.fullmatch(r"1(?:\.[0-9]+){0,2}", version) is None
        ):
            raise _invalid_result("incompatible protocolVersion (expected major 1)")
        result_mode = result.get("mode")
        if not isinstance(result_mode, str) or result_mode != mode:
            raise _invalid_result("mode must match the requested extraction mode")
        entries = result.get("frames")
        if not isinstance(entries, list):
            raise _invalid_result("frames must be an array")
        total = _wire_field(result, "totalFrames", "total_frames")
        if type(total) is not int or total < 0 or total != len(entries):
            raise _invalid_result(
                "totalFrames must be an integer matching frames length"
            )

        frames: list[Frame] = []
        for entry in entries:
            if not isinstance(entry, dict):
                raise _invalid_result("each frame must be an object")
            index = entry.get("index")
            if type(index) is not int or index < 0:
                raise _invalid_result("frame index must be a nonnegative integer")
            pts = _wire_field(entry, "ptsSec", "pts_sec")
            if isinstance(pts, bool) or not isinstance(pts, (int, float)):
                raise _invalid_result("frame PTS must be a finite number")
            try:
                pts_sec = float(pts)
            except OverflowError as exc:
                raise _invalid_result("frame PTS must be a finite number") from exc
            if not math.isfinite(pts_sec):
                raise _invalid_result("frame PTS must be a finite number")
            raw_path = entry.get("path")
            if not isinstance(raw_path, str) or not raw_path or "\x00" in raw_path:
                raise _invalid_result("frame path must be a nonempty path string")
            try:
                path = Path(raw_path).resolve(strict=True)
                path.relative_to(scratch_root)
                if not path.is_file():
                    raise _invalid_result("frame path must name an existing file")
            except (OSError, RuntimeError, ValueError) as exc:
                raise _invalid_result(
                    "frame path must name a file inside scratch_dir"
                ) from exc
            frames.append(Frame(index=index, pts_sec=pts_sec, path=path))
        return frames
