"""18-case lifecycle/host-contract smoke, NOT GPU or video inference.

The host returns deliberately undecodable frame files. Recognition degrades
through the real application path; provider support is tested independently.
Real FFmpeg/footage checks live in test_real_video_baseline.py.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from unittest.mock import patch

import onnxruntime as ort
import pytest

from src.app import Application
from src.errors import RpcError
from src.frames import PROTOCOL_VERSION
from src.ocr import _explicit_session

ROOT = Path(__file__).resolve().parents[1]


class MockHost:
    """File-delivery host stub; no MP4 decoding or recognizer is substituted."""

    def __init__(self) -> None:
        self.requests: list[dict] = []
        self.notifications: list[tuple[str, dict]] = []

    def request(self, method, params, cancel_event, timeout=330.0):
        assert method == "frames.extract"
        self.requests.append(params)
        options = params["options"]
        scratch = Path(options["scratch_dir"])
        target = scratch / "frame.png"
        target.write_bytes(b"matrix metadata stub, not a PNG")
        return {
            "protocolVersion": PROTOCOL_VERSION,
            "mode": options["mode"],
            "frames": [{"index": 0, "ptsSec": options["time_range"][0], "path": str(target)}],
            "totalFrames": 1,
        }

    def notify(self, method, params):
        self.notifications.append((method, params))


def _application(tmp_path: Path) -> tuple[Application, MockHost, dict]:
    video = tmp_path / "录像.mp4"
    video.write_bytes(b"original source; never decoded by the contract stub")
    output = tmp_path / "输出"
    output.mkdir()
    host = MockHost()
    app = Application(host, ROOT / "plugin.json")
    initialized = app.initialize(
        {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {"hostServices": {"frames.extract": PROTOCOL_VERSION}},
        }
    )
    assert initialized["pluginName"] == "apex-highlight"
    return app, host, {
        "taskId": "matrix",
        "video": {"path": str(video), "duration": 4.0},
        "outputDir": str(output),
    }


@pytest.mark.parametrize("extract_mode", ["keyframes", "image2"])
@pytest.mark.parametrize("backend", ["cpu", "dml", "cuda"])
@pytest.mark.parametrize("sample_fps", [1, 2, 4])
def test_parameter_matrix(tmp_path, extract_mode, backend, sample_fps):
    app, host, raw = _application(tmp_path)
    video = Path(raw["video"]["path"])
    original_hash = hashlib.sha256(video.read_bytes()).hexdigest()
    raw["params"] = {
        "extract_mode": extract_mode,
        "backend": backend,
        "sample_fps": sample_fps,
    }
    app.prepare_run(raw)
    # These spies fail if metadata smoke ever accidentally reaches real OCR.
    with (
        patch("src.ocr.create_ocr_engine", side_effect=AssertionError("unexpected inference")),
        patch("src.ocr.create_full_ocr_engine", side_effect=AssertionError("unexpected inference")),
    ):
        result = app.run()
    assert result["protocolVersion"] == PROTOCOL_VERSION
    assert result["pluginName"] == "apex-highlight"
    assert result["marks"] == []
    assert result["evidencePaths"] == []
    assert f"backend={backend}" in " | ".join(result["logs"])
    # First request is the layout probe, which has a documented 2 Hz floor.
    assert len(host.requests) >= 2
    assert host.requests[0]["options"]["sample_fps"] == max(sample_fps, 2.0)
    for request in host.requests[1:]:
        assert request["options"]["sample_fps"] == sample_fps
    for request in host.requests:
        options = request["options"]
        assert request["videoPath"] == str(video)
        assert options["mode"] == extract_mode
        assert options["delivery"] == "files"
        assert options["output_format"] == "png"
        scratch = Path(options["scratch_dir"])
        assert scratch.is_relative_to(Path(raw["outputDir"]))
        assert not scratch.exists(), "completed host scratch should be removed"
    progress = [params for method, params in host.notifications if method == "progress"]
    assert progress[-1]["progress"] == 1.0
    assert progress[-1]["stage"] == "complete"
    assert hashlib.sha256(video.read_bytes()).hexdigest() == original_hash


@pytest.mark.parametrize("bad", [
    {"extract_mode": "mpegts"}, {"backend": "tensorrt"},
    {"sample_fps": 0}, {"sample_fps": -2},
])
def test_invalid_parameter_is_rejected_before_extraction(tmp_path, bad):
    app, host, raw = _application(tmp_path)
    raw["params"] = bad
    with pytest.raises(RpcError):
        app.prepare_run(raw)
    assert not host.requests


@pytest.mark.parametrize("backend,provider", [
    ("cpu", "CPUExecutionProvider"),
    ("dml", "DmlExecutionProvider"),
    ("cuda", "CUDAExecutionProvider"),
])
def test_backend_provider_availability_is_explicit(backend, provider):
    """CPU creates a real session; absent GPU providers must fail explicitly."""
    model = ROOT / "resources" / "models" / "ppocrv6_rec_small.onnx"
    assert model.is_file(), "bundled model is required, not an optional skip"
    if provider in ort.get_available_providers():
        session = _explicit_session(model, backend)
        assert session.get_providers()[0] == provider
    else:
        with pytest.raises(RuntimeError, match=provider):
            _explicit_session(model, backend)
