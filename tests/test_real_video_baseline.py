"""Opt-in local FFmpeg checks and real AHC parity; no private video in CI.

TIMEPIN_FFMPEG_CONTRACT=1 runs a generated, non-Apex video extraction check.
TIMEPIN_RUN_REAL_VIDEO=1 requires TIMEPIN_REAL_VIDEO and TIMEPIN_AHC_ANALYSIS.
The latter invokes tests/baseline.py and fails on a missing/mismatched oracle.
Neither generated footage nor contract smoke certifies real Apex accuracy.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from baseline import FFmpegFrameService, ffmpeg_path, main, run_plugin, sha256
from src.frames import PROTOCOL_VERSION


@pytest.mark.skipif(os.environ.get("TIMEPIN_FFMPEG_CONTRACT") != "1",
                    reason="opt-in: TIMEPIN_FFMPEG_CONTRACT=1; requires FFmpeg and PyAV")
def test_generated_video_extraction_pts_and_source_integrity(tmp_path):
    video = tmp_path / "合成 fixture.mp4"
    output = tmp_path / "输出"
    output.mkdir()
    ffmpeg = ffmpeg_path()
    FFmpegFrameService._run([
        ffmpeg, "-hide_banner", "-nostdin", "-y", "-f", "lavfi", "-i",
        "testsrc2=size=640x360:rate=24", "-t", "3", "-c:v", "libx264",
        "-g", "24", "-keyint_min", "24", "-sc_threshold", "0", "-pix_fmt", "yuv420p",
        str(video),
    ])
    original_hash = sha256(video)
    service = FFmpegFrameService(ffmpeg, video, output)
    for mode, start, end, expected in [
        ("image2", 0.0, 1.5, [0.0, 0.5, 1.0]),
        ("image2", 1.0, 2.5, [1.0, 1.5, 2.0]),
        ("keyframes", 0.0, 2.5, [0.0, 1.0, 2.0]),
        ("keyframes", 1.0, 2.5, [1.0, 2.0]),
    ]:
        request = {
            "protocolVersion": PROTOCOL_VERSION, "videoPath": str(video),
            "options": {"mode": mode, "time_range": [start, end], "sample_fps": 2.0,
                        "scratch_dir": str(output / f"{mode}-{start}"),
                        "output_format": "png", "delivery": "files"},
        }
        result = service.extract(request)
        assert result["totalFrames"] == len(expected)
        assert [frame["ptsSec"] for frame in result["frames"]] == pytest.approx(expected)
        for frame in result["frames"]:
            # Unicode-safe, just like the SDK's decode_frame path.
            image = cv2.imdecode(np.frombuffer(Path(frame["path"]).read_bytes(), np.uint8),
                                 cv2.IMREAD_COLOR)
            assert image is not None and image.shape == (360, 640, 3)
            assert Path(frame["path"]).resolve().is_relative_to(output.resolve())
    assert sha256(video) == original_hash
    # A real CPU application run on both decoded-frame modes. This is still
    # synthetic transport/lifecycle coverage, not a recognition accuracy claim.
    for mode in ("image2", "keyframes"):
        plugin_output = output / f"plugin-{mode}"
        plugin_output.mkdir()
        run = run_plugin(video, plugin_output, mode=mode)
        assert run["sourceSha256"] == original_hash
        assert run["extractCalls"] >= 2
        assert run["result"]["protocolVersion"] == PROTOCOL_VERSION
        assert isinstance(run["result"]["marks"], list)


@pytest.mark.real_video
@pytest.mark.skipif(os.environ.get("TIMEPIN_RUN_REAL_VIDEO") != "1",
                    reason="opt-in: private footage and independent AHC analysis required")
def test_real_video_ahc_baseline(tmp_path):
    video = os.environ.get("TIMEPIN_REAL_VIDEO")
    oracle = os.environ.get("TIMEPIN_AHC_ANALYSIS")
    assert video and Path(video).is_file(), "Set TIMEPIN_REAL_VIDEO to the authorized recording"
    assert oracle and Path(oracle).is_file(), "Set TIMEPIN_AHC_ANALYSIS to original AHC analysis.json"
    output = Path(os.environ.get("TIMEPIN_BASELINE_OUTPUT", str(tmp_path / "baseline")))
    code = main([
        video, "--ahc-analysis", oracle, "--out", str(output),
        "--mode", os.environ.get("TIMEPIN_EXTRACT_MODE", "image2"),
        "--backend", os.environ.get("TIMEPIN_BACKEND", "cpu"),
        "--fps", os.environ.get("TIMEPIN_SAMPLE_FPS", "2"),
    ])
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert code == 0, report["comparison"]
    assert report["passed"] is True
