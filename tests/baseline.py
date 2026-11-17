"""Local, opt-in video baseline: real plugin run vs AHC analysis.json.

Run ``python tests/baseline.py --help``. This is a Python test host, NOT the
Rust/Tauri host. Only this host opens the video; the plugin receives PNG files.
Nothing is uploaded, exported, re-encoded in place, or written beside the input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.app import Application  # noqa: E402
from src.frames import PROTOCOL_VERSION  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ffmpeg_path() -> str:
    configured = os.environ.get("TIMEPIN_FFMPEG") or shutil.which("ffmpeg")
    if configured:
        return configured
    try:
        import imageio_ffmpeg
    except ImportError as exc:
        raise RuntimeError("Install FFmpeg or set TIMEPIN_FFMPEG (imageio-ffmpeg also works)") from exc
    return imageio_ffmpeg.get_ffmpeg_exe()


def media_info(video: Path) -> dict:
    try:
        import av
    except ImportError as exc:
        raise RuntimeError("Install PyAV for the opt-in test host: python -m pip install av") from exc
    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        duration = (
            float(container.duration) / 1_000_000
            if container.duration is not None
            else float(stream.duration * stream.time_base)
        )
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("Video must have a finite positive duration")
        return {"duration": duration, "width": stream.width, "height": stream.height}


class FFmpegFrameService:
    """Local file-delivery adapter using real FFmpeg; not host equivalence proof.

    image2 samples decoded frames with fps/showinfo. keyframes selects sync
    packet PTS from PyAV, then seeks each with FFmpeg. Returned PTS are absolute
    video seconds, including for nonzero extraction windows. The packet index
    is cached once per run, not rebuilt for every batch.
    """

    MAX_FRAMES = 512

    def __init__(self, ffmpeg: str, video_path: Path, output_dir: Path) -> None:
        self.ffmpeg = ffmpeg
        self.video_path = video_path.resolve()
        self.output_dir = output_dir.resolve()
        self.calls: list[dict] = []
        self._sync_pts: list[float] | None = None

    def extract(self, request: dict) -> dict:
        if Path(request["videoPath"]).resolve() != self.video_path:
            raise PermissionError("videoPath is not the authorized video")
        options = request["options"]
        scratch = Path(options["scratch_dir"]).resolve()
        if not scratch.is_relative_to(self.output_dir):
            raise PermissionError("scratch_dir escaped outputDir")
        start, end = options["time_range"]
        fps = options["sample_fps"]
        if not (0 <= start < end and math.isfinite(end) and math.isfinite(fps) and fps > 0):
            raise ValueError("Invalid frame extraction range or fps")
        scratch.mkdir(parents=True, exist_ok=True)
        work = Path(tempfile.mkdtemp(prefix="local-host-", dir=scratch))
        mode = options["mode"]
        if mode == "image2":
            frames = self._image2(work, start, end, fps)
        elif mode == "keyframes":
            frames = self._keyframes(work, start, end, fps)
        else:
            raise ValueError(f"Unsupported extract mode: {mode}")
        self.calls.append(request)
        return {"protocolVersion": PROTOCOL_VERSION, "totalFrames": len(frames),
                "mode": mode, "frames": frames}

    def _image2(self, work: Path, start: float, end: float, fps: float) -> list[dict]:
        stderr = self._run([
            self.ffmpeg, "-hide_banner", "-nostdin", "-y", "-ss", f"{start:.6f}",
            "-i", str(self.video_path), "-t", f"{end - start:.6f}",
            "-vf", f"fps={fps:.6f},showinfo", "-frames:v", str(self.MAX_FRAMES),
            "-fps_mode", "passthrough", "-f", "image2", str(work / "frame_%06d.png"),
        ])
        pts = [float(match.group(1)) + start for line in stderr.splitlines()
               if "showinfo" in line
               for match in [re.search(r"pts_time:([-+0-9.eE]+)", line)] if match]
        files = sorted(work.glob("frame_*.png"))
        if len(pts) < len(files):
            raise RuntimeError("FFmpeg showinfo is missing frame timestamps")
        return [{"index": i, "ptsSec": pts[i], "path": str(path)}
                for i, path in enumerate(files) if start <= pts[i] < end]

    def _keyframe_pts(self) -> list[float]:
        if self._sync_pts is None:
            import av

            with av.open(str(self.video_path)) as container:
                stream = container.streams.video[0]
                origin = float(stream.start_time * stream.time_base) if stream.start_time else 0.0
                self._sync_pts = sorted({
                    float(packet.pts * stream.time_base) - origin
                    for packet in container.demux(stream)
                    if packet.is_keyframe and packet.pts is not None
                })
            if not self._sync_pts:
                raise RuntimeError("No sync packet timestamps available for keyframes mode")
        return self._sync_pts

    def _keyframes(self, work: Path, start: float, end: float, fps: float) -> list[dict]:
        selected = []
        for pts in self._keyframe_pts():
            if start <= pts < end and (not selected or pts - selected[-1] >= 1 / fps):
                selected.append(pts)
                if len(selected) == self.MAX_FRAMES:
                    break
        frames = []
        for index, pts in enumerate(selected):
            path = work / f"keyframe_{index:06d}.png"
            self._run([
                self.ffmpeg, "-hide_banner", "-nostdin", "-y", "-ss", f"{pts:.6f}",
                "-i", str(self.video_path), "-frames:v", "1", "-fps_mode", "passthrough",
                "-f", "image2", str(path),
            ])
            if not path.is_file():
                raise RuntimeError(f"Keyframe extraction produced no file at {pts}")
            frames.append({"index": index, "ptsSec": pts, "path": str(path)})
        return frames

    @staticmethod
    def _run(command: list[str]) -> str:
        result = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                                encoding="utf-8", errors="replace", timeout=120, check=False)
        if result.returncode:
            raise RuntimeError("FFmpeg failed: " + "\n".join(result.stderr.splitlines()[-8:]))
        return result.stderr


class LocalFrameServiceHost:
    def __init__(self, service: FFmpegFrameService) -> None:
        self.service = service
        self.notifications: list[tuple[str, dict]] = []

    def request(self, method, params, cancel_event=None, timeout=330.0):
        if method != "frames.extract":
            raise ValueError(f"Unsupported host method: {method}")
        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError("Extraction cancelled")
        return self.service.extract(params)

    def notify(self, method, params):
        self.notifications.append((method, params))


def run_plugin(video: Path, output_dir: Path, *, mode: str = "image2",
               backend: str = "cpu", fps: float = 2.0, gap: float = 35.0) -> dict:
    import onnxruntime as ort

    provider = {"cpu": "CPUExecutionProvider", "dml": "DmlExecutionProvider",
                "cuda": "CUDAExecutionProvider"}[backend]
    if provider not in ort.get_available_providers():
        raise RuntimeError(f"Requested backend {backend} requires unavailable {provider}")
    video = video.resolve(strict=True)
    output_dir = output_dir.resolve(strict=True)
    before = sha256(video)
    service = FFmpegFrameService(ffmpeg_path(), video, output_dir)
    host = LocalFrameServiceHost(service)
    app = Application(host, ROOT / "plugin.json")
    app.initialize({"protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {"hostServices": {"frames.extract": PROTOCOL_VERSION}}})
    info = media_info(video)
    app.prepare_run({
        "taskId": "tpin11-baseline", "video": {"path": str(video), **info},
        "outputDir": str(output_dir),
        # Neutralize host-only clipping so differences cannot be hidden by min/max.
        "params": {"extract_mode": mode, "backend": backend, "sample_fps": fps,
                   "gap_seconds": gap, "min_segment_seconds": 1e-9,
                   "max_segment_seconds": max(1.0, info["duration"]),
                   "enable_weapon_detection": True},
    })
    started = time.perf_counter()
    try:
        result = app.run()
    finally:
        if sha256(video) != before:
            raise RuntimeError("Source video changed during baseline run")
    progress = [p for method, p in host.notifications if method == "progress"]
    if not progress or progress[-1].get("stage") != "complete":
        raise RuntimeError("Plugin did not complete the baseline run")
    return {"result": result, "elapsedSeconds": round(time.perf_counter() - started, 3),
            "sourceSha256": before, "media": info, "extractCalls": len(service.calls),
            "parameters": {"extract_mode": mode, "backend": backend, "sample_fps": fps,
                           "gap_seconds": gap, "min_segment_seconds": 1e-9,
                           "max_segment_seconds": max(1.0, info["duration"])},
            "host": "local Python FFmpeg adapter (not Rust/Tauri)"}


def _boundary(item: dict, start_key: str, end_key: str) -> tuple[float, float]:
    start = float(item[start_key])
    end = start if item.get(end_key) is None else float(item[end_key])
    if not (math.isfinite(start) and math.isfinite(end) and 0 <= start <= end):
        raise ValueError("Invalid segment boundaries")
    return start, end


def compare_marks(marks: list[dict], segments: list[dict], tolerance: float = 0.5) -> dict:
    """Strict ALL-Mark comparison. No review/protected marks are silently dropped."""
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("Tolerance must be finite and positive")
    actual = sorted(_boundary(mark, "startTime", "endTime") for mark in marks)
    expected = sorted(_boundary(segment, "start", "end") for segment in segments)
    deltas = [{"index": index, "startError": abs(a[0] - e[0]),
               "endError": abs(a[1] - e[1]), "actual": list(a), "expected": list(e)}
              for index, (a, e) in enumerate(zip(actual, expected))]
    counts_match = len(actual) == len(expected)
    return {"passed": counts_match and all(
                d["startError"] < tolerance and d["endError"] < tolerance for d in deltas),
            "countsMatch": counts_match, "markCount": len(actual),
            "segmentCount": len(expected), "toleranceSecondsExclusive": tolerance,
            "boundaries": deltas, "unmatchedMarks": [list(x) for x in actual[len(expected):]],
            "unmatchedSegments": [list(x) for x in expected[len(actual):]]}


def validate_source(analysis: dict, video: Path) -> None:
    """Reject an oracle cached for another file or a stale version of this file."""
    source = analysis.get("source", {})
    stat = video.stat()
    if (not isinstance(source, dict) or not isinstance(source.get("path"), str)
            or Path(source["path"]).resolve() != video.resolve()
            or source.get("size") != stat.st_size or source.get("mtime_ns") != stat.st_mtime_ns):
        raise ValueError("AHC analysis source fingerprint does not match this video")
    if not isinstance(analysis.get("segments"), list):
        raise ValueError("AHC analysis must contain a segments array")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--ahc-analysis", type=Path, required=True,
                        help="analysis.json from original AHC analyze on this exact video")
    parser.add_argument("--out", type=Path, required=True, help="new, private output directory")
    parser.add_argument("--mode", choices=["image2", "keyframes"], default="image2")
    parser.add_argument("--backend", choices=["cpu", "dml", "cuda"], default="cpu")
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--gap", type=float, default=35.0)
    args = parser.parse_args(argv)
    if not math.isfinite(args.fps) or not 0 < args.fps <= 120:
        parser.error("fps must be > 0 and <= 120")
    if not math.isfinite(args.gap) or args.gap <= 0:
        parser.error("gap must be finite and positive")
    video = args.video.resolve(strict=True)
    analysis_path = args.ahc_analysis.resolve(strict=True)
    analysis = json.loads(analysis_path.read_text(encoding="utf-8-sig"))
    validate_source(analysis, video)
    parameters = analysis.get("parameters", {})
    if any(parameters.get(key) != value for key, value in
           (("gap", args.gap), ("pre", 10), ("post", 15))):
        parser.error("AHC must use matching --gap, --pre 10 and --post 15")
    if "fps" in parameters and parameters["fps"] != args.fps:
        parser.error("AHC sample fps does not match --fps")
    # Refuse overwrite; each baseline is reproducible and isolated from learned weapons.
    args.out.mkdir(parents=True, exist_ok=False)
    run = run_plugin(video, args.out, mode=args.mode, backend=args.backend,
                     fps=args.fps, gap=args.gap)
    comparison = compare_marks(run["result"]["marks"], analysis["segments"])
    degraded = [line for line in run["result"]["logs"]
                if any(word in line.lower() for word in ("degraded", "undecodable", "unconfirmed"))]
    # Empty-vs-empty cannot certify positive recognition accuracy.
    passed = comparison["passed"] and comparison["segmentCount"] > 0 and not degraded
    report = {"schemaVersion": "1.0.0", "passed": passed, "plugin": run,
              "ahcAnalysisSha256": sha256(analysis_path), "comparison": comparison,
              "recognitionWarnings": degraded,
              "notes": ["Strict all-Mark count; includes review and independent downed marks.",
                        "AHC CLI analyzes candidates; worker own-result filtering differs.",
                        "AHC sampling uses half-frame timestamps; host adapter uses frame PTS."]}
    (args.out / "marks.json").write_text(json.dumps(run["result"]["marks"], ensure_ascii=False,
                                                   indent=2), encoding="utf-8")
    (args.out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                         encoding="utf-8")
    print(json.dumps({"passed": passed, "marks": comparison["markCount"],
                      "segments": comparison["segmentCount"],
                      "recognitionWarnings": len(degraded)}))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
