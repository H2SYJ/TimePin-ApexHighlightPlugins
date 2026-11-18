"""Real sidecar subprocess tests with a bidirectional mock TimePin host."""

from __future__ import annotations

import hashlib
import json
import queue
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class Host:
    def __init__(self, cwd: Path):
        self.process = subprocess.Popen(
            [sys.executable, "-u", str(ROOT / "main.py")],
            cwd=cwd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        self.messages = queue.Queue()
        self.observed = []
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()

    def _read(self):
        try:
            for line in self.process.stdout:
                self.messages.put(json.loads(line))
        except Exception as exc:
            self.messages.put(exc)
        finally:
            self.messages.put(EOFError("Sidecar stdout closed"))

    def send(self, message):
        self.process.stdin.write(json.dumps(message, ensure_ascii=False) + "\n")
        self.process.stdin.flush()

    def call(self, ident, method, params=None):
        self.send(
            {"jsonrpc": "2.0", "id": ident, "method": method, "params": params or {}}
        )

    def next(self):
        try:
            message = self.messages.get(timeout=5)
        except queue.Empty as exc:
            raise AssertionError(
                "Sidecar did not emit a message within 5 seconds"
            ) from exc
        if isinstance(message, Exception):
            raise message
        self.observed.append(message)
        return message

    def until(self, predicate):
        for _ in range(100):
            message = self.next()
            if predicate(message):
                return message
        raise AssertionError("Too many unrelated protocol messages")

    def response(self, ident):
        return self.until(lambda m: m.get("id") == ident and "method" not in m)

    def initialize(self, key="frames.extract"):
        self.call(
            1,
            "initialize",
            {
                "protocolVersion": "1.0.0",
                "capabilities": {"hostServices": {key: "1.0.0"}},
            },
        )
        return self.response(1)

    def close(self):
        if self.process.poll() is None:
            self.process.stdin.close()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # This is the exact Python subprocess created by this test.
                self.process.kill()
                self.process.wait(timeout=5)
                raise AssertionError("Sidecar failed to exit after EOF")
        if not self.process.stdin.closed:
            self.process.stdin.close()
        self.reader.join(timeout=1)
        error = self.process.stderr.read()
        self.process.stdout.close()
        self.process.stderr.close()
        if self.process.returncode != 0:
            raise AssertionError(f"Sidecar exited {self.process.returncode}: {error}")
        if error:
            raise AssertionError(f"Unexpected sidecar stderr: {error}")


class SidecarTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="时签协议测试-")
        self.root = Path(self.temp.name).resolve()
        self.video = self.root / "比赛 录像.mp4"
        self.video.write_bytes(b"original video bytes; host mock does not decode")
        self.original_hash = hashlib.sha256(self.video.read_bytes()).hexdigest()
        self.host = Host(self.root)  # The entry point must not depend on cwd.

    def tearDown(self):
        try:
            self.host.close()
            self.assertEqual(
                hashlib.sha256(self.video.read_bytes()).hexdigest(), self.original_hash
            )
        finally:
            self.temp.cleanup()

    def run_params(self, **parameters):
        return {
            "taskId": "任务-一",
            "video": {
                "id": "v",
                "path": str(self.video),
                "duration": 2.0,
                "width": 1920,
                "height": 1080,
                "fps": 60.0,
                "metadata": {},
            },
            "params": parameters,
            "outputDir": str(self.root),
        }

    def next_extraction(self):
        return self.host.until(lambda m: m.get("method") == "frames.extract")

    def answer_frames(self, request, snake=False):
        scratch = Path(request["params"]["options"]["scratch_dir"])
        self.assertTrue(scratch.is_dir())
        self.assertTrue(scratch.is_relative_to(self.root))
        frames = []
        for index, timestamp in enumerate((0.125, 0.625, 1.125)):
            frame = scratch / f"帧-{index}.png"
            frame.write_bytes(b"stub image")
            frames.append(
                {
                    "index": index,
                    "pts_sec" if snake else "ptsSec": timestamp,
                    "path": str(frame),
                }
            )
        self.host.send(
            {
                "jsonrpc": "2.0",
                "id": request["id"],
                "result": {
                    "protocolVersion": "1.0.0",
                    "mode": request["params"]["options"]["mode"],
                    "frames": frames,
                    "total_frames" if snake else "totalFrames": len(frames),
                },
            }
        )
        return scratch

    def answer_until_response(self, ident, snake=False):
        """Keep answering extract requests until the run response arrives.

        识别管线会多次调用 frames.extract（布局探测 + 分段分析）；
        mock 主机逐一应答，直到拿到 run 的响应为止。
        """
        for _ in range(50):
            message = self.host.next()
            if message.get("method") == "frames.extract":
                self.answer_frames(message, snake=snake)
                continue
            if message.get("id") == ident and "method" not in message:
                return message
        raise AssertionError("Run did not complete after 50 extract requests")

    def test_initialize_run_progress_shutdown_unicode(self):
        initialized = self.host.initialize()["result"]
        self.assertEqual(initialized["pluginName"], "apex-highlight")
        self.assertEqual(initialized["version"], "0.1.1")
        self.assertTrue(initialized["capabilities"]["cancellation"])
        self.host.call(2, "run", self.run_params(extract_mode="image2"))
        request = self.next_extraction()
        self.assertEqual(request["params"]["videoPath"], str(self.video))
        self.assertEqual(request["params"]["options"]["sample_fps"], 2)
        self.assertEqual(request["params"]["options"]["mode"], "image2")
        scratch = self.answer_frames(request)
        result = self.answer_until_response(2)["result"]
        self.assertEqual(result["marks"], [])
        self.assertEqual(result["evidencePaths"], [])
        self.assertEqual(result["protocolVersion"], "1.0.0")
        self.assertTrue(any("finalize:" in log for log in result["logs"]))
        self.assertFalse(scratch.exists())
        progress = [
            m["params"] for m in self.host.observed if m.get("method") == "progress"
        ]
        self.assertEqual(progress[-1]["progress"], 1)
        self.assertEqual(progress[-1]["stage"], "complete")
        self.assertTrue(
            all(0 <= p["progress"] <= 1 and p["taskId"] == "任务-一" for p in progress)
        )
        self.host.call(3, "shutdown")
        self.assertEqual(self.host.response(3)["result"], {})
        self.assertEqual(self.host.process.wait(timeout=5), 0)

    def test_documented_host_field_names(self):
        self.host.initialize("framesExtract")
        self.host.call(2, "run", self.run_params())
        request = self.next_extraction()
        self.assertEqual(request["params"]["video_path"], str(self.video))
        self.assertNotIn("videoPath", request["params"])
        self.answer_frames(request, snake=True)
        self.assertEqual(
            self.answer_until_response(2, snake=True)["result"]["marks"], []
        )

    def test_cancel_while_waiting_host_and_reuse_after_late_response(self):
        self.host.initialize()
        self.host.call(2, "run", self.run_params())
        request = self.next_extraction()
        self.host.call(3, "cancel", {"taskId": "任务-一"})
        seen = {}
        while not {2, 3}.issubset(seen):
            message = self.host.next()
            if "method" not in message:
                seen[message["id"]] = message
        self.assertTrue(seen[3]["result"]["cancelled"])
        self.assertTrue(
            any("cancelled" in log.lower() for log in seen[2]["result"]["logs"])
        )
        self.host.send(
            {"jsonrpc": "2.0", "id": request["id"], "result": {"late": True}}
        )
        # A stale response must not poison a later run or initialize request.
        self.host.call(4, "run", self.run_params())
        self.assertEqual(
            self.answer_until_response(4)["result"]["marks"], []
        )

    def test_wrong_task_cancel_and_duplicate_run(self):
        self.host.initialize()
        self.host.call(2, "run", self.run_params())
        request = self.next_extraction()
        self.host.call(3, "cancel", {"taskId": "different-task"})
        self.assertFalse(self.host.response(3)["result"]["cancelled"])
        self.host.call(4, "run", self.run_params())
        self.assertEqual(self.host.response(4)["error"]["code"], -32001)
        self.answer_frames(request)
        self.assertEqual(self.answer_until_response(2)["result"]["marks"], [])

    def test_host_error_propagates_without_crash(self):
        self.host.initialize()
        self.host.call(2, "run", self.run_params())
        request = self.next_extraction()
        self.host.send(
            {
                "jsonrpc": "2.0",
                "id": request["id"],
                "error": {"code": -32000, "message": "ffmpeg unavailable"},
            }
        )
        response = self.host.response(2)
        self.assertEqual(response["error"]["code"], -32000)
        self.assertIn("ffmpeg unavailable", response["error"]["message"])
        self.host.call(3, "shutdown")
        self.assertEqual(self.host.response(3)["result"], {})

    def test_invalid_params_before_frame_request(self):
        self.host.initialize()
        self.host.call(2, "run", self.run_params(sample_fps=0))
        self.assertEqual(self.host.response(2)["error"]["code"], -32602)
        self.assertFalse(
            any(m.get("method") == "frames.extract" for m in self.host.observed)
        )

    def test_shutdown_during_extraction_exits(self):
        self.host.initialize()
        self.host.call(2, "run", self.run_params())
        self.next_extraction()
        self.host.call(3, "shutdown")
        self.assertEqual(self.host.response(3)["result"], {})
        self.assertEqual(self.host.process.wait(timeout=5), 0)

    def test_bad_json_and_notifications_do_not_break_session(self):
        self.host.process.stdin.write("{not-json}\n")
        self.host.process.stdin.flush()
        self.assertEqual(self.host.next()["error"]["code"], -32700)
        self.host.send({"jsonrpc": "2.0", "method": "unknown", "params": {}})
        self.host.call(7, "unknown")
        self.assertEqual(self.host.next()["id"], 7)
        self.host.call(8, "shutdown")
        self.assertEqual(self.host.response(8)["result"], {})


if __name__ == "__main__":
    unittest.main()
