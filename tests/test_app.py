from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.app import Application
from src.config import parse_params
from src.errors import RpcError

ROOT = Path(__file__).resolve().parents[1]


class FakeServer:
    def __init__(self, frames=(), delay=None):
        self.requests = []
        self.notifications = []
        self.frames = frames
        self.delay = delay

    def request(self, method, params, cancel_event, timeout=330.0):
        self.requests.append((method, params))
        if self.delay:
            if cancel_event.wait(self.delay):
                from src.errors import Cancelled

                raise Cancelled()
        result = []
        for index, source in enumerate(self.frames):
            path = Path(params["options"]["scratch_dir"]) / f"{index:06}.png"
            path.write_bytes(b"png")
            result.append({"index": index, "ptsSec": source, "path": str(path)})
        return {
            "protocolVersion": "1.0.0",
            "frames": result,
            "totalFrames": len(result),
            "mode": params["options"]["mode"],
        }

    def notify(self, method, params):
        self.notifications.append((method, params))


class ApplicationTest(unittest.TestCase):
    def make_app(self):
        server = FakeServer()
        app = Application(server, ROOT / "plugin.json")
        result = app.initialize(
            {
                "protocolVersion": "1.0.0",
                "capabilities": {"hostServices": {"frames.extract": "1.0.0"}},
            }
        )
        self.assertEqual(result["protocolVersion"], "1.0.0")
        self.assertTrue(result["capabilities"]["cancellation"])
        return app, server

    def test_manifest_has_current_host_contract(self):
        manifest = json.loads((ROOT / "plugin.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["capabilities"], ["videoframes", "filewrite"])
        self.assertIsInstance(manifest["parametersSchema"], dict)
        self.assertIn("extract_mode", manifest["parametersSchema"])
        self.assertTrue(
            manifest["dependencies"][0].startswith("timepin-vision @ https://")
        )

    def test_weapon_detection_parameter_is_strict_boolean_and_default_on(self):
        app, _ = self.make_app()
        schema = app.manifest["parametersSchema"]
        self.assertIs(parse_params({}, schema)["enable_weapon_detection"], True)
        self.assertIs(
            parse_params({"enable_weapon_detection": False}, schema)[
                "enable_weapon_detection"
            ],
            False,
        )
        for invalid in (0, 1, "false", None, [], {}):
            with self.subTest(value=invalid), self.assertRaises(RpcError):
                parse_params({"enable_weapon_detection": invalid}, schema)

    def test_initialize_accepts_documented_framesExtract_key(self):
        app = Application(FakeServer(), ROOT / "plugin.json")
        app.initialize(
            {
                "protocolVersion": "1.0.0",
                "capabilities": {"hostServices": {"framesExtract": "1.0.0"}},
            }
        )
        self.assertEqual(app.wire_style, "snake")

    def test_run_rejects_nonexistent_chinese_video_path(self):
        app, _ = self.make_app()
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(RpcError) as cm:
                app.prepare_run(
                    {
                        "taskId": "任务一",
                        "video": {
                            "id": "v",
                            "path": str(Path(tmp) / "中文视频.mp4"),
                            "duration": 1,
                            "width": 1,
                            "height": 1,
                            "fps": 1,
                            "metadata": {},
                        },
                        "params": {},
                        "outputDir": tmp,
                    }
                )
            self.assertEqual(cm.exception.code, -32602)
            self.assertIsInstance(app.dispatch("cancel", {"taskId": "任务一"}), dict)

    def test_prepare_run_failure_preserves_cancel_flag(self):
        app, _ = self.make_app()
        bad_params = {"taskId": "t", "video": {}, "params": {}, "outputDir": "/"}
        with self.assertRaises(RpcError):
            app.prepare_run(bad_params)
        # The failed reservation must not leave a run context behind.
        result = app.dispatch("cancel", {})
        self.assertEqual(result, {"cancelled": False})

    def test_run_without_prepare_fails(self):
        app, _ = self.make_app()
        with self.assertRaises(RpcError) as cm:
            app.run()
        self.assertEqual(cm.exception.code, -32001)

    def test_run_requests_frames_and_returns_stub_marks(self):
        app, server = self.make_app()
        with tempfile.TemporaryDirectory() as tmp:
            video = Path(tmp) / "中文视频.mp4"
            video.write_bytes(b"mp4")
            server.frames = [0.5]
            app.prepare_run(
                {
                    "taskId": "t",
                    "video": {
                        "id": "v",
                        "path": str(video),
                        "duration": 1,
                        "width": 2560,
                        "height": 1080,
                        "fps": 60,
                        "metadata": {},
                    },
                    "params": {"sample_fps": 2},
                    "outputDir": tmp,
                }
            )
            result = app.run()
            self.assertEqual(result["marks"], [])
            self.assertEqual(result["protocolVersion"], "1.0.0")
            request = server.requests[0][1]
            self.assertIn("videoPath", request)
            self.assertTrue(any(n[0] == "progress" for n in server.notifications))


if __name__ == "__main__":
    unittest.main()
