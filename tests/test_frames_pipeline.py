"""Stdlib-only contract tests for host frames and the no-recognition pipeline."""

from __future__ import annotations

from collections.abc import Iterator
from copy import deepcopy
from dataclasses import FrozenInstanceError
import math
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch
from uuid import UUID

from src.errors import Cancelled, RpcError
from src.events import Mark, event_to_mark
from src.frames import Frame, FrameClient
from src.pipeline import PROGRESS_INTERVAL, analyze


class FrameClientTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.scratch = self.root / "scratch"
        self.scratch.mkdir()
        self.frame_path = self.scratch / "extraction" / "frame.png"
        self.frame_path.parent.mkdir()
        self.frame_path.write_bytes(b"metadata test: not an image")
        self.video = self.root / "video.mp4"
        self.video.write_bytes(b"original video must not change")
        self.result: dict[str, object] = {
            "protocolVersion": "1.0.0",
            "frames": [{"index": 0, "ptsSec": 0.125, "path": str(self.frame_path)}],
            "totalFrames": 1,
            "mode": "image2",
        }
        self.request = Mock(return_value=self.result)
        self.client = FrameClient(self.request)

    def extract(self, **overrides: object) -> list[Frame]:
        arguments: dict[str, object] = {
            "time_range": (0.0, 4.0),
            "scratch_dir": str(self.scratch),
            "mode": "image2",
            "sample_fps": 2.0,
        }
        arguments.update(overrides)
        return self.client.extract(str(self.video), **arguments)

    def assert_invalid_result(self, result: object) -> None:
        self.request.return_value = result
        with self.assertRaises(RpcError) as raised:
            self.extract()
        self.assertEqual(raised.exception.code, -32603)

    def test_camel_request_contract_and_immutable_frame(self) -> None:
        before = self.video.read_bytes()
        frames = self.extract()
        self.request.assert_called_once_with(
            "frames.extract",
            {
                "protocolVersion": "1.0.0",
                "videoPath": str(self.video),
                "options": {
                    "mode": "image2",
                    "time_range": [0.0, 4.0],
                    "sample_fps": 2.0,
                    "output_format": "png",
                    "scratch_dir": str(self.scratch),
                    "delivery": "files",
                },
            },
        )
        self.assertEqual(frames, [Frame(0, 0.125, self.frame_path.resolve())])
        with self.assertRaises(FrozenInstanceError):
            frames[0].index = 3
        self.assertEqual(self.video.read_bytes(), before)

    def test_snake_request_changes_only_video_path(self) -> None:
        self.client = FrameClient(self.request, wire_style="snake")
        self.result.update(
            {
                "frames": [{"index": 0, "pts_sec": 1.25, "path": str(self.frame_path)}],
                "total_frames": self.result.pop("totalFrames"),
                "mode": "keyframes",
            }
        )
        self.assertEqual(self.extract(mode="keyframes")[0].pts_sec, 1.25)
        self.request.assert_called_once_with(
            "frames.extract",
            {
                "protocolVersion": "1.0.0",
                "video_path": str(self.video),
                "options": {
                    "mode": "keyframes",
                    "time_range": [0.0, 4.0],
                    "sample_fps": 2.0,
                    "output_format": "png",
                    "scratch_dir": str(self.scratch),
                    "delivery": "files",
                },
            },
        )

    def test_response_style_is_independent_of_request_style(self) -> None:
        self.result["frames"] = [
            {"index": 0, "pts_sec": 1, "path": str(self.frame_path)}
        ]
        self.result["total_frames"] = self.result.pop("totalFrames")
        self.assertEqual(self.extract()[0].pts_sec, 1.0)
        self.client = FrameClient(self.request, wire_style="snake")
        self.result["frames"] = [
            {"index": 0, "ptsSec": 2, "path": str(self.frame_path)}
        ]
        self.result["totalFrames"] = self.result.pop("total_frames")
        self.assertEqual(self.extract()[0].pts_sec, 2.0)

    def test_empty_results_are_valid(self) -> None:
        self.result.update(frames=[], totalFrames=0)
        self.assertEqual(self.extract(), [])

    def test_accepts_compatible_major_and_preserves_real_pts(self) -> None:
        for version in ("1", "1.2", "1.9.7"):
            with self.subTest(version=version):
                self.result["protocolVersion"] = version
                self.result["frames"] = [
                    {"index": 0, "ptsSec": -0.125, "path": str(self.frame_path)}
                ]
                self.assertEqual(self.extract()[0].pts_sec, -0.125)

    def test_rejects_bad_versions(self) -> None:
        for version in (None, 1, True, "", "0.1.0", "2.0.0", "1.bad.0", "1.0.0.extra"):
            with self.subTest(version=version):
                result = dict(self.result, protocolVersion=version)
                self.assert_invalid_result(result)

    def test_rejects_malformed_result_shapes(self) -> None:
        for result in (None, [], "result", 1):
            with self.subTest(result=result):
                self.assert_invalid_result(result)
        for frames in (None, {}, (), [None], [[]]):
            with self.subTest(frames=frames):
                self.assert_invalid_result(dict(self.result, frames=frames))
        for field in ("protocolVersion", "frames", "totalFrames", "mode"):
            with self.subTest(missing=field):
                result = dict(self.result)
                del result[field]
                self.assert_invalid_result(result)

    def test_rejects_bad_count_and_mode(self) -> None:
        for count in (None, True, 1.0, "1", -1, 0, 2):
            with self.subTest(count=count):
                self.assert_invalid_result(dict(self.result, totalFrames=count))
        for mode in (None, True, "unknown", "keyframes", []):
            with self.subTest(mode=mode):
                self.assert_invalid_result(dict(self.result, mode=mode))

    def test_rejects_bad_frame_fields(self) -> None:
        cases = {
            "index": (None, True, -1, 0.0, "0"),
            "ptsSec": (None, True, "0.1", math.nan, math.inf, -math.inf, 10**400),
            "path": (None, True, "", "bad\x00path", self.frame_path),
        }
        for field, values in cases.items():
            for value in values:
                with self.subTest(field=field, value=value):
                    frame = {"index": 0, "ptsSec": 0.125, "path": str(self.frame_path)}
                    frame[field] = value
                    self.assert_invalid_result(dict(self.result, frames=[frame]))
        for missing in ("index", "ptsSec", "path"):
            with self.subTest(missing=missing):
                frame = {"index": 0, "ptsSec": 0.125, "path": str(self.frame_path)}
                del frame[missing]
                self.assert_invalid_result(dict(self.result, frames=[frame]))

    def test_rejects_conflicting_wire_aliases(self) -> None:
        self.assert_invalid_result(dict(self.result, total_frames=2))
        frame = {
            "index": 0,
            "ptsSec": 0.125,
            "pts_sec": 0.25,
            "path": str(self.frame_path),
        }
        self.assert_invalid_result(dict(self.result, frames=[frame]))

    def test_rejects_missing_directory_outside_and_traversal_paths(self) -> None:
        outside = self.root / "outside.png"
        outside.write_bytes(b"outside")
        sibling = self.root / "scratch-other"
        sibling.mkdir()
        sibling_file = sibling / "frame.png"
        sibling_file.write_bytes(b"sibling prefix must not authorize")
        paths = (
            self.scratch / "missing.png",
            self.scratch,
            outside,
            sibling_file,
            self.scratch / ".." / outside.name,
        )
        for path in paths:
            with self.subTest(path=path):
                frame = {"index": 0, "ptsSec": 0.125, "path": str(path)}
                self.assert_invalid_result(dict(self.result, frames=[frame]))

    def test_resolves_safe_dot_segments(self) -> None:
        frame = {
            "index": 0,
            "ptsSec": 0.125,
            "path": str(self.frame_path.parent / ".." / "extraction" / "frame.png"),
        }
        self.request.return_value = dict(self.result, frames=[frame])
        self.assertEqual(self.extract()[0].path, self.frame_path.resolve())

    def test_rejects_symlink_escape(self) -> None:
        outside = self.root / "outside.png"
        outside.write_bytes(b"outside")
        link = self.scratch / "escape.png"
        try:
            link.symlink_to(outside)
        except (OSError, NotImplementedError) as exc:
            self.skipTest(f"symlink creation unavailable: {exc}")
        frame = {"index": 0, "ptsSec": 0.125, "path": str(link)}
        self.assert_invalid_result(dict(self.result, frames=[frame]))

    def test_rejects_invalid_request_without_sending(self) -> None:
        cases = (
            {"time_range": None},
            {"time_range": (0,)},
            {"time_range": (-1, 1)},
            {"time_range": (1, 1)},
            {"time_range": (2, 1)},
            {"time_range": (True, 1)},
            {"time_range": (0, math.inf)},
            {"time_range": (math.nan, 1)},
            {"mode": "unsupported"},
            {"mode": []},
            {"sample_fps": True},
            {"sample_fps": 0},
            {"sample_fps": -1},
            {"sample_fps": math.inf},
            {"sample_fps": math.nan},
            {"sample_fps": 1001},
            {"sample_fps": 10**400},
            {"scratch_dir": ""},
            {"scratch_dir": str(self.scratch / "missing")},
            {"scratch_dir": str(self.frame_path)},
        )
        for options in cases:
            with self.subTest(options=options):
                with self.assertRaises(RpcError) as raised:
                    self.extract(**options)
                self.assertEqual(raised.exception.code, -32602)
        self.request.assert_not_called()

    def test_rejects_invalid_constructor_and_video_path(self) -> None:
        with self.assertRaises(ValueError):
            FrameClient(self.request, wire_style="other")
        with self.assertRaises(TypeError):
            FrameClient(None)
        for video in (None, "", "bad\x00path"):
            with self.subTest(video=video), self.assertRaises(RpcError):
                self.client.extract(
                    video,
                    time_range=(0, 1),
                    scratch_dir=str(self.scratch),
                    mode="image2",
                    sample_fps=1,
                )
        self.request.assert_not_called()

    def test_transport_errors_propagate_unchanged(self) -> None:
        for error in (RpcError(-32000, "host refused"), Cancelled()):
            with self.subTest(error=error):
                self.request.side_effect = error
                with self.assertRaises(type(error)) as raised:
                    self.extract()
                self.assertIs(raised.exception, error)


class PipelineTests(unittest.TestCase):
    def frames(self, count: int) -> list[Frame]:
        # These deliberately do not exist: the skeleton must never open images.
        return [
            Frame(index, index / 2.0, Path(f"missing-{index}.png"))
            for index in range(count)
        ]

    def test_progress_every_32_and_final_without_duplicates(self) -> None:
        self.assertEqual(PROGRESS_INTERVAL, 32)
        for count, expected in (
            (0, [(0, 0)]),
            (1, [(1, 1)]),
            (31, [(31, 31)]),
            (32, [(32, 32)]),
            (33, [(32, 33), (33, 33)]),
            (64, [(32, 64), (64, 64)]),
            (65, [(32, 65), (64, 65), (65, 65)]),
        ):
            with self.subTest(count=count):
                progress: list[tuple[int, int]] = []
                result = analyze(
                    self.frames(count),
                    {},
                    cancel_event=threading.Event(),
                    on_progress=lambda done, total: progress.append((done, total)),
                )
                self.assertEqual(result, [])
                self.assertEqual(progress, expected)

    def test_no_recognition_no_io_no_synthetic_marks(self) -> None:
        params = {
            "template": "missing.png",
            "model": "missing.onnx",
            "fake_event": True,
        }
        before = deepcopy(params)
        with (
            patch("pathlib.Path.open", side_effect=AssertionError("no image I/O")),
            patch("builtins.open", side_effect=AssertionError("no media I/O")),
            patch("src.events.event_to_mark") as converter,
        ):
            self.assertEqual(
                analyze(self.frames(3), params, cancel_event=threading.Event()), []
            )
            converter.assert_not_called()
        self.assertEqual(params, before)

    def test_pre_cancel_including_empty_input(self) -> None:
        event = threading.Event()
        event.set()
        callback = Mock()
        for count in (0, 1, 33):
            with self.subTest(count=count), self.assertRaises(Cancelled):
                analyze(
                    self.frames(count), {}, cancel_event=event, on_progress=callback
                )
        callback.assert_not_called()

    def test_checks_cancellation_on_each_frame_not_only_at_progress(self) -> None:
        event = threading.Event()
        visited: list[int] = []

        class CancellingFrames(list[Frame]):
            def __iter__(self) -> Iterator[Frame]:
                for frame in super().__iter__():
                    visited.append(frame.index)
                    if frame.index == 1:
                        event.set()
                    yield frame

        callback = Mock()
        with self.assertRaises(Cancelled):
            analyze(
                CancellingFrames(self.frames(10)),
                {},
                cancel_event=event,
                on_progress=callback,
            )
        self.assertEqual(visited, [0, 1])
        callback.assert_not_called()

    def test_cancellation_from_progress_and_final_callback(self) -> None:
        for count in (0, 1, 32, 33, 64):
            with self.subTest(count=count):
                event = threading.Event()
                progress: list[tuple[int, int]] = []

                def cancel(done: int, total: int) -> None:
                    progress.append((done, total))
                    event.set()

                with self.assertRaises(Cancelled):
                    analyze(
                        self.frames(count), {}, cancel_event=event, on_progress=cancel
                    )
                self.assertEqual(len(progress), 1)


class EventTests(unittest.TestCase):
    def test_point_mark_exact_host_fields_and_uuid(self) -> None:
        mark = event_to_mark(0, "击杀")
        self.assertEqual(
            set(mark),
            {
                "id",
                "startTime",
                "endTime",
                "label",
                "color",
                "confidence",
                "sourcePlugin",
                "thumbnailPath",
                "metadata",
            },
        )
        self.assertEqual(set(Mark.__required_keys__), set(mark))
        self.assertEqual(Mark.__optional_keys__, frozenset())
        self.assertEqual(UUID(mark["id"]).version, 4)
        self.assertEqual(mark["startTime"], 0.0)
        self.assertIsNone(mark["endTime"])
        self.assertEqual(mark["label"], "击杀")
        self.assertIsNone(mark["color"])
        self.assertIsNone(mark["confidence"])
        self.assertEqual(mark["sourcePlugin"], "apex-highlight")
        self.assertIsNone(mark["thumbnailPath"])
        self.assertEqual(mark["metadata"], {})
        self.assertNotEqual(mark["id"], event_to_mark(0, "击杀")["id"])

    def test_range_mark_and_independent_metadata(self) -> None:
        metadata = {"rule": "template", "evidence": "frame.png"}
        mark = event_to_mark(
            1.25, "Event", end_time=2.5, confidence=0.75, metadata=metadata
        )
        self.assertEqual(mark["startTime"], 1.25)
        self.assertEqual(mark["endTime"], 2.5)
        self.assertEqual(mark["confidence"], 0.75)
        self.assertEqual(mark["metadata"], metadata)
        metadata["rule"] = "changed"
        self.assertEqual(mark["metadata"]["rule"], "template")
        mark["metadata"]["new"] = "value"
        self.assertNotIn("new", metadata)
        for confidence in (0, 1):
            self.assertEqual(
                event_to_mark(1, "boundary", end_time=1, confidence=confidence)[
                    "endTime"
                ],
                1.0,
            )

    def test_rejects_invalid_start_end_confidence_and_label(self) -> None:
        invalid_numbers = (True, "1", math.nan, math.inf, -math.inf, 10**400)
        for start in (-1, *invalid_numbers):
            with self.subTest(start=start), self.assertRaises(ValueError):
                event_to_mark(start, "Event")
        for end in (-1, 0.5, *invalid_numbers):
            with self.subTest(end=end), self.assertRaises(ValueError):
                event_to_mark(1, "Event", end_time=end)
        for confidence in (-0.01, 1.01, *invalid_numbers):
            with self.subTest(confidence=confidence), self.assertRaises(ValueError):
                event_to_mark(0, "Event", confidence=confidence)
        for label in (None, 1, True):
            with self.subTest(label=label), self.assertRaises(ValueError):
                event_to_mark(0, label)

    def test_rejects_non_string_metadata(self) -> None:
        for metadata in ([], "text", {"count": 1}, {1: "value"}, {"key": None}):
            with self.subTest(metadata=metadata), self.assertRaises(ValueError):
                event_to_mark(0, "Event", metadata=metadata)
        first = event_to_mark(0, "First")
        second = event_to_mark(0, "Second")
        first["metadata"]["not-shared"] = "yes"
        self.assertEqual(second["metadata"], {})


if __name__ == "__main__":
    unittest.main()
