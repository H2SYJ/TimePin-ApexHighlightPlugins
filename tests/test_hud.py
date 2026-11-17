"""TPIN-11 deterministic HUD unit tests (no real OCR, no ONNX, no video).

Every recognizer here is an injected, scripted stand-in: ``_digit_reader``
lambdas return fixed ``(text, confidence)`` pairs and ``_ammo_engine`` /
``_counter_fallback`` are ``SimpleNamespace`` stubs. Nothing in this file
loads a model, runs real OCR, or decodes video — assertions are about the
deterministic plumbing of ``src.hud``: ROI crop geometry, per-slot read
mapping, kills icon-anchor fallback branches, dtype/cancel validation and
the pure helpers (``numeric``, ``ammo_number``, ``inactive_view``,
``crop``).

Complements ``test_hud_recognition.py`` (single-frame rows, CTC codec,
agreement protocol) rather than duplicating it: here the focus is
multi-frame batch ordering, slot→read alignment, the icon-anchor numeric
path (``kills_source='icon_anchor'``) which the recognition file never
reaches, and input-shape rejection.
"""

from __future__ import annotations

import threading
import unittest
from types import SimpleNamespace

import cv2
import numpy as np

from src.hud import HudReader, ammo_number, crop, inactive_view, numeric
from src.layout import RESOURCE_DIR, make_profile, template_for


def _marker():
    marker = cv2.imdecode(
        np.frombuffer((RESOURCE_DIR / "templates/damage-marker.png").read_bytes(), np.uint8),
        cv2.IMREAD_GRAYSCALE,
    )
    assert marker is not None, "Bundled damage-marker.png must decode"
    return marker


def _frame_with_marker(profile, fill=70):
    """Paste the counter marker at its profile anchor on a half-res canvas,
    then scale back up — same synthesis as test_hud_recognition.py."""
    marker = _marker()
    resized = template_for(profile, marker)
    width, height = profile["reference_size"]
    canvas = np.full((height // 2, width // 2, 3), fill, dtype=np.uint8)
    x = round(profile["counter_marker_origin"][0] / 2)
    y = round(profile["counter_marker_origin"][1] / 2)
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized[
        :, :, None
    ]
    return cv2.resize(canvas, (width, height), interpolation=cv2.INTER_NEAREST)


def _reader(profile):
    reader = HudReader(profile, backend="cpu")
    reader._marker = _marker()
    # Icon-anchor fallback stubbed off unless a test installs its own.
    reader._counter_fallback = SimpleNamespace(kills_crop=lambda *a: None)
    return reader


class CropGeometryTests(unittest.TestCase):
    """crop() scales by frame/reference ratio then enlarges 3x."""

    def test_crop_scales_region_and_enlarges_three_times(self):
        profile = make_profile(2560, 1080)
        frame = np.zeros((1080, 2560, 3), np.uint8)
        x1, y1, x2, y2 = profile["damage"]  # e.g. [2362,94,2402,122]
        region = crop(frame, profile["damage"], profile["reference_size"])
        self.assertEqual(region.shape, ((y2 - y1) * 3, (x2 - x1) * 3, 3))

    def test_crop_maps_fractional_reference_to_frame_pixels(self):
        # Half-resolution frame: box lands at half the reference coords.
        profile = make_profile(2560, 1080)
        frame = np.zeros((540, 1280, 3), np.uint8)
        x1, y1, x2, y2 = profile["kills"]
        frame[
            round(y1 * 540 / 1080) : round(y2 * 540 / 1080),
            round(x1 * 1280 / 2560) : round(x2 * 1280 / 2560),
        ] = 200
        region = crop(frame, profile["kills"], profile["reference_size"])
        self.assertGreater(float(region.mean()), 100.0)

    def test_crop_rejects_empty_region(self):
        frame = np.zeros((100, 100, 3), np.uint8)
        with self.assertRaises(ValueError):
            crop(frame, [50, 50, 50, 60], [2560, 1080])


class NumericAndAmmoEdgeTests(unittest.TestCase):
    """Boundary cases test_hud_recognition.py does not cover."""

    def test_numeric_boundary_and_whitespace(self):
        self.assertEqual(numeric(" 60 ", 0.9, 60), 60)  # strip then accept
        self.assertIsNone(numeric("61", 0.9, 60))  # limit is inclusive
        self.assertIsNone(numeric("", 0.99, 60))
        self.assertIsNone(numeric("007a", 0.99, 60))

    def test_ammo_ring_zero_aliases_and_thresholds(self):
        # 〇 (ideographic zero) and Ø follow the O-alias branch at >=0.80.
        self.assertEqual(ammo_number("3〇", 0.85), 30)
        self.assertEqual(ammo_number("3Ø", 0.85), 30)
        # '日' replacement needs its own 0.67 floor — lower than digit path.
        self.assertEqual(ammo_number("9日", 0.67), 90)
        self.assertIsNone(ammo_number("9日", 0.66))
        # Non-alias garbage never reaches numeric().
        self.assertIsNone(ammo_number("abc", 0.99))
        self.assertIsNone(ammo_number("3日5", 0.5))

    def test_ammo_alias_limit_is_1000(self):
        self.assertIsNone(numeric("1001", 0.99, 1000))


class InactiveViewTests(unittest.TestCase):
    """inactive_view looks only at the bottom control-band strip."""

    def test_bright_bottom_band_keeps_view_active(self):
        frame = np.zeros((1080, 2560, 3), np.uint8)
        frame[round(1080 * 0.985) :, round(2560 * 0.12) : round(2560 * 0.88)] = 200
        self.assertFalse(inactive_view(frame))

    def test_band_edges_do_not_leak_into_detection(self):
        # Bright ABOVE the band must not flip the verdict: band still black.
        frame = np.zeros((1080, 2560, 3), np.uint8)
        frame[: round(1080 * 0.985), :] = 255
        self.assertTrue(inactive_view(frame))
        # Bright OUTSIDE the band's x-range must not flip it either.
        frame = np.zeros((1080, 2560, 3), np.uint8)
        frame[round(1080 * 0.985) :, : round(2560 * 0.10)] = 255
        self.assertTrue(inactive_view(frame))

    def test_partial_band_needs_over_72_percent_dark(self):
        # inactive_view returns mean(dark) > 0.72 over the bottom band.
        # Lighting 25% of the band leaves 75% dark -> still inactive;
        # lighting 30% leaves 70% dark -> no longer inactive.
        frame = np.zeros((1080, 2560, 3), np.uint8)
        band = frame[round(1080 * 0.985) :, round(2560 * 0.12) : round(2560 * 0.88)]
        band[:, : round(band.shape[1] * 0.25)] = 255
        self.assertTrue(inactive_view(frame))
        frame = np.zeros((1080, 2560, 3), np.uint8)
        band = frame[round(1080 * 0.985) :, round(2560 * 0.12) : round(2560 * 0.88)]
        band[:, : round(band.shape[1] * 0.30)] = 255
        self.assertFalse(inactive_view(frame))


class ReadBatchSlotMappingTests(unittest.TestCase):
    """read_batch must align scripted reads to the right frames/slots."""

    def test_two_frames_map_digit_reads_by_frame_and_key(self):
        profile = make_profile(2560, 1080)
        frame = _frame_with_marker(profile)
        reader = _reader(profile)
        calls = []

        def digit_reader(crops):
            calls.append(len(crops))
            # Deterministic per-crop answers: frame0 damage=100,kills=2;
            # frame1 damage=250,kills=3.
            return [("100", 0.99), ("2", 0.99), ("250", 0.99), ("3", 0.99)][
                : len(crops)
            ], 0.0

        reader._digit_reader = digit_reader
        reader._ammo_engine = SimpleNamespace(
            text_rec=lambda crops: ([("30", 0.95)] * len(crops), 0.0)
        )
        rows = reader.read_batch([frame, frame], [10.0, 10.5])
        # The batching contract: ONE digit_reader call for all counter crops.
        self.assertEqual(calls, [4])
        self.assertEqual(
            [(r["damage"], r["kills"], r["ammo"]) for r in rows],
            [(100, 2, 30), (250, 3, 30)],
        )
        self.assertEqual([r["t"] for r in rows], [10.0, 10.5])
        # All crops resolved, so no icon-anchor retry may fire.
        self.assertNotIn("kills_source", rows[0])

    def test_kills_icon_anchor_numeric_path_marks_source(self):
        """kills fixed-box miss + icon-anchor crop reading a real number."""
        profile = make_profile(2560, 1080)
        frame = _frame_with_marker(profile)
        reader = _reader(profile)
        # damage ok, kills unread by the primary agreement.
        reader._digit_reader = lambda crops: ([("150", 0.99), ("", 0.0)], 0.0)
        icon_crop = np.full((30, 40, 3), 220, np.uint8)
        reader._counter_fallback = SimpleNamespace(
            kills_crop=lambda *a: icon_crop
        )
        ammo_calls = []

        def rec(crops):
            ammo_calls.append(len(crops))
            # First call is the ammo ROI; second is the kills-icon retry.
            if len(ammo_calls) == 1:
                return [("45", 0.9)], 0.0
            return [("7", 0.9)], 0.0

        reader._ammo_engine = SimpleNamespace(text_rec=rec)
        (row,) = reader.read_batch([frame], [1.0])
        self.assertEqual(ammo_calls, [1, 1])
        self.assertEqual(row["kills"], 7)
        self.assertEqual(row["kills_source"], "icon_anchor")
        self.assertEqual(row["damage"], 150)
        self.assertEqual(row["ammo"], 45)

    def test_kills_retry_skipped_when_fallback_crop_absent(self):
        profile = make_profile(2560, 1080)
        frame = _frame_with_marker(profile)
        reader = _reader(profile)
        reader._digit_reader = lambda crops: ([("150", 0.99), ("", 0.0)], 0.0)
        # kills_crop -> None: no retry may be issued at all.
        seen = []

        def rec(crops):
            seen.append(len(crops))
            return [("45", 0.9)] * len(crops), 0.0

        reader._ammo_engine = SimpleNamespace(text_rec=rec)
        (row,) = reader.read_batch([frame], [0.0])
        self.assertEqual(seen, [1])  # only the ammo call
        self.assertIsNone(row["kills"])
        self.assertNotIn("kills_source", row)

    def test_kills_retry_alias_requires_ring_proof_and_alias_on_both(self):
        """Glyph path needs BOTH fixed-box and icon reads in ZERO_ALIASES."""
        profile = make_profile(2560, 1080)
        frame = _frame_with_marker(profile)
        reader = _reader(profile)
        # Fixed box reads 'O' (alias) but icon crop reads a real digit '4':
        # numeric() accepts '4' first, so this lands on icon_anchor, not ring.
        reader._digit_reader = lambda crops: ([("150", 0.99), ("O", 0.5)], 0.0)
        reader._counter_fallback = SimpleNamespace(
            kills_crop=lambda *a: np.full((30, 40, 3), 220, np.uint8)
        )
        reads = iter([[("45", 0.9)], [("4", 0.9)]])

        def rec(crops):
            return next(reads), 0.0

        reader._ammo_engine = SimpleNamespace(text_rec=rec)
        (row,) = reader.read_batch([frame], [0.0])
        self.assertEqual(row["kills"], 4)
        self.assertEqual(row["kills_source"], "icon_anchor")

    def test_non_bgr_frames_rejected_before_any_read(self):
        profile = make_profile(2560, 1080)
        reader = _reader(profile)
        reader._digit_reader = lambda crops: (_ for _ in ()).throw(
            AssertionError("must not be called")
        )
        gray = np.zeros((1080, 2560), np.uint8)  # no channel axis
        with self.assertRaises(ValueError):
            reader.read_batch([gray], [0.0])
        f32 = np.zeros((1080, 2560, 3), np.float32)
        with self.assertRaises(ValueError):
            reader.read_batch([f32], [0.0])

    def test_cancel_between_frames_interrupts_batch(self):
        profile = make_profile(2560, 1080)
        frame = _frame_with_marker(profile)
        cancel = threading.Event()
        reader = HudReader(profile, cancel_event=cancel)
        reader._marker = _marker()

        original = reader._prepare_frame

        def prepare_and_cancel(image):
            cancel.set()  # simulate cancel arriving after frame 0 is prepped
            return original(image)

        reader._prepare_frame = prepare_and_cancel
        with self.assertRaises(InterruptedError):
            reader.read_batch([frame, frame], [0.0, 0.5])

    def test_rows_keep_raw_text_and_score_for_event_merge(self):
        profile = make_profile(2560, 1080)
        frame = _frame_with_marker(profile)
        reader = _reader(profile)
        reader._digit_reader = lambda crops: ([("87", 0.96), ("1", 0.95)], 0.0)
        reader._ammo_engine = SimpleNamespace(
            text_rec=lambda crops: ([("2日", 0.9)], 0.0)
        )
        (row,) = reader.read_batch([frame], [4.25])
        # Raw fields survive alongside parsed values for downstream merge.
        self.assertEqual(row["damage_text"], "87")
        self.assertAlmostEqual(row["damage_score"], 0.96)
        self.assertEqual(row["ammo_text"], "2日")
        self.assertEqual(row["ammo"], 20)  # 日->0 alias applied
        # The make_profile profile carries a downed_marker config and the
        # template exists in resources, so a clean background frame scores
        # below threshold -> downed resolves to False (not None).
        self.assertFalse(row["downed"])
        self.assertTrue(hasattr(row["weapon_icon"], "__len__"))


class ReadBatchDownedInteractionTests(unittest.TestCase):
    """downed rows must keep ammo semantics (inactive is suppressed)."""

    def test_downed_frame_is_not_marked_inactive(self):
        profile = make_profile(2560, 1080)
        # A missing/corrupt bundled template is a packaging failure, not a skip.
        downed = cv2.imdecode(
            np.frombuffer((RESOURCE_DIR / "templates/knockdown-shield.png").read_bytes(), np.uint8),
            cv2.IMREAD_GRAYSCALE,
        )
        self.assertIsNotNone(downed, "Bundled knockdown-shield.png must decode")
        reader = _reader(profile)
        reader._downed = downed
        frame = _frame_with_marker(profile)
        # Paste the shield at the search region's top-left so downed_score is high.
        scaled = template_for(profile, downed)
        sx1, sy1, _, _ = [round(v / 2) for v in profile["downed_marker_search"]]
        half = cv2.resize(frame, (2560 // 2, 1080 // 2))
        half[sy1 : sy1 + scaled.shape[0], sx1 : sx1 + scaled.shape[1]] = scaled[
            :, :, None
        ]
        frame = cv2.resize(half, (2560, 1080), interpolation=cv2.INTER_NEAREST)
        reader._digit_reader = lambda crops: ([("10", 0.9), ("1", 0.9)], 0.0)
        reader._ammo_engine = SimpleNamespace(
            text_rec=lambda crops: ([("18", 0.9)], 0.0)
        )
        (row,) = reader.read_batch([frame], [0.0])
        self.assertTrue(row["downed"])
        self.assertFalse(row["inactive_view"])  # downed suppresses inactive
        self.assertGreaterEqual(
            row["downed_score"], profile["downed_marker_threshold"]
        )


if __name__ == "__main__":
    unittest.main()
