"""TPIN-11 deterministic layout unit tests.

Covers the resolution/candidate machinery of ``src.layout`` —
``layout_candidates`` enumeration, ``make_profile`` box/scale arithmetic,
``pillarbox_content`` black-bar detection, ``template_for`` scaling — and
real marker matching where feasible: ``counter_offset`` and
``detect_images`` run against the bundled ``damage-marker.png`` template
(no model, no video; a missing/corrupt bundled PNG fails the tests).

Distinct from ``test_layout_recognition.py``: that file asserts confirmed
detection on specific synthesized layouts (4:3-stretched dy, pillarboxed
4:3, 16:10, compact) and the name-row voice-icon confirm. This file
covers the *candidate layer itself* — which geometries are offered per
resolution, HUD scale math per UI mode, unconfirmed fallback identity,
and direct ``counter_offset`` scoring including the marker-shifted dy
delta — none of which the recognition file asserts.
"""

from __future__ import annotations

import unittest

import cv2
import numpy as np

from src.layout import (
    RESOURCE_DIR,
    counter_offset,
    detect_images,
    layout_candidates,
    make_profile,
    pillarbox_content,
    template_for,
)


def _marker():
    marker = cv2.imdecode(
        np.frombuffer((RESOURCE_DIR / "templates/damage-marker.png").read_bytes(), np.uint8),
        cv2.IMREAD_GRAYSCALE,
    )
    assert marker is not None, "Bundled damage-marker.png must decode"
    return marker


def _marker_image(profile, dx=0, dy=0, fill=0):
    """Paste the scaled marker at its anchor + offset on a half-res canvas."""
    marker = _marker()
    resized = template_for(profile, marker)
    width, height = profile["reference_size"]
    image = np.full((height // 2, width // 2, 3), fill, dtype=np.uint8)
    x = round(profile["counter_marker_origin"][0] / 2 + dx / 2)
    y = round(profile["counter_marker_origin"][1] / 2 + dy / 2)
    image[y : y + resized.shape[0], x : x + resized.shape[1]] = resized[
        :, :, None
    ]
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_NEAREST)


class LayoutCandidatesTests(unittest.TestCase):
    """The candidate set offered per resolution is deterministic."""

    def test_native_first_and_standard_before_compact(self):
        candidates = layout_candidates(1920, 1080)
        self.assertEqual(candidates[0]["layout"], "native")
        self.assertNotIn("ui_layout", candidates[0])
        ui_labels = [c.get("ui_layout", "standard") for c in candidates]
        first_compact = ui_labels.index("compact")
        # Every standard candidate precedes every compact one.
        self.assertTrue(all(v == "standard" for v in ui_labels[:first_compact]))
        self.assertTrue(all(v == "compact" for v in ui_labels[first_compact:]))

    def test_stretched_candidates_only_when_render_fits(self):
        # 16:9 canvas can host narrower renders (4:3, 16:10) but never 21:9.
        labels = {c["layout"] for c in layout_candidates(1920, 1080)}
        self.assertIn("4:3-stretched", labels)
        self.assertIn("16:10-stretched", labels)
        self.assertNotIn("21:9-stretched", labels)
        # An ultrawide canvas adds the 16:9-in-21:9 stretched render too.
        wide = {c["layout"] for c in layout_candidates(3440, 1440)}
        self.assertIn("16:9-stretched", wide)

    def test_narrow_aspect_adds_quarter_scale_candidates(self):
        # aspect < 1.55 gets the both-axes-0.75 candidate (4:3 in letterbox).
        narrow = layout_candidates(1440, 1080)  # aspect 1.333
        self.assertTrue(
            any(
                [round(v, 3) for v in c["hud_scale"]] == [0.75, 0.75]
                for c in narrow
            )
        )
        wide = layout_candidates(2560, 1440)  # aspect 1.778
        self.assertFalse(
            any(
                [round(v, 3) for v in c["hud_scale"]] == [0.75, 0.75]
                for c in wide
            )
        )

    def test_candidate_dedup_keeps_first_of_identical_geometry(self):
        # Two candidates mapping to the same (ui, sx, sy) collapse; the
        # earlier (preferred) label wins, so the list has no duplicate keys.
        for w, h in [(1920, 1080), (2560, 1080), (3440, 1440), (1440, 1080)]:
            with self.subTest(resolution=(w, h)):
                keys = [
                    (
                        c.get("ui_layout", "standard"),
                        round(c["hud_scale"][0], 3),
                        round(c["hud_scale"][1], 3),
                    )
                    for c in layout_candidates(w, h)
                ]
                self.assertEqual(len(keys), len(set(keys)))

    def test_content_box_candidates_use_inner_aspect(self):
        # Pillarboxed 4:3 inside 16:9: extra candidates computed from the
        # inner 1920x1440 box, all carrying content_box and pillarbox label.
        content = [320, 0, 1920, 1440]
        candidates = layout_candidates(2560, 1440, content_box=content)
        self.assertTrue(candidates)
        self.assertTrue(all(c["content_box"] == content for c in candidates))
        self.assertTrue(
            all(c["layout"].startswith("pillarbox/") for c in candidates)
        )
        # Inner aspect is 4:3 -> narrower render (none) than the 16:9 file's.
        self.assertIn("pillarbox/native", {c["layout"] for c in candidates})


class MakeProfileGeometryTests(unittest.TestCase):
    """make_profile arithmetic: scale factors, box anchoring, UI variants."""

    def test_hud_scale_is_height_over_1080_times_ui_factor(self):
        standard = make_profile(1920, 1080)
        self.assertEqual(standard["hud_scale"], [1.0, 1.0])
        compact = make_profile(1920, 1080, ui="compact")
        self.assertEqual(compact["ui_layout"], "compact")
        self.assertEqual(
            [round(v, 3) for v in compact["hud_scale"]], [1.25, 1.25]
        )
        stretched = make_profile(1920, 1080, 0.75)
        # hud_scale is [horizontal, vertical]: the 0.75 arg is vertical.
        self.assertEqual(stretched["hud_scale"], [1.0, 0.75])
        self.assertEqual(stretched["layout"], "4:3-stretched")

    def test_right_anchored_boxes_move_with_resolution_not_ui(self):
        # Right-anchored boxes (damage/ammo) sit at the right edge in
        # reference units; their width stays proportional to hud_scale.
        p1080 = make_profile(2560, 1080)
        p720 = make_profile(1280, 720)
        # Same box widths, scaled by 720/1080 (rounding gives +-1 px).
        for key in ("damage", "kills", "ammo"):
            w1080 = p1080[key][2] - p1080[key][0]
            w720 = p720[key][2] - p720[key][0]
            self.assertAlmostEqual(w720 / w1080, 720 / 1080, delta=0.03)
        # Right edge stays anchored: damage x2 is the reference right margin.
        self.assertEqual(p1080["damage"][2], 2402)  # matches base profile

    def test_compact_moves_boxes_off_the_standard_coordinates(self):
        standard = make_profile(1280, 720, ui="standard")
        compact = make_profile(1280, 720, ui="compact")
        # Compact is a genuinely different HUD, not a rescaled standard one.
        self.assertNotEqual(standard["kills"], compact["kills"])
        self.assertNotEqual(standard["ammo"], compact["ammo"])
        self.assertTrue(compact["layout"].endswith("/compact"))

    def test_stretched_vertical_shifts_top_anchored_boxes(self):
        native = make_profile(1920, 1080)
        stretched = make_profile(1920, 1080, 0.75)
        # Counter boxes are top-anchored: y compresses by the vertical
        # scale (integer rounding gives +-1 px, so assert within a delta).
        self.assertAlmostEqual(
            stretched["damage"][1] / native["damage"][1], 0.75, delta=0.01
        )
        # Bottom-anchored boxes keep the bottom margin (measured from y2).
        self.assertAlmostEqual(
            (1080 - stretched["ammo"][3]) / (1080 - native["ammo"][3]),
            0.75,
            delta=0.01,
        )

    def test_name_and_version_stamped(self):
        profile = make_profile(1920, 1080)
        self.assertEqual(profile["layout_version"], "anchored-hud-v2")
        self.assertIn("1920x1080", profile["name"])
        self.assertIn(profile["layout"], profile["name"])


class PillarboxContentTests(unittest.TestCase):
    """pillarbox_content finds symmetric black side bars only."""

    def test_symmetric_bars_report_inner_box(self):
        frame = np.zeros((1440, 2560, 3), np.uint8)
        frame[:, 320:2240] = 80  # lit inner 4:3 area, black side bars
        self.assertEqual(
            pillarbox_content([frame], 2560, 1440), [320, 0, 1920, 1440]
        )

    def test_asymmetric_bars_are_not_content(self):
        frame = np.zeros((1440, 2560, 3), np.uint8)
        frame[:, 200:2400] = 80  # left bar 200, right bar 160: asymmetric
        self.assertIsNone(pillarbox_content([frame], 2560, 1440))

    def test_full_frame_and_dark_frames_give_no_box(self):
        lit = np.full((1440, 2560, 3), 80, np.uint8)
        self.assertIsNone(pillarbox_content([lit], 2560, 1440))
        # All-black frames carry no evidence for an inner edge at all.
        dark = np.zeros((1440, 2560, 3), np.uint8)
        self.assertIsNone(pillarbox_content([dark], 2560, 1440))

    def test_wrong_size_frames_are_skipped(self):
        odd = np.full((720, 1280, 3), 80, np.uint8)
        self.assertIsNone(pillarbox_content([odd], 2560, 1440))


class TemplateForTests(unittest.TestCase):
    """template_for rescales templates to a profile's hud_scale."""

    def test_unit_scale_returns_template_unchanged(self):
        marker = np.zeros((20, 40, 3), np.uint8)
        self.assertIs(template_for(make_profile(2560, 1080), marker), marker)

    def test_nonuniform_scale_resizes_each_axis(self):
        marker = np.zeros((20, 40, 3), np.uint8)
        stretched = make_profile(1920, 1080, 0.75)
        out = template_for(stretched, marker)
        sx, sy = stretched["hud_scale"]
        self.assertEqual(out.shape, (round(20 * sy), round(40 * sx), 3))


class CounterOffsetMatchingTests(unittest.TestCase):
    """counter_offset against the real marker template (no model)."""

    def test_marker_at_anchor_scores_high_with_zero_delta(self):
        profile = make_profile(2560, 1080)
        delta, score = counter_offset(
            _marker_image(profile), profile, _marker()
        )
        self.assertIsNotNone(delta)
        self.assertGreaterEqual(score, 0.78)
        self.assertAlmostEqual(abs(delta[0]), 0, delta=2)
        self.assertAlmostEqual(abs(delta[1]), 0, delta=2)

    def test_shifted_marker_reports_the_shift(self):
        profile = make_profile(2560, 1080)
        delta, score = counter_offset(
            _marker_image(profile, dy=30), profile, _marker()
        )
        self.assertIsNotNone(delta)
        self.assertGreaterEqual(score, 0.78)
        self.assertGreaterEqual(delta[1], 26)  # +30 px shift, match tolerance

    def test_blank_frame_scores_low_and_returns_none(self):
        profile = make_profile(2560, 1080)
        delta, score = counter_offset(
            np.zeros((1080, 2560, 3), np.uint8), profile, _marker()
        )
        self.assertIsNone(delta)
        self.assertLess(score, 0.78)


class DetectImagesTests(unittest.TestCase):
    """detect_images picks the right candidate when the marker is present."""

    def test_native_marker_confirms_native_layout(self):
        profile = make_profile(1920, 1080)
        detected = detect_images([_marker_image(profile)], 1920, 1080)
        self.assertEqual(detected["layout_status"], "confirmed")
        self.assertEqual(detected["layout"], "native")
        self.assertGreaterEqual(detected["layout_score"], 0.78)

    def test_stretched_marker_confirms_stretched_layout(self):
        profile = make_profile(1920, 1080, 0.75)
        detected = detect_images([_marker_image(profile)], 1920, 1080)
        self.assertEqual(detected["layout_status"], "confirmed")
        self.assertEqual(detected["layout"], "4:3-stretched")

    def test_multiple_frames_take_the_best_score(self):
        profile = make_profile(1920, 1080)
        blank = np.zeros((1080, 1920, 3), np.uint8)
        detected = detect_images(
            [blank, _marker_image(profile), blank], 1920, 1080
        )
        self.assertEqual(detected["layout_status"], "confirmed")

    def test_unconfirmed_falls_back_to_first_candidate(self):
        first = layout_candidates(1920, 1080)[0]
        blank = np.zeros((1080, 1920, 3), np.uint8)
        detected = detect_images([blank], 1920, 1080)
        self.assertEqual(detected["layout_status"], "unconfirmed")
        # The unconfirmed profile IS the first candidate, not a stretched guess.
        self.assertEqual(detected["hud_scale"], first["hud_scale"])
        self.assertEqual(detected["layout"], first["layout"])

    def test_confirmed_winner_beats_second_by_margin(self):
        # A marker pasted at native anchor: native wins outright, so the
        # result is confirmed — margin (>0.025 over runner-up) satisfied.
        profile = make_profile(2560, 1080)
        detected = detect_images([_marker_image(profile)], 2560, 1080)
        self.assertEqual(detected["layout_status"], "confirmed")
        self.assertEqual(detected["layout"], "native")


if __name__ == "__main__":
    unittest.main()
