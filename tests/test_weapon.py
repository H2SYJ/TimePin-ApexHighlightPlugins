"""TPIN-11 deterministic weapon unit tests (no real OCR, no ONNX, no video).

Covers the weapon-name -> catalog mapping and slot/highlight logic with
scripted stand-ins: ``text_reader`` lambdas return fixed
``(text, confidence)`` pairs, and frames are synthetic BGR ndarrays. No
OCR engine, ONNX session, or video decode runs anywhere in this file.

Coverage (complements the ``src.weapon`` contract, which has no dedicated
test file yet):
- ``resolve_weapon_text``: alias matching, digit repair (O->0, I->1),
  variants (RE-45 Burst, Hemlok Breach), akimbo gating to
  p2020/mozambique, confidence floors and ambiguity rejection.
- ``normalized`` / ``active_slot`` / ``highlight_score`` / signature
  helpers as pure deterministic functions.
- ``WeaponReader.read``: active slot selection by highlight and by
  name-panel background, native text -> weapon mapping, icon-only
  fallback, excluded/inactive frames, ``unconfirmed_layout``.
- ``annotate_events`` + ``context_from_observations``: shoot annotation,
  held/nearby-firing scopes, excluded rows, kill attribution never set.
- ``enable_weapon_detection`` disabled path through ``finalize_marks``
  with ``annotate_events`` patched to prove it is never invoked.
"""

from __future__ import annotations

import base64
import unittest
from copy import deepcopy
from unittest.mock import patch

import numpy as np

from src.events import detect_events
from src.layout import make_profile
from src.pipeline import finalize_marks
from src.weapon import (
    ICON_BYTES,
    WeaponReader,
    _signature,
    active_slot,
    annotate_events,
    context_from_observations,
    highlight_score,
    icon_similarity,
    load_catalog,
    normalized,
    recognize_text_views,
    resolve_weapon_text,
)

PROFILE = make_profile(2560, 1080)


def _obs(t, weapon, *, slot=0, confidence=0.95, agreement=2, ammo=None,
         ammo_score=0.9, icon=None, mask=None, excluded=False):
    """A usable WeaponReader observation row for annotation/context tests."""
    icon = icon if icon is not None else _random_icon(int(t * 997) or 1)
    return {
        "actual_pts": t,
        "time": t,
        "weapon": deepcopy(weapon),
        "weapon_id": weapon["id"],
        "weapon_name": weapon["name"],
        "weapon_method": "native_hud_text",
        "confidence": confidence,
        "slot": slot,
        "active_weapon_slot": slot,
        "name_view_agreement": agreement,
        "excluded_view": excluded,
        "inactive_view": False,
        "inactive": False,
        "ammo": ammo,
        "ammo_score": ammo_score,
        "weapon_icon": icon,
        "weapon_icon_mask": mask if mask is not None else icon,
    }


def _random_icon(seed):
    """Deterministic packed icon with a healthy edge count (not blank)."""
    rng = np.random.RandomState(seed)
    bits = np.zeros(ICON_BYTES * 8, np.uint8)
    bits[rng.choice(bits.size, size=400, replace=False)] = 1
    return base64.b64encode(np.packbits(bits).tobytes()).decode("ascii")


R301 = {"id": "r301", "name": "R301", "category": "assault", "variant": None}
CAR = {"id": "car", "name": "CAR", "category": "smg", "variant": None}


class NormalizeAndSignatureTests(unittest.TestCase):
    def test_normalized_uppercases_and_strips(self):
        self.assertEqual(normalized("R-301 Carbine"), "R301CARBINE")
        self.assertEqual(normalized("  c.a.r.  "), "CAR")
        self.assertEqual(normalized(""), "")

    def test_signature_validates_shape_and_rejects_blank(self):
        packed = np.zeros(ICON_BYTES, np.uint8)
        blank = base64.b64encode(packed.tobytes()).decode("ascii")
        with self.assertRaises(ValueError):
            _signature(blank)  # all-zero edges carry no geometry
        full = base64.b64encode(
            np.full(ICON_BYTES, 0xFF, np.uint8).tobytes()
        ).decode("ascii")
        with self.assertRaises(ValueError):
            _signature(full)  # >80% set is also unusable
        sparse = np.zeros(ICON_BYTES, np.uint8)
        sparse[:4] = 0x80  # four set bits = minimum usable geometry
        ok = base64.b64encode(sparse.tobytes()).decode("ascii")
        self.assertIsNotNone(_signature(ok))
        with self.assertRaises(ValueError):
            _signature("!!! not base64 !!!")

    def test_icon_similarity_is_safe_on_bad_input(self):
        valid = _random_icon(7)
        self.assertEqual(icon_similarity(valid, valid), 1.0)
        self.assertEqual(icon_similarity(valid, "garbage"), 0.0)
        self.assertEqual(icon_similarity("garbage", "garbage"), 0.0)


class ResolveWeaponTextTests(unittest.TestCase):
    """Weapon-name string -> catalog entry mapping."""

    def setUp(self):
        self.catalog = load_catalog()

    def test_exact_ascii_alias_resolves(self):
        self.assertEqual(resolve_weapon_text("R-301", 0.95, self.catalog)["id"], "r301")
        self.assertEqual(resolve_weapon_text("C.A.R.", 0.95, self.catalog)["id"], "car")
        self.assertEqual(resolve_weapon_text("R99", 0.95, self.catalog)["id"], "r99")

    def test_digit_repairs_o_and_i_to_0_and_1(self):
        # OCR reads 'O'/'I' for '0'/'1' inside the model-number shape [RP]xx.
        self.assertEqual(resolve_weapon_text("P2O2O", 0.95, self.catalog)["id"], "p2020")
        self.assertEqual(resolve_weapon_text("R-3O1", 0.95, self.catalog)["id"], "r301")

    def test_confidence_floor_rejects_weak_reads(self):
        self.assertIsNone(resolve_weapon_text("R-301", 0.74, self.catalog))
        self.assertIsNotNone(resolve_weapon_text("R-301", 0.75, self.catalog))

    def test_substring_alias_with_sufficient_coverage(self):
        # 'R-301' inside a longer HUD string resolves when >=70% of it is alias.
        self.assertIsNotNone(resolve_weapon_text("R-301 卡賓槍", 0.95, self.catalog))

    def test_variant_burst_and_breach_recognized(self):
        re45 = resolve_weapon_text("RE-45 Burst", 0.95, self.catalog)
        self.assertEqual((re45["id"], re45["variant"]), ("re45", "burst"))
        hemlok = resolve_weapon_text("Hemlok Breach", 0.95, self.catalog)
        self.assertEqual((hemlok["id"], hemlok["variant"]), ("hemlok", "breach"))

    def test_akimbo_only_for_p2020_and_mozambique(self):
        self.assertEqual(
            resolve_weapon_text("AKIMBO P2020", 0.95, self.catalog)["variant"],
            "akimbo",
        )
        self.assertIsNone(resolve_weapon_text("AKIMBO R-301", 0.95, self.catalog))

    def test_ambiguous_reads_rejected(self):
        # 'P20' alone is too short/similar to multiple entries -> None.
        self.assertIsNone(resolve_weapon_text("P2", 0.95, self.catalog))

    def test_short_normalized_text_never_resolves(self):
        for text in ("R", "7", "", None):
            with self.subTest(text=text):
                self.assertIsNone(resolve_weapon_text(text, 0.95, self.catalog))

    def test_result_carries_catalog_fields(self):
        match = resolve_weapon_text("HAVOC", 0.95, self.catalog)
        self.assertEqual(match["id"], "havoc")
        self.assertEqual(match["category"], "assault")
        self.assertIn("text_match_score", match)


class RecognizeTextViewsTests(unittest.TestCase):
    """Two caption views -> single label + match, or best-text + None."""

    def setUp(self):
        self.catalog = load_catalog()

    def test_agreeing_views_pick_the_stronger(self):
        label, match = recognize_text_views(
            [("R-301", 0.90), ("R-301", 0.97)], self.catalog
        )
        self.assertEqual((label["text"], label["confidence"]), ("R-301", 0.97))
        self.assertEqual(match["id"], "r301")

    def test_conflicting_named_views_yield_no_match(self):
        # Both views resolve — but to DIFFERENT weapons: no match, keep text.
        label, match = recognize_text_views(
            [("R-301", 0.95), ("C.A.R.", 0.95)], self.catalog
        )
        self.assertIsNone(match)
        self.assertIn(label["text"], {"R-301", "C.A.R."})

    def test_no_view_resolves_returns_best_text_no_match(self):
        label, match = recognize_text_views(
            [("???", 0.95), ("--", 0.80)], self.catalog
        )
        self.assertIsNone(match)
        self.assertEqual(label["text"], "???")  # higher confidence wins


class ActiveSlotAndHighlightTests(unittest.TestCase):
    def test_active_slot_picks_brightest_by_margin(self):
        self.assertEqual(active_slot([200.0, 40.0]), 0)
        self.assertEqual(active_slot([40.0, 200.0]), 1)
        self.assertIsNone(active_slot([100.0, 110.0]))  # inside 25px margin
        self.assertIsNone(active_slot([200.0]))         # needs exactly two
        self.assertIsNone(active_slot([200.0, 10.0, 5.0]))

    def test_highlight_score_needs_bright_foreground(self):
        dark = np.full((20, 60, 3), 40, np.uint8)
        self.assertEqual(highlight_score(dark), 0.0)
        # Sparse bright strokes in a dark panel -> high score.
        patch = np.full((20, 60, 3), 40, np.uint8)
        patch[2:6, 4:56] = 255
        patch[10:14, 4:56] = 255
        self.assertGreater(highlight_score(patch), 200.0)


class WeaponReaderReadTests(unittest.TestCase):
    """read() with scripted text_reader on a synthetic frame."""

    def _frame(self, fill=40):
        frame = np.full((1080, 2560, 3), fill, np.uint8)
        # Bright bottom control band so inactive_view() stays False.
        frame[
            round(1080 * 0.985) :, round(2560 * 0.12) : round(2560 * 0.88)
        ] = 200
        return frame

    def _lit_slot0(self, frame):
        """Bright sparse strokes in metric box 0 -> slot 0 highlighted."""
        x1, y1, x2, y2 = PROFILE["weapon_name_metrics"][0]
        frame[y1 + 2 : y1 + 6, x1 + 4 : x2 - 4] = 255
        frame[y1 + 10 : y1 + 14, x1 + 4 : x2 - 4] = 255

    def _reader(self, script):
        return WeaponReader(
            PROFILE,
            text_reader=lambda crops: (list(script)[: len(crops)], 0.0),
        )

    def test_native_text_resolves_active_slot_weapon(self):
        frame = self._frame()
        self._lit_slot0(frame)
        # 4 crops: slot0 [masked, raw], slot1 [masked, raw]. Slot 0 active.
        reader = self._reader(
            [("R-301", 0.95), ("R-301", 0.93), ("", 0.0), ("", 0.0)]
        )
        result = reader.read(frame, 3.0)
        self.assertEqual(result["weapon_id"], "r301")
        self.assertEqual(result["weapon_method"], "native_hud_text")
        self.assertEqual(result["slot"], 0)
        self.assertEqual(result["active_weapon_slot"], 0)
        self.assertEqual(result["name_view_agreement"], 2)
        self.assertAlmostEqual(result["confidence"], 0.95 * 1.0, places=3)

    def test_slot1_highlight_selects_the_other_name(self):
        frame = self._frame()
        x1, y1, x2, y2 = PROFILE["weapon_name_metrics"][1]
        frame[y1 + 2 : y1 + 6, x1 + 4 : x2 - 4] = 255
        frame[y1 + 10 : y1 + 14, x1 + 4 : x2 - 4] = 255
        reader = self._reader(
            [("", 0.0), ("", 0.0), ("C.A.R.", 0.95), ("C.A.R.", 0.94)]
        )
        result = reader.read(frame, 3.0)
        self.assertEqual((result["weapon_id"], result["slot"]), ("car", 1))

    def test_ambiguous_highlights_fall_back_to_panel_background(self):
        frame = self._frame()
        # Equal nonzero highlights (ambiguous) + bright slot-0 name panel:
        # active_slot gives up, then |bg0 - bg1| >= 30 picks slot 0.
        x1, y1, x2, y2 = PROFILE["weapon_name_boxes"][0]
        frame[y1:y2, x1:x2] = 200
        for x1, y1, x2, y2 in PROFILE["weapon_name_metrics"]:
            frame[y1 + 2 : y1 + 6, x1 + 4 : x2 - 4] = 255
            frame[y1 + 10 : y1 + 14, x1 + 4 : x2 - 4] = 255
        reader = self._reader(
            [("R99", 0.95), ("R99", 0.95), ("", 0.0), ("", 0.0)]
        )
        result = reader.read(frame, 3.0)
        self.assertEqual(result["slot"], 0)
        self.assertEqual(result["slot_basis"], "name_panel_background")
        self.assertEqual(result["weapon_id"], "r99")

    def test_ambiguous_everything_is_unknown_not_guessed(self):
        frame = self._frame()  # uniform 40: same highlight AND background
        reader = self._reader([("R-301", 0.95)] * 4)
        result = reader.read(frame, 3.0)
        self.assertIsNone(result["slot"])
        self.assertIsNone(result["weapon_id"])
        self.assertEqual(result["reason"], "ambiguous_active_slot")

    def test_inactive_frame_is_excluded_without_calling_ocr(self):
        frame = np.zeros((1080, 2560, 3), np.uint8)  # black -> inactive_view
        called = []

        def text_reader(crops):
            called.append(len(crops))
            return [("R-301", 0.95)] * len(crops), 0.0

        reader = WeaponReader(PROFILE, text_reader=text_reader)
        result = reader.read(frame, 3.0)
        self.assertEqual(called, [])
        self.assertTrue(result["excluded_view"])
        self.assertTrue(result["inactive_view"])
        self.assertIsNone(result["weapon_id"])
        self.assertEqual(result["reason"], "view_change_or_inactive")

    def test_excluded_row_flag_short_circuits_read(self):
        reader = self._reader([("R-301", 0.95)] * 4)
        result = reader.read(
            self._frame(), 3.0, row={"downed": False, "spectator": True}
        )
        self.assertTrue(result["excluded_view"])
        self.assertEqual(result["reason"], "view_change_or_inactive")

    def test_unconfirmed_layout_declines(self):
        profile = dict(PROFILE, layout_status="unconfirmed")
        reader = WeaponReader(
            profile, text_reader=lambda crops: ([("R-301", 0.95)] * len(crops), 0.0)
        )
        frame = self._frame()
        self._lit_slot0(frame)
        result = reader.read(frame, 3.0)
        self.assertIsNone(result["weapon_id"])
        self.assertEqual(result["reason"], "unconfirmed_layout")

    def test_ocr_result_count_mismatch_is_an_error(self):
        frame = self._frame()
        self._lit_slot0(frame)
        reader = WeaponReader(
            PROFILE, text_reader=lambda crops: ([("R-301", 0.9)], 0.0)
        )
        with self.assertRaises(ValueError):
            reader.read(frame, 3.0)

    def test_unreadable_names_leave_weapon_unknown(self):
        frame = self._frame()
        self._lit_slot0(frame)
        reader = self._reader([("???", 0.2), ("???", 0.2), ("", 0.0), ("", 0.0)])
        result = reader.read(frame, 3.0)
        # Icon fallback also fails on a blank icon -> stays unknown.
        self.assertIsNone(result["weapon_id"])
        self.assertEqual(result["weapon_method"], "unknown")
        self.assertEqual(result["reason"], "insufficient_or_conflicting_hud_evidence")


class AnnotateEventsTests(unittest.TestCase):
    """annotate_events attaches weapon fields to shoot events only when
    the observation evidence is trusted."""

    def test_trusted_observation_annotates_shoot(self):
        events = [{"time": 0.5, "kind": "shoot", "before": 28, "after": 24}]
        obs = [_obs(0.5, R301)]
        out = annotate_events(events, obs)
        self.assertEqual(out[0]["weapon"], "R301")
        self.assertEqual(out[0]["weapon_id"], "r301")
        self.assertEqual(out[0]["weapon_method"], "native_hud_text")
        self.assertGreaterEqual(out[0]["weapon_confidence"], 0.90)

    def test_excluded_or_untrusted_observation_does_not_annotate(self):
        events = [{"time": 0.5, "kind": "shoot"}]
        # Excluded view -> not usable.
        excluded = _obs(0.5, R301, excluded=True)
        out = annotate_events(deepcopy(events), [excluded])
        self.assertIsNone(out[0]["weapon_id"])
        self.assertEqual(out[0]["weapon"], "未知枪")
        # Low confidence alone (below 0.90, method not icon) -> untrusted.
        weak = _obs(0.5, R301, confidence=0.60)
        out = annotate_events(deepcopy(events), [weak])
        self.assertIsNone(out[0]["weapon_id"])

    def test_kill_event_gets_context_not_weapon_fields(self):
        events = [{"time": 0.5, "kind": "kill", "ownership": "self", "confirmed": True}]
        obs = [_obs(0.5, R301), _obs(0.4, R301)]
        out = annotate_events(events, obs)
        self.assertNotIn("weapon_id", out[0])
        self.assertIn("weapon_context", out[0])
        self.assertFalse(out[0]["weapon_context"]["kill_attribution"])

    def test_non_self_or_unconfirmed_kill_gets_unknown_context(self):
        events = [
            {"time": 0.5, "kind": "kill", "ownership": "teammate", "confirmed": True},
            {"time": 0.6, "kind": "kill", "ownership": "self", "confirmed": False},
        ]
        out = annotate_events(events, [_obs(0.5, R301)])
        for event in out:
            self.assertEqual(event["weapon_context"]["status"], "unknown")
            self.assertEqual(
                event["weapon_context"]["reason"],
                "unconfirmed_or_unsupported_event",
            )
            self.assertFalse(event["weapon_context"]["kill_attribution"])

    def test_input_events_are_copied_not_mutated(self):
        events = [{"time": 0.5, "kind": "shoot"}]
        before = deepcopy(events)
        annotate_events(events, [_obs(0.5, R301)])
        self.assertEqual(events, before)


class ContextFromObservationsTests(unittest.TestCase):
    """context_from_observations classifies nearby firing / held weapons."""

    def test_nearby_firing_scope_when_ammo_drops_on_same_weapon(self):
        icon = _random_icon(11)
        obs = [
            _obs(0.0, R301, ammo=28, icon=icon),
            _obs(0.4, R301, ammo=24, icon=icon),  # drop 4 within 0.75s
        ]
        context = context_from_observations({"time": 0.5}, obs)
        self.assertEqual(context["status"], "observed")
        self.assertEqual(context["scope"], "nearby_firing")
        self.assertEqual(context["weapon"]["id"], "r301")
        self.assertFalse(context["kill_attribution"])
        self.assertEqual(len(context["shots"]), 1)
        self.assertEqual(context["shots"][0]["ammo_drop"], 4)

    def test_held_at_event_needs_two_frames_near_the_event(self):
        icon = _random_icon(12)
        obs = [
            _obs(0.48, R301, icon=icon),
            _obs(0.55, R301, icon=icon),
        ]
        context = context_from_observations({"time": 0.5}, obs)
        self.assertEqual(context["status"], "observed")
        self.assertEqual(context["scope"], "held_at_event")
        self.assertEqual(context["weapon"]["id"], "r301")

    def test_conflicting_usable_weapons_is_unknown_not_guessed(self):
        # Two usable reads naming DIFFERENT weapons in the window -> the
        # chain breaks; result must not pick one arbitrarily.
        icon = _random_icon(13)
        obs = [
            _obs(0.45, R301, icon=icon),
            _obs(0.55, CAR, icon=icon),
        ]
        context = context_from_observations({"time": 0.5}, obs)
        self.assertEqual(context["status"], "unknown")

    def test_excluded_view_invalidates_context(self):
        obs = [_obs(0.5, R301, excluded=True), _obs(0.45, R301)]
        context = context_from_observations({"time": 0.5}, obs)
        self.assertEqual(context["status"], "unknown")
        self.assertEqual(context["reason"], "view_change_or_inactive")

    def test_empty_window_reports_unknown(self):
        context = context_from_observations({"time": 50.0}, [_obs(0.5, R301)])
        self.assertEqual(context["status"], "unknown")
        self.assertEqual(context["reason"], "insufficient_or_conflicting_hud_evidence")


class WeaponDetectionFlagTests(unittest.TestCase):
    """enable_weapon_detection=False through the real pipeline boundary."""

    def _rows(self):
        obs = _obs(0.5, R301)
        return [
            {
                "time": 0, "ammo": 30, "damage": 100, "kills": 0,
                "inactive_view": False, "weapon_icon": _random_icon(21),
                "weapon_observation": dict(obs, actual_pts=0),
            },
            {
                "time": 0.5, "ammo": 24, "damage": 160, "kills": 0,
                "inactive_view": False, "weapon_icon": _random_icon(21),
                "weapon_observation": obs,
            },
            {
                "time": 1.0, "ammo": 24, "damage": 160, "kills": 0,
                "inactive_view": False, "weapon_icon": _random_icon(21),
            },
        ]

    def test_disabled_flag_never_calls_annotate_events(self):
        with patch(
            "src.weapon.annotate_events",
            side_effect=AssertionError("must not run"),
        ) as spy:
            result = finalize_marks(
                self._rows(), 10, 2, params={"enable_weapon_detection": False}
            )
        spy.assert_not_called()
        self.assertTrue(result["marks"])  # pipeline still completed

    def test_enabled_flag_invokes_annotate_and_shoot_gets_weapon_fields(self):
        captured = {}

        def spy_annotate(events, observations, *, cancel_event=None):
            captured["events"] = events
            captured["observations"] = observations
            return annotate_events(
                events, observations, cancel_event=cancel_event
            )

        with patch("src.weapon.annotate_events", side_effect=spy_annotate):
            result = finalize_marks(
                self._rows(), 10, 2, params={"enable_weapon_detection": True}
            )
        self.assertTrue(captured["observations"])  # obs flowed in from rows
        shoot = next(e for e in captured["events"] if e["kind"] == "shoot")
        # After annotation the shoot event carries the observed weapon.
        annotated = annotate_events(
            [dict(shoot)], captured["observations"]
        )[0]
        self.assertEqual(annotated["weapon_id"], "r301")
        self.assertTrue(result["marks"])

    def test_missing_flag_defaults_to_enabled(self):
        with patch(
            "src.weapon.annotate_events", wraps=annotate_events
        ) as spy:
            finalize_marks(self._rows(), 10, 2, params={})
        spy.assert_called_once()

    def test_weapon_id_switch_blocks_shoot_upstream_of_annotation(self):
        # Same ammo drop but different weapon ids on the two rows -> the
        # shoot never forms, so annotation has nothing to attach to.
        rows = self._rows()
        rows[0]["current_weapon"] = {"id": "r301"}
        rows[1]["current_weapon"] = {"id": "car"}
        self.assertEqual(detect_events(rows), [])


if __name__ == "__main__":
    unittest.main()
