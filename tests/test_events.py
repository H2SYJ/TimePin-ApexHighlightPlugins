"""TPIN-11 deterministic event-chain unit tests (stdlib-only, no cv2/numpy).

Feeds scripted numeric HUD sequences through ``src.events`` and asserts the
emitted event kinds/fields: damage/kills counter steps -> hit/kill,
ammo drops -> shoot with weapon-switch gating, multi-kill expansion,
counter-drop rebaseline semantics, and a detect->segment round trip.

Distinct from ``test_combat_events.py``: that file holds the upstream
golden dictionaries; this file probes the *sequence semantics* — how a
counter or ammo series maps onto kinds — plus switching behavior the
goldens do not exercise (mid-stream slot/icon swaps blocking one pair
while later same-weapon pairs still shoot). The
``enable_weapon_detection`` disabled path is covered in
``test_weapon.py`` where ``annotate_events`` can be patched; this file
deliberately imports only the stdlib-safe chain (events + pipeline head)
matching the ImportBoundaryTests contract.
"""

from __future__ import annotations

import base64
import unittest

from src.events import detect_events, segments_from_events, stable_counter
from src.pipeline import finalize_marks

# Distinct packed 96x28 edge signatures: complementary bit patterns -> Dice 0.
ICON = base64.b64encode(bytes([0xAA] * 336)).decode("ascii")
OTHER = base64.b64encode(bytes([0x55] * 336)).decode("ascii")


def row(t, ammo=20, damage=0, kills=0, inactive=False, **extra):
    item = {
        "time": t,
        "ammo": ammo,
        "damage": damage,
        "kills": kills,
        "inactive_view": inactive,
        "weapon_icon": ICON,
    }
    item.update(extra)
    return item


def kinds(events):
    return [e["kind"] for e in events]


class NumericSequenceToEventKindTests(unittest.TestCase):
    """Scripted counter/ammo sequences -> expected event kinds, in order."""

    def test_damage_increment_sequence_yields_hit_then_kill_on_kills_step(self):
        rows = [
            row(0, 28, 100, 0),
            row(0.5, 28, 100, 0),
            row(1, 18, 172, 0),   # damage +72 (<=1000) confirmed -> hit
            row(1.5, 18, 172, 0),
            row(2, 18, 172, 1),   # kills +1 confirmed -> kill
            row(2.5, 18, 172, 1),
        ]
        events = detect_events(rows)
        self.assertEqual(kinds(events), ["hit", "shoot", "kill"])
        hit = events[0]
        self.assertEqual(
            (hit["before"], hit["after"], hit["evidence"]),
            (100, 172, "own_damage_counter"),
        )
        kill = events[-1]
        self.assertEqual(
            (kill["before"], kill["after"], kill["evidence"]),
            (0, 1, "own_kills_counter"),
        )
        # The shoot is anchored to the ammo drop 28->18 at t=1.
        shoot = events[1]
        self.assertEqual((shoot["before"], shoot["after"]), (28, 18))
        self.assertEqual(shoot["time"], 1)

    def test_multi_step_kill_expands_one_event_per_step(self):
        # kills 1->3 in one confirmed jump: two kill events, sequential.
        rows = [
            row(0, kills=1),
            row(0.5, kills=1),
            row(1, kills=3),
            row(1.5, kills=3),
        ]
        events = detect_events(rows)
        self.assertEqual(kinds(events), ["kill", "kill"])
        self.assertEqual(
            [(e["before"], e["after"]) for e in events], [(1, 2), (2, 3)]
        )

    def test_step_above_max_is_not_broken_into_events(self):
        # damage jump of 1200 > max_step 1000 -> rejected, no hit emitted.
        rows = [
            row(0, damage=100),
            row(0.5, damage=100),
            row(1, damage=1300),
            row(1.5, damage=1300),
        ]
        self.assertEqual(detect_events(rows), [])
        # kills jump of 4 > max_step 3 -> rejected too.
        rows = [
            row(0, kills=1),
            row(0.5, kills=1),
            row(1, kills=5),
            row(1.5, kills=5),
        ]
        self.assertEqual(detect_events(rows), [])

    def test_counter_drop_keeps_old_baseline_until_value_exceeds_it(self):
        # A counter reset (new life) is not an event; the baseline stays at
        # the pre-drop value, and only a later climb ABOVE it emits a hit.
        rows = [
            row(0, damage=500),
            row(0.5, damage=500),
            row(1, damage=0),
            row(1.5, damage=0),
            row(2, damage=45),
            row(2.5, damage=45),
        ]
        self.assertEqual(detect_events(rows), [])
        rows += [row(3, damage=560), row(3.5, damage=560)]
        events = detect_events(rows)
        self.assertEqual(kinds(events), ["hit"])
        self.assertEqual(
            (events[0]["before"], events[0]["after"], events[0]["time"]),
            (500, 560, 3),
        )

    def test_ammo_sequence_reload_increase_is_not_a_shot(self):
        # ammo 18 -> 5 (shot), 5 -> 18 (reload) must not emit a second shoot.
        rows = [
            row(0, ammo=18),
            row(0.5, ammo=5),
            row(1.0, ammo=18),
        ]
        events = detect_events(rows)
        self.assertEqual(kinds(events), ["shoot"])
        self.assertEqual((events[0]["before"], events[0]["after"]), (18, 5))

    def test_ammo_drop_boundaries_are_inclusive_one_and_fifty(self):
        self.assertEqual(
            kinds(detect_events([row(0, 21), row(0.5, 20)])), ["shoot"]
        )
        self.assertEqual(
            kinds(detect_events([row(0, 70), row(0.5, 20)])), ["shoot"]
        )
        self.assertEqual(detect_events([row(0, 71), row(0.5, 20)]), [])

    def test_unconfirmed_single_read_is_not_an_event(self):
        # A counter value seen exactly once (one vote) can never confirm,
        # so an OCR flicker producing a single-frame step is invisible.
        rows = [
            row(0, damage=100),
            row(0.5, damage=100),
            row(1, damage=250),     # single read -> no second vote
            row(1.5, damage=100),
            row(2, damage=100),
        ]
        self.assertEqual(detect_events(rows), [])
        self.assertNotIn(2, stable_counter(rows, "damage"))


class WeaponSwitchingTests(unittest.TestCase):
    """Switching mid-sequence breaks 'same weapon' so the drop is not a shot."""

    def test_slot_switch_blocks_shoot_even_with_same_icon(self):
        rows = [
            row(0, 24, active_weapon_slot=0),
            row(0.5, 18, active_weapon_slot=1),
            row(1.0, 18, active_weapon_slot=1),
        ]
        self.assertEqual(detect_events(rows), [])

    def test_weapon_id_switch_blocks_shoot(self):
        rows = [
            row(0, 24, current_weapon={"id": "r301"}),
            row(0.5, 18, current_weapon={"id": "car"}),
        ]
        self.assertEqual(detect_events(rows), [])

    def test_icon_switch_blocks_then_recovers_on_new_baseline(self):
        # Swap frame makes drop not-a-shot; subsequent same-icon drop is one.
        rows = [
            row(0, 24),
            row(0.5, 18, weapon_icon=OTHER),  # swap: not a shot
            row(1.0, 18, weapon_icon=OTHER),  # new baseline at 18
            row(1.5, 12, weapon_icon=OTHER),  # same icon, drop -> shot
        ]
        events = detect_events(rows)
        self.assertEqual(kinds(events), ["shoot"])
        self.assertEqual(events[0]["time"], 1.5)
        self.assertEqual(events[0]["icon_similarity"], 1.0)

    def test_same_slot_and_same_id_permits_shoot(self):
        rows = [
            row(0, 24, current_weapon={"id": "r301"}, active_weapon_slot=0),
            row(0.5, 18, current_weapon={"id": "r301"}, active_weapon_slot=0),
        ]
        self.assertEqual(kinds(detect_events(rows)), ["shoot"])

    def test_mixed_switch_only_blocks_the_switching_pair(self):
        # 0->0.5 same slot (shot), 0.5->1.0 slot switch (blocked),
        # 1.0->1.5 same slot again (shot) — only the middle pair is blocked.
        rows = [
            row(0, 30, active_weapon_slot=0),
            row(0.5, 24, active_weapon_slot=0),
            row(1.0, 18, active_weapon_slot=1),
            row(1.5, 12, active_weapon_slot=1),
        ]
        shoots = [e for e in detect_events(rows) if e["kind"] == "shoot"]
        self.assertEqual([s["time"] for s in shoots], [0.5, 1.5])


class SequenceIntegrationTests(unittest.TestCase):
    """End-to-end: numeric rows -> events -> segments stay consistent."""

    def test_detect_then_segment_round_trip(self):
        # First cluster: shoot + kill around t=10.5-11. Second cluster far
        # enough away that the >20s counter rule suppresses an immediate
        # hit, then a two-read climb emits hit at t=81.
        rows = [
            row(10, 30, 100, 0),
            row(10.5, 24, 160, 0),
            row(11.0, 24, 160, 1),
            row(11.5, 24, 160, 1),  # second kills=1 vote to confirm the step
            row(80, 30, 360, 1),
            row(80.5, 24, 360, 1),
            row(81, 24, 400, 1),
            row(81.5, 24, 400, 1),
        ]
        events = detect_events(rows)
        self.assertEqual(kinds(events), ["shoot", "kill", "shoot", "hit"])
        segments = segments_from_events(events, 120, gap=35)
        self.assertEqual(len(segments), 2)
        self.assertEqual(segments[0]["first_signal"], 10.5)
        self.assertEqual(segments[1]["first_signal"], 80.5)
        self.assertEqual(segments[1]["last_signal"], 81)

    def test_stable_counter_window_is_two_rows_not_time_span(self):
        # Three rows 1.0s apart with the same value: the middle one needs 2
        # votes inside +-2 index neighbours AND <=1.1s time distance.
        rows = [
            row(0, damage=50),
            row(1.0, damage=50),   # 1.0 <= 1.1 -> votes
            row(2.0, damage=50),   # 1.0 <= 1.1 -> votes
            row(4.0, damage=50),   # 2.0s away -> too far for row 1's vote
        ]
        confirmed = stable_counter(rows, "damage")
        # rows 0..2 mutually confirm; row3 alone at 4.0 lacks a second vote.
        self.assertIn(1, confirmed)
        self.assertNotIn(3, confirmed)

    def test_finalize_marks_end_to_end_without_weapon_observations(self):
        # No weapon_observation on rows -> annotate path has nothing to
        # consume; the finalize chain still emits marks + segments.
        rows = [
            row(0, 30, 100, 0),
            row(0.5, 24, 160, 0),
            row(1.0, 24, 160, 1),
        ]
        result = finalize_marks(rows, 30, 2)
        self.assertTrue(result["marks"])
        self.assertTrue(result["segments"])
        for mark in result["marks"]:
            self.assertIn("startTime", mark)
            self.assertIn("metadata", mark)


if __name__ == "__main__":
    unittest.main()
