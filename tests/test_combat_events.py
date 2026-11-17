"""Stdlib-only AHC event, segmentation and downed lifecycle regression tests.

Golden outputs below were produced by AST-extracting only the pure functions
from the original apex_clipper.py (not importing its app/media stack).
Upstream file SHA256:
ec08974cef2db455b378494593319a59b4d353a565116928b12be3d997efa918
No F: drive, SDK, OCR/model dependency or source recording is needed on CI.
"""

from __future__ import annotations

import base64
import subprocess
import sys
import unittest
from copy import deepcopy
from pathlib import Path

from src.downed import downed_intervals
from src.events import (
    classify_combat_candidates,
    detect_events,
    event_to_mark,
    icon_similarity,
    segments_from_events,
    stable_counter,
)

ICON = base64.b64encode(bytes([0b10101010] * 336)).decode("ascii")
OTHER = base64.b64encode(bytes([0b01010101] * 336)).decode("ascii")


def row(t, ammo=20, damage=0, kills=0, inactive=False, **extra):
    return {
        "time": t,
        "ammo": ammo,
        "damage": damage,
        "kills": kills,
        "inactive_view": inactive,
        "weapon_icon": ICON,
        **extra,
    }


GOLDEN_ROWS = {
    "mid_round": [
        row(0, 28, 900, 4),
        row(0.5, 24, 900, 4),
        row(1, 20, 950, 4),
        row(1.5, 20, 950, 4),
    ],
    "new_round": [
        row(0, 20, 150, 1),
        row(0.5, 20, 150, 1),
        row(5, None, None, None, True, round_ended=True),
        row(30, 20, None, None),
        row(30.5, 15, 150, 1),
        row(31, 10, 150, 1),
        row(31.5, 10, 150, 1),
    ],
    "jump_recovery": [
        row(0, 20, 100, 1),
        row(0.5, 20, 100, 1),
        row(1, 20, 5000, 50),
        row(1.5, 20, 5000, 50),
        row(2, 20, 150, 2),
        row(2.5, 20, 150, 2),
    ],
    "first_visible": [
        row(0, 20, None, None),
        row(15, 20, None, None),
        row(15.5, 16, 100, 1),
        row(16, 12, 100, 1),
    ],
    "multi_kill": [
        row(0, 20, 0, 1),
        row(0.5, 20, 0, 1),
        row(1, 20, 0, 3),
        row(1.5, 20, 0, 3),
    ],
}

# Full upstream output dictionaries, including same-timestamp ordering and all
# evidence/index fields. Deliberately not regenerated from the port under test.
GOLDEN_EVENTS = {
    "mid_round": [
        {
            "time": 0.5,
            "kind": "shoot",
            "before": 28,
            "after": 24,
            "evidence": "ammo_decrease",
            "same_weapon": True,
            "icon_similarity": 1.0,
            "index": 1,
        },
        {
            "time": 1,
            "kind": "hit",
            "before": 900,
            "after": 950,
            "evidence": "own_damage_counter",
        },
        {
            "time": 1,
            "kind": "shoot",
            "before": 24,
            "after": 20,
            "evidence": "ammo_decrease",
            "same_weapon": True,
            "icon_similarity": 1.0,
            "index": 2,
        },
    ],
    "new_round": [
        {
            "time": 30.5,
            "kind": "shoot",
            "before": 20,
            "after": 15,
            "evidence": "ammo_decrease",
            "same_weapon": True,
            "icon_similarity": 1.0,
            "index": 4,
        },
        {
            "time": 30.5,
            "kind": "hit",
            "before": 0,
            "after": 150,
            "evidence": "new_round_counter",
        },
        {
            "time": 30.5,
            "kind": "kill",
            "before": 0,
            "after": 1,
            "evidence": "new_round_counter",
        },
        {
            "time": 31,
            "kind": "shoot",
            "before": 15,
            "after": 10,
            "evidence": "ammo_decrease",
            "same_weapon": True,
            "icon_similarity": 1.0,
            "index": 5,
        },
    ],
    "jump_recovery": [
        {
            "time": 2,
            "kind": "hit",
            "before": 100,
            "after": 150,
            "evidence": "own_damage_counter",
        },
        {
            "time": 2,
            "kind": "kill",
            "before": 1,
            "after": 2,
            "evidence": "own_kills_counter",
        },
    ],
    "first_visible": [
        {
            "time": 15.5,
            "kind": "shoot",
            "before": 20,
            "after": 16,
            "evidence": "ammo_decrease",
            "same_weapon": True,
            "icon_similarity": 1.0,
            "index": 2,
        },
        {
            "time": 15.5,
            "kind": "hit",
            "before": None,
            "after": 100,
            "evidence": "own_counter_became_visible",
        },
        {
            "time": 15.5,
            "kind": "kill",
            "before": None,
            "after": 1,
            "evidence": "own_counter_became_visible",
        },
        {
            "time": 16,
            "kind": "shoot",
            "before": 16,
            "after": 12,
            "evidence": "ammo_decrease",
            "same_weapon": True,
            "icon_similarity": 1.0,
            "index": 3,
        },
    ],
    "multi_kill": [
        {
            "time": 1,
            "kind": "kill",
            "before": 1,
            "after": 2,
            "evidence": "own_kills_counter",
        },
        {
            "time": 1,
            "kind": "kill",
            "before": 2,
            "after": 3,
            "evidence": "own_kills_counter",
        },
    ],
}


class CounterAndShootTests(unittest.TestCase):
    def test_upstream_counter_and_event_golden(self):
        for name, rows in GOLDEN_ROWS.items():
            with self.subTest(name=name):
                self.assertEqual(detect_events(rows), GOLDEN_EVENTS[name])

    def test_one_frame_misread_does_not_confirm_or_poison_baseline(self):
        rows = [
            row(0, damage=330),
            row(0.5, damage=330),
            row(1, damage=880),
            row(1.5, damage=330),
            row(2, damage=330),
            row(2.5, damage=347),
            row(3, damage=347),
        ]
        self.assertNotIn(2, stable_counter(rows, "damage"))
        self.assertEqual(
            detect_events(rows),
            [
                {
                    "time": 2.5,
                    "kind": "hit",
                    "before": 330,
                    "after": 347,
                    "evidence": "own_damage_counter",
                },
            ],
        )

    def test_stable_counter_requires_local_time_and_row_neighbours(self):
        self.assertEqual(stable_counter([row(0), row(1.1)], "damage"), {0: 0, 1: 0})
        self.assertEqual(stable_counter([row(0), row(1.1001)], "damage"), {})
        rows = [
            row(0, damage=10),
            row(0.1, damage=11),
            row(0.2, damage=12),
            row(0.3, damage=10),
        ]
        self.assertEqual(stable_counter(rows, "damage"), {})

    def test_stable_counter_preserves_upstream_inactive_vote_semantics(self):
        # AHC excludes inactive candidate rows, but its neighbouring counter
        # votes are intentionally not filtered a second time.
        rows = [row(0, inactive=True), row(0.5)]
        self.assertEqual(stable_counter(rows, "damage"), {1: 0})
        self.assertEqual(
            detect_events([row(0, 28, inactive=True), row(0.5, 20, inactive=True)]), []
        )

    def test_inventory_gap_is_not_a_new_round(self):
        rows = deepcopy(GOLDEN_ROWS["new_round"])
        rows[2].pop("round_ended")
        self.assertFalse(any(e["kind"] in {"hit", "kill"} for e in detect_events(rows)))

    def test_new_round_without_shooting_is_not_credited(self):
        rows = deepcopy(GOLDEN_ROWS["new_round"])
        for item in rows:
            item["ammo"] = 20
        self.assertEqual(detect_events(rows), [])

    def test_same_round_decrease_does_not_reset_counter(self):
        rows = [
            row(0, damage=100),
            row(0.5, damage=100),
            row(1, damage=0),
            row(1.5, damage=0),
            row(2, damage=80),
            row(2.5, damage=80),
            row(3, damage=110),
            row(3.5, damage=110),
        ]
        self.assertEqual(
            detect_events(rows),
            [
                {
                    "time": 3,
                    "kind": "hit",
                    "before": 100,
                    "after": 110,
                    "evidence": "own_damage_counter",
                },
            ],
        )

    def test_initial_positive_counter_needs_15_seconds_and_shoot_support(self):
        rows = [
            row(0, damage=None, kills=None),
            row(10, 20, None, None),
            row(10.5, 16, 50, 1),
            row(11, 12, 50, 1),
        ]
        self.assertEqual([e["kind"] for e in detect_events(rows)], ["shoot", "shoot"])
        for item in rows:
            item["ammo"] = 20
        rows[1]["time"], rows[2]["time"], rows[3]["time"] = 15, 15.5, 16
        self.assertEqual(detect_events(rows), [])

    def test_weapon_swap_guard_checks_icon_weapon_id_and_slot(self):
        cases = [
            [row(0, 28), row(0.5, 20, weapon_icon=OTHER), row(1, 12, weapon_icon=ICON)],
            [
                row(0, 28, current_weapon={"id": "r301"}),
                row(0.5, 20, current_weapon={"id": "car"}),
            ],
            [row(0, 28, active_weapon_slot=0), row(0.5, 20, active_weapon_slot=1)],
        ]
        for rows in cases:
            with self.subTest(rows=rows):
                self.assertEqual(detect_events(rows), [])

    def test_hit_feedback_can_confirm_a_weapon_swap_candidate(self):
        rows = [
            row(0, 28),
            row(0.5, 28),
            row(1, 20, damage=60, weapon_icon=OTHER),
            row(1.5, 20, damage=60, weapon_icon=OTHER),
        ]
        events = detect_events(rows)
        self.assertEqual([e["kind"] for e in events], ["hit", "shoot"])
        self.assertFalse(events[1]["same_weapon"])
        self.assertEqual(events[1]["icon_similarity"], 0.0)

    def test_inactive_frame_resets_ammo_and_gaps_are_not_shots(self):
        self.assertEqual(
            detect_events([row(0, 28), row(0.5, 24, inactive=True), row(1, 20)]), []
        )
        self.assertEqual(detect_events([row(0, 28), row(1.2, 20)]), [])
        self.assertEqual(detect_events([row(0, 80), row(0.5, 20)]), [])
        self.assertEqual(detect_events([row(0, 20), row(0.5, 28)]), [])

    def test_explicit_ignore_weapon_allows_ammo_only_rows(self):
        rows = [{"t": 0, "ammo": 28}, {"t": 0.5, "ammo": 20}]
        self.assertEqual(
            [e["kind"] for e in detect_events(rows, ignore_weapon=True)], ["shoot"]
        )
        with self.assertRaises(KeyError):
            detect_events(rows)

    def test_packed_icon_dice_matches_upstream_bit_counts(self):
        def packed(value):
            return base64.b64encode(bytes([value])).decode("ascii")

        self.assertEqual(icon_similarity(ICON, ICON), 1)
        self.assertEqual(icon_similarity(ICON, OTHER), 0)
        self.assertEqual(icon_similarity(packed(0), packed(0)), 0)
        self.assertEqual(icon_similarity(packed(0b1111), packed(0b1100)), 4 / 6)
        with self.assertRaises(ValueError):
            icon_similarity(packed(1), "")

    def test_host_aliases_match_upstream_and_inputs_are_not_mutated(self):
        original = deepcopy(GOLDEN_ROWS["mid_round"])
        host = []
        for item in original:
            converted = dict(item)
            converted["t"] = converted.pop("time")
            converted["inactive"] = converted.pop("inactive_view")
            converted["weapon"] = "r301"
            converted["weapon_slot"] = 0
            host.append(converted)
        before = deepcopy(host)
        self.assertEqual(detect_events(host), GOLDEN_EVENTS["mid_round"])
        self.assertEqual(
            stable_counter(host, "damage"), stable_counter(original, "damage")
        )
        self.assertEqual(host, before)
        self.assertEqual(GOLDEN_ROWS["mid_round"], original)
        host[1]["weapon_slot"] = 1
        for item in host:
            item["damage"] = 0
            item["kills"] = 0
        self.assertFalse(any(e["kind"] == "shoot" for e in detect_events(host)))
        host[1]["weapon_slot"] = 0
        host[1]["weapon"] = "car"
        self.assertFalse(any(e["kind"] == "shoot" for e in detect_events(host)))


class SegmentAndClassificationTests(unittest.TestCase):
    def test_global_60_second_window_boundaries_are_not_segment_boundaries(self):
        events = [{"time": t} for t in [59.5, 60, 89, 118, 119.5, 120.5, 200]]
        self.assertEqual(
            segments_from_events(events, 240),
            [
                {
                    "start": 49.5,
                    "end": 135.5,
                    "first_signal": 59.5,
                    "last_signal": 120.5,
                },
                {"start": 190, "end": 215, "first_signal": 200, "last_signal": 200},
            ],
        )

    def test_long_pull_has_no_duration_cap(self):
        events = [{"time": t} for t in [100, 145, 200, 270, 310, 430]]
        result = segments_from_events(events, 500, gap=75, pre=25, post=40)
        self.assertEqual(
            [(s["start"], s["end"]) for s in result], [(75, 350), (405, 470)]
        )

    def test_padding_overlap_is_merged_and_half_open_duration_filters_signals(self):
        events = [{"t": t} for t in [100, -1, 40, 20, 20]]
        before = deepcopy(events)
        self.assertEqual(
            segments_from_events(events, 100, gap=5),
            [
                {"start": 10, "end": 55, "first_signal": 20, "last_signal": 40},
            ],
        )
        self.assertEqual(events, before)
        self.assertEqual(segments_from_events([], 0), [])

    def test_invalid_parameters_and_protected_intervals_fail(self):
        for key in ["duration", "gap", "pre", "post"]:
            for value in [-1, float("inf"), float("nan")]:
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    arguments = dict(duration=100, gap=35, pre=10, post=15)
                    arguments[key] = value
                    segments_from_events([], **arguments)
        for start, end in [(4, 4), (-1, 5), (0, 101), (float("nan"), 5)]:
            with self.subTest(start=start, end=end), self.assertRaises(ValueError):
                segments_from_events(
                    [],
                    100,
                    protected_intervals=[
                        {"start": start, "end": end, "reason": "downed_recovery_tail"},
                    ],
                )

    def test_protected_intervals_without_events_are_kept_and_reasons_deduplicated(self):
        intervals = [
            {
                "start": 20,
                "end": 40,
                "last_downed": 25,
                "reason": "downed_recovery_tail",
            },
            {
                "start": 30,
                "end": 60,
                "last_downed": 50,
                "reason": "downed_until_terminal",
            },
            {
                "start": 35,
                "end": 55,
                "last_downed": 45,
                "reason": "downed_until_terminal",
            },
        ]
        before = deepcopy(intervals)
        self.assertEqual(
            segments_from_events([], 100, protected_intervals=intervals),
            [
                {
                    "start": 10,
                    "end": 60,
                    "first_signal": 20,
                    "last_signal": 50,
                    "protected_reasons": [
                        "downed_recovery_tail",
                        "downed_until_terminal",
                    ],
                },
            ],
        )
        self.assertEqual(intervals, before)

    def test_classification_retains_review_and_all_upstream_evidence_reasons(self):
        events = [
            {"time": 10, "kind": "hit", "before": 0, "after": 60},
            {"time": 11, "kind": "shoot"},
            {"time": 12, "kind": "hit", "before": 60, "after": 77},
            {"time": 13, "kind": "kill", "before": 0, "after": 1},
            {"time": 40, "kind": "shoot"},
            {"time": 45, "kind": "hit", "before": 77, "after": 94},
        ]
        segments = [
            {"start": 0, "end": 30, "protected_reasons": ["downed_until_terminal"]},
            {"start": 30, "end": 60},
        ]
        before = deepcopy((events, segments))
        confirmed, review = classify_combat_candidates(events, segments)
        self.assertEqual(
            confirmed,
            [
                {
                    "start": 0,
                    "end": 30,
                    "protected_reasons": ["downed_until_terminal"],
                    "classification": "confirmed",
                    "combat_evidence": {
                        "shoot_samples": 1,
                        "hit_updates": 2,
                        "repeated_hit_window": 12,
                        "accepted_reasons": [
                            "own_kill",
                            "downed_continuation",
                            "repeated_hit_feedback",
                            "strong_hit_with_shot",
                        ],
                    },
                }
            ],
        )
        self.assertEqual(
            review,
            [
                {
                    "start": 30,
                    "end": 60,
                    "classification": "review",
                    "combat_evidence": {
                        "shoot_samples": 1,
                        "hit_updates": 1,
                        "repeated_hit_window": 12,
                        "accepted_reasons": [],
                    },
                    "review_reason": "只有开枪或孤立命中，缺少连续命中/击杀/倒地延续证据",
                }
            ],
        )
        self.assertEqual((events, segments), before)

    def test_repeated_hits_respect_window_and_segment_half_open_end(self):
        events = [
            {"t": 10, "kind": "hit", "before": 0, "after": 17},
            {"t": 22, "kind": "hit", "before": 17, "after": 34},
        ]
        confirmed, review = classify_combat_candidates(
            events, [{"start": 10, "end": 23}]
        )
        self.assertEqual(len(confirmed), 1)
        self.assertEqual(review, [])
        self.assertEqual(
            classify_combat_candidates(events, [{"start": 10, "end": 22}])[0], []
        )
        self.assertEqual(
            classify_combat_candidates(
                events, [{"start": 10, "end": 23}], hit_window=11
            )[0],
            [],
        )
        events[1]["t"] = 10
        self.assertEqual(
            classify_combat_candidates(events, [{"start": 0, "end": 23}])[0], []
        )


class DownedLifecycleTests(unittest.TestCase):
    def test_upstream_downed_and_protected_segment_golden(self):
        rows = [
            {
                "time": t / 2,
                "downed": 464 <= t / 2 <= 481 or 488 <= t / 2 <= 501,
                "inactive_view": t / 2 >= 502,
            }
            for t in range(920, 1010)
        ]
        protected = downed_intervals(rows, 548)
        self.assertEqual(
            protected,
            [
                {
                    "start": 464,
                    "end": 504,
                    "last_downed": 501,
                    "terminal_time": 502,
                    "reason": "downed_until_terminal",
                },
            ],
        )
        events = [{"time": t} for t in [318, 337, 350, 424, 434, 455, 463]]
        self.assertEqual(
            segments_from_events(events, 548, protected_intervals=protected),
            [
                {"start": 308, "end": 365, "first_signal": 318, "last_signal": 350},
                {
                    "start": 414,
                    "end": 504,
                    "first_signal": 424,
                    "last_signal": 501,
                    "protected_reasons": ["downed_until_terminal"],
                },
            ],
        )

    def test_single_false_frame_and_inactive_spectator_are_ignored(self):
        self.assertEqual(
            downed_intervals([row(t, downed=t == 5) for t in range(10)], 20), []
        )
        self.assertEqual(
            downed_intervals(
                [row(t / 2, downed=True, inactive=True) for t in range(10)], 20
            ),
            [],
        )

    def test_terminal_requires_contiguous_inactive_rows_not_just_two_votes(self):
        rows = [
            row(10, downed=True),
            row(10.5, downed=True),
            row(11, inactive=True),
            row(11.5),
            row(12, inactive=True),
        ]
        self.assertEqual(downed_intervals(rows, 40)[0]["end"], 25.5)
        rows.extend([row(12.5, inactive=True)])
        self.assertEqual(downed_intervals(rows, 40)[0]["terminal_time"], 12)
        self.assertEqual(downed_intervals(rows, 40)[0]["end"], 14)
        rows[-1]["time"] = 13.2
        self.assertIsNone(downed_intervals(rows, 40)[0]["terminal_time"])

    def test_recovery_does_not_extend_to_unrelated_later_terminal(self):
        rows = [
            {
                "time": t / 2,
                "downed": 100 <= t / 2 <= 110,
                "inactive_view": t / 2 >= 180,
            }
            for t in range(190, 380)
        ]
        self.assertEqual(
            downed_intervals(rows, 200),
            [
                {
                    "start": 100,
                    "end": 125,
                    "last_downed": 110,
                    "terminal_time": None,
                    "reason": "downed_recovery_tail",
                },
            ],
        )

    def test_host_aliases_and_terminal_tail_are_clamped_without_input_mutation(self):
        rows = [
            {"t": 10, "downed": True},
            {"t": 10.5, "downed": True},
            {"t": 11, "inactive": True},
            {"t": 11.5, "inactive": True},
        ]
        before = deepcopy(rows)
        self.assertEqual(
            downed_intervals(rows, 12, terminal_tail=5),
            [
                {
                    "start": 10,
                    "end": 12,
                    "last_downed": 10.5,
                    "terminal_time": 11,
                    "reason": "downed_until_terminal",
                },
            ],
        )
        self.assertEqual(rows, before)

    def test_large_down_gap_creates_separate_intervals(self):
        rows = [row(t, downed=True) for t in [10, 10.5, 46, 46.5]]
        self.assertEqual(
            [(s["start"], s["end"]) for s in downed_intervals(rows, 100)],
            [(10, 25.5), (46, 61.5)],
        )


class ImportBoundaryTests(unittest.TestCase):
    def test_pure_modules_import_with_no_site_packages_or_heavy_modules(self):
        root = Path(__file__).resolve().parents[1]
        code = (
            f"import sys; sys.path.insert(0, {str(root)!r}); "
            "from src import events, downed; "
            "assert not ({'numpy', 'cv2', 'onnxruntime', 'timepin_vision', "
            "'rapidocr_onnxruntime'} & set(sys.modules)); "
            "assert events.detect_events([]) == []; "
            "assert downed.downed_intervals([], 0) == []"
        )
        result = subprocess.run(
            [sys.executable, "-I", "-S", "-c", code],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_existing_mark_conversion_contract_is_unchanged(self):
        metadata = {"evidence": "own_damage_counter"}
        mark = event_to_mark(
            10, "combat", end_time=20, confidence=0.8, metadata=metadata
        )
        self.assertEqual(
            (mark["startTime"], mark["endTime"], mark["sourcePlugin"]),
            (10, 20, "apex-highlight"),
        )
        metadata["evidence"] = "changed"
        self.assertEqual(mark["metadata"], {"evidence": "own_damage_counter"})
        with self.assertRaises(ValueError):
            event_to_mark(float("nan"), "invalid")


if __name__ == "__main__":
    unittest.main()
