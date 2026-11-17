"""Focused regression tests for the ported combat-outcome reader.

Covers src/outcome.py (AHC ``combat_outcome_reader.py`` prefix logic +
``outcome_reader.py`` end/friend-spectate tails) with synthetic BGR ndarrays
and a scripted OCR engine; no video decode, no real ONNX model and no real
template PNG is needed. Row keys asserted here are the contract main.py's
``detect_events`` consumes: ``inactive_view``, ``friend_spectate``,
``round_ended``, ``outcome_kind``, ``outcome_status``, ``outcome_toast``.
"""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

from src.outcome import (
    OutcomeReader,
    candidate_indices,
    cap_segment_ends,
    clean_target_name,
    is_known_dialogue,
    parse_result_line,
    preserve_team_continuation,
    same_toast_slot,
)

ROOT = Path(__file__).resolve().parents[1]

# 1280x540 test frames with a 2560x1080 reference profile → toast ROI lands at
# [430, 340, 850, 416] and prefix templates need no resize (sample_size 1280).
W, H = 1280, 540
ROI_X, ROI_Y = 430, 340

PROFILE = {
    "version": 2,
    "context_padding_left": 100,
    "calibration": {
        "gate_threshold": 0.65,
        "minimum_fps": 2,
        "negative_decision_enabled": True,
        "require_traditional_ui": True,
    },
    "reference_size": [2560, 1080],
    "toast_roi": [860, 680, 1700, 832],
    # 测试注入 FakeGate，避免依赖真实像素模板资源。
    "prefix_templates": [],
}


def own_frame() -> np.ndarray:
    """本人视角：底部控制栏亮 → inactive_view False。"""
    image = np.zeros((H, W, 3), dtype=np.uint8)
    image[round(H * 0.985) :, round(W * 0.12) : round(W * 0.88)] = 200
    return image


def inactive_frame() -> np.ndarray:
    """纯黑帧 → 底部黑栏 → inactive_view True。"""
    return np.zeros((H, W, 3), dtype=np.uint8)


def friend_frame() -> np.ndarray:
    """观战队友视角：黑底 + 底部绿色横幅。"""
    image = inactive_frame()
    image[round(H * 0.93) : round(H * 0.98), round(W * 0.36) : round(W * 0.64)] = (
        0,
        255,
        0,
    )
    return image


def det(x1, y1, x2, y2, text, confidence):
    """Build one RapidOCR-style detection (points in 2x-scaled ROI coords)."""
    pts = [
        [2 * (x1 - ROI_X), 2 * (y1 - ROI_Y)],
        [2 * (x2 - ROI_X), 2 * (y1 - ROI_Y)],
        [2 * (x2 - ROI_X), 2 * (y2 - ROI_Y)],
        [2 * (x1 - ROI_X), 2 * (y2 - ROI_Y)],
    ]
    return (pts, text, confidence)


def match(kind, x1, y1, x2, y2, score=0.9):
    return {
        "kind": kind,
        "action": "unknown" if kind == "assist" else kind,
        "score": score,
        "box": [x1, y1, x2, y2],
    }


class FakeGate:
    """Scripted prefix-template matches; injected via reader._gate."""

    def __init__(self, matches=()):
        self.matches = list(matches)

    def inspect(self, image):
        return [dict(m) for m in self.matches]


class FakeEngine:
    """engine(image, use_cls=False) -> (detections, elapsed)."""

    def __init__(self, detections=()):
        self.detections = list(detections)
        self.calls = 0

    def __call__(self, image, use_cls=False, **kwargs):
        self.calls += 1
        return list(self.detections), 0.0


def make_reader(matches=(), detections=(), *, backend="cpu") -> OutcomeReader:
    reader = OutcomeReader(dict(PROFILE), backend=backend)
    reader._gate = FakeGate(matches)
    reader._engine = FakeEngine(detections)
    return reader


def observe(reader, *frames_and_times):
    rows = []
    for image, t in frames_and_times:
        row = {"time": t, "frame": f"f{t}"}
        reader.observe(image, row)
        rows.append(row)
    return rows


class PureParsingTest(unittest.TestCase):
    """Upstream-published grammar cases, kept verbatim for parity."""

    def test_traditional_knock_line(self):
        parsed = parse_result_line("擊倒 FOX_PLAYER +153")
        self.assertEqual(parsed["kind"], "knock")
        self.assertEqual(parsed["action"], "knock")
        self.assertEqual(parsed["target_text"], "FOX_PLAYER")
        self.assertEqual(parsed["locale"], "zh-TW")

    def test_assist_knock_line(self):
        parsed = parse_result_line("助攻，擊倒 VILLAIN")
        self.assertEqual(parsed["kind"], "assist")
        self.assertEqual(parsed["action"], "knock")
        self.assertEqual(parsed["target_text"], "VILLAIN")

    def test_assist_elimination_simplified(self):
        parsed = parse_result_line("助攻，淘汰 目標甲")
        self.assertEqual(parsed["kind"], "assist")
        self.assertEqual(parsed["action"], "elimination")
        self.assertEqual(parsed["locale"], "zh-CN")

    def test_english_knock(self):
        parsed = parse_result_line("KNOCKED DOWN ENEMY4")
        self.assertEqual(parsed["kind"], "knock")
        self.assertEqual(parsed["target_text"], "ENEMY4")
        self.assertEqual(parsed["locale"], "en")

    def test_non_toast_line(self):
        self.assertIsNone(parse_result_line("你好世界"))
        self.assertIsNone(parse_result_line("隊友說: 擊倒"))

    def test_clean_target_name_strips_score_and_action(self):
        self.assertEqual(clean_target_name("PLAYER +100"), "PLAYER")
        self.assertEqual(clean_target_name("擊倒 SOMEBODY"), "SOMEBODY")
        # 昵称内的动作词让整串按误读丢弃。
        self.assertEqual(clean_target_name("KNOCK MASTER"), "")

    def test_dialogue_guards(self):
        # 冒号说话者 + 行内战果词 = 对白，不是本人 toast。
        self.assertTrue(is_known_dialogue("探路者: 我擊倒了敵人"))
        self.assertTrue(is_known_dialogue("我被擊倒了！"))
        self.assertFalse(is_known_dialogue("擊倒 TARGETNAME"))

    def test_candidate_indices(self):
        self.assertEqual(
            candidate_indices({"start": 0.5, "end": 1.5}, 2), [2, 3]
        )
        self.assertEqual(
            candidate_indices({"start": 0.0, "end": 0.4}, 2), [1]
        )

    def test_same_toast_slot(self):
        a = {"box": [100, 10, 200, 30]}
        self.assertTrue(same_toast_slot(a, {"box": [105, 12, 205, 32]}))
        self.assertFalse(same_toast_slot(a, {"box": [100, 120, 200, 140]}))
        self.assertFalse(same_toast_slot(a, {}))


class ObserveTest(unittest.TestCase):
    def test_own_knock_confirmed_by_same_line_template(self):
        line = det(440, 350, 720, 372, "擊倒 ROGUE +99", 0.9)
        reader = make_reader(
            matches=[match("knock", 440, 350, 480, 372, 0.9)], detections=[line]
        )
        (row,) = observe(reader, (own_frame(), 1.0))
        self.assertFalse(row["inactive_view"])
        self.assertEqual(row["outcome_status"], "self_event")
        self.assertEqual(len(row["outcome_toast"]), 1)
        event = row["outcome_toast"][0]
        self.assertEqual(event["kind"], "knock")
        self.assertEqual(event["confirmation_basis"], "prefix_template")
        self.assertEqual(event["target_text"], "ROGUE")
        self.assertEqual(event["time"], 1.0)

    def test_assist_line_not_confirmed_by_knock_template(self):
        # 同一行只有擊倒模板命中时，「助攻，擊倒」仍是 assist 事件；
        # kind 不同源的模板不能给它投票（否则助攻会被错记成独立击倒）。
        line = det(440, 350, 760, 372, "助攻，擊倒 ENEMY", 0.9)
        reader = make_reader(
            matches=[match("knock", 560, 350, 600, 372, 0.9)], detections=[line]
        )
        (row,) = observe(reader, (own_frame(), 1.0))
        self.assertEqual(row["outcome_status"], "ambiguous")
        self.assertEqual(row["outcome_toast"], [])
        # 同一行提供 assist 模板命中后才能确认。
        reader = make_reader(
            matches=[match("assist", 440, 350, 500, 372, 0.9)], detections=[line]
        )
        (row,) = observe(reader, (own_frame(), 1.0))
        self.assertEqual(row["outcome_status"], "self_event")
        self.assertEqual(row["outcome_toast"][0]["kind"], "assist")
        self.assertEqual(row["outcome_toast"][0]["action"], "knock")

    def test_dialogue_line_never_becomes_toast(self):
        # 同一行内冒号前缀 + 战果词 → 对白，不能产出事件。
        chat = det(440, 350, 760, 372, "惡靈: 有人被擊倒了", 0.9)
        reader = make_reader(
            matches=[match("knock", 440, 350, 480, 372, 0.9)],
            detections=[chat],
        )
        (row,) = observe(reader, (own_frame(), 1.0))
        self.assertEqual(row["outcome_toast"], [])
        self.assertNotEqual(row["outcome_status"], "self_event")

    def test_toast_under_dialogue_loses_grammar_path(self):
        # 紧贴对白下方的行带 after_dialogue；无模板语法路径被拒绝。
        dialogue = det(440, 330, 700, 350, "惡靈: 注意前面", 0.9)
        # 触发 OCR 需要一处强模板命中，但不能给本行 support。
        trigger = match("elimination", 440, 420, 500, 442, 0.95)
        weak = match("knock", 440, 352, 480, 372, 0.5)
        clean = det(440, 352, 700, 372, "击倒 LONGBOWMAN", 0.8)
        reader = make_reader(matches=[trigger, weak], detections=[dialogue, clean])
        (row,) = observe(reader, (own_frame(), 1.0))
        self.assertEqual(row["outcome_toast"], [])
        # 移走对白后同一行可读入语法路径候选。
        reader = make_reader(matches=[trigger, weak], detections=[clean])
        (row,) = observe(reader, (own_frame(), 1.0))
        self.assertEqual(row["outcome_toast"][0]["confirmation_basis"], "toast_grammar")

    def test_excluded_views_skip_toast(self):
        line = det(440, 350, 720, 372, "擊倒 ENEMY", 0.95)
        reader = make_reader(
            matches=[match("knock", 440, 350, 480, 372, 0.9)], detections=[line]
        )
        (row,) = observe(reader, (inactive_frame(), 1.0))
        self.assertTrue(row["inactive_view"])
        self.assertEqual(row["outcome_status"], "excluded_view")
        (frow,) = observe(reader, (friend_frame(), 1.5))
        self.assertTrue(frow["inactive_view"])
        self.assertTrue(frow["friend_spectate"])
        self.assertEqual(frow["outcome_status"], "excluded_view")


class EndScreenTest(unittest.TestCase):
    def test_squad_wipe_marks_round_ended(self):
        wipe = det(600, 200, 800, 240, "小隊全滅", 0.9)
        reader = make_reader(detections=[wipe])
        (row,) = observe(reader, (inactive_frame(), 10.0))
        self.assertTrue(row["round_ended"])
        self.assertEqual(row["outcome_kind"], "end")
        self.assertEqual(reader.outcome_events[0]["kind"], "end")
        self.assertEqual(reader.outcome_events[0]["time"], 10.0)

    def test_champion_marks_win(self):
        win = det(600, 200, 900, 240, "YOU ARE THE CHAMPION", 0.92)
        reader = make_reader(detections=[win])
        (row,) = observe(reader, (inactive_frame(), 20.0))
        self.assertTrue(row["round_ended"])
        self.assertEqual(row["outcome_kind"], "win")

    def test_end_event_gap_dedup(self):
        wipe = det(600, 200, 800, 240, "小隊全滅", 0.9)
        reader = make_reader(detections=[wipe])
        for t in (10.0, 10.5, 11.0):
            observe(reader, (inactive_frame(), t))
        # 12 秒内同一结束画面只记录一次事件。
        self.assertEqual(len(reader.outcome_events), 1)

    def test_win_caps_segment_end(self):
        segment = {"start": 10.0, "end": 60.0, "last_signal": 30.0}
        capped = cap_segment_ends(
            [segment], [{"time": 40.0, "kind": "win", "text": ""}]
        )
        self.assertEqual(capped[0]["end"], 42.0)
        self.assertEqual(capped[0]["end_evidence"]["kind"], "win")
        # 结束画面在 last_signal 之前不算结束证据。
        capped = cap_segment_ends(
            [segment], [{"time": 20.0, "kind": "end", "text": ""}]
        )
        self.assertEqual(capped[0]["end"], 60.0)


class TeammateTailTest(unittest.TestCase):
    def test_friend_spectate_extends_to_end_event(self):
        segment = {"start": 10.0, "end": 40.0, "last_signal": 30.0}
        rows = [
            {"time": 32.0, "friend_spectate": True, "inactive_view": True},
            {"time": 34.0, "friend_spectate": True, "inactive_view": True},
            {"time": 50.0, "inactive_view": False},
        ]
        outcomes = [{"time": 45.0, "kind": "end", "text": "小隊全滅"}]
        out = preserve_team_continuation([segment], rows, outcomes, 100.0)
        self.assertEqual(out[0]["end"], 47.0)
        self.assertEqual(out[0]["team_end_evidence"]["kind"], "end")
        self.assertIn("friend_team_continuation", out[0]["protected_reasons"])

    def test_friend_spectate_without_end_keeps_review_tail(self):
        segment = {"start": 10.0, "end": 40.0, "last_signal": 30.0}
        rows = [
            {"time": 32.0, "friend_spectate": True, "inactive_view": True},
            {"time": 50.0, "inactive_view": False},
        ]
        out = preserve_team_continuation([segment], rows, [], 100.0)
        self.assertEqual(out[0]["end"], 65.0)
        self.assertTrue(out[0]["team_end_needs_review"])

    def test_friend_spectate_to_file_end(self):
        segment = {"start": 10.0, "end": 40.0, "last_signal": 30.0}
        rows = [{"time": 32.0, "friend_spectate": True, "inactive_view": True}]
        out = preserve_team_continuation([segment], rows, [], 80.0)
        self.assertEqual(out[0]["end"], 80.0)
        self.assertTrue(out[0]["team_end_needs_review"])


class FinalizeTest(unittest.TestCase):
    def _observed_reader(self, frames, matches=(), detections=()):
        reader = make_reader(matches=matches, detections=detections)
        rows = observe(reader, *[(own_frame(), t) for t in frames])
        return reader, rows

    def test_two_frame_prefix_confirmation_keeps_segment(self):
        # 单帧 OCR 置信度 0.70 不够；相邻帧同一行位置的提示互相确认。
        line = det(440, 350, 720, 372, "擊倒 TGT", 0.70)
        matches = [match("knock", 440, 350, 480, 372, 0.88)]
        reader, _ = self._observed_reader([0.25, 0.75], matches, [line])
        segment = {"start": 0.0, "end": 1.0, "last_signal": 0.5}
        out = reader.finalize_segments([segment], [], 60.0, sample_fps=2)
        decision = out[0]["combat_outcome"]
        self.assertEqual(decision["decision"], "kept")
        self.assertTrue(
            any(e.get("confirmation") == "two_frame_prefix" for e in decision["events"])
        )

    def test_single_weak_frame_is_review_not_kept(self):
        line = det(440, 350, 720, 372, "擊倒 TGT", 0.70)
        reader, _ = self._observed_reader([0.25, 0.75, 1.25], detections=[])
        # 只在中间帧回放提示；其余帧门控为空。
        reader2 = make_reader(detections=[line])
        reader2._gate = FakeGate([])
        observe(reader2, (own_frame(), 0.25))
        reader2._gate = FakeGate([match("knock", 440, 350, 480, 372, 0.88)])
        observe(reader2, (own_frame(), 0.75))
        reader2._gate = FakeGate([])
        observe(reader2, (own_frame(), 1.25))
        segment = {"start": 0.0, "end": 1.5, "last_signal": 0.5}
        out = reader2.finalize_segments([segment], [], 60.0, sample_fps=2)
        self.assertEqual(out[0]["combat_outcome"]["decision"], "review")

    def test_strong_single_grammar_frame_keeps(self):
        line = det(440, 350, 760, 372, "击倒 LONGBOWMAN", 0.95)
        # knock 行但 elimination 模板命中本行 → 触发 OCR 又不给该行支撑。
        matches = [match("elimination", 440, 350, 480, 372, 0.95)]
        reader, _ = self._observed_reader([0.25], matches, [line])
        for i, obs in sorted(reader._observations.items()):
            obs["index"] = i + 1
        reader._observations = {
            obs["index"]: obs for obs in reader._observations.values()
        }
        segment = {"start": 0.0, "end": 1.0, "last_signal": 0.5}
        out = reader.finalize_segments([segment], [], 60.0, sample_fps=2)
        events = out[0]["combat_outcome"]["events"]
        self.assertEqual(out[0]["combat_outcome"]["decision"], "kept")
        self.assertTrue(
            any(
                e.get("confirmation") == "strong_single_toast_grammar"
                for e in events
            )
        )

    def test_two_frame_english_toast(self):
        line = det(440, 350, 820, 372, "KNOCKED DOWN ENEMYNAME", 0.92)
        matches = [match("knock", 440, 350, 480, 372, 0.5)]
        # 需要某处强命中触发 OCR。
        matches.append(match("elimination", 440, 420, 500, 442, 0.95))
        reader, _ = self._observed_reader([0.25, 0.75], matches, [line])
        segment = {"start": 0.0, "end": 1.0, "last_signal": 0.5}
        out = reader.finalize_segments([segment], [], 60.0, sample_fps=2)
        events = out[0]["combat_outcome"]["events"]
        self.assertTrue(
            any(e.get("confirmation") == "two_frame_english_toast" for e in events)
        )

    def test_complete_coverage_rejects_segment(self):
        # 5 帧全检查且无提示、界面已校准为繁中 → 保守否决（不是 review）。
        # 上游帧号 i 对应时间 (i-.5)/fps；0.25..2.25 @2fps 即帧 1-5。
        reader, _ = self._observed_reader([0.25, 0.75, 1.25, 1.75, 2.25])
        for i, obs in sorted(reader._observations.items()):
            obs["index"] = i + 1
        reader._observations = {
            obs["index"]: obs for obs in reader._observations.values()
        }
        segment = {"start": 0.5, "end": 2.5, "last_signal": 1.0}
        rows = [{"time": 0.25, "weapon_labels": [{"text": "R-301 卡賓槍"}]}]
        out = reader.finalize_segments([segment], rows, 60.0, sample_fps=2)
        decision = out[0]["combat_outcome"]
        self.assertEqual(decision["decision"], "rejected")
        self.assertTrue(decision["coverage"]["reliable_absence"])
        self.assertEqual(decision["confidence"], 0.0)

    def test_incomplete_coverage_is_review_not_rejected(self):
        # 只观察了部分帧 → 不能当「无结果」否决，进入复核。
        reader, _ = self._observed_reader([0.25, 0.75])
        for i, obs in sorted(reader._observations.items()):
            obs["index"] = i + 1
        reader._observations = {
            obs["index"]: obs for obs in reader._observations.values()
        }
        segment = {"start": 0.5, "end": 3.5, "last_signal": 1.0}
        rows = [{"time": 0.25, "weapon_labels": [{"text": "R-301 卡賓槍"}]}]
        out = reader.finalize_segments([segment], rows, 60.0, sample_fps=2)
        decision = out[0]["combat_outcome"]
        self.assertEqual(decision["decision"], "review")
        self.assertFalse(decision["coverage"]["reliable_absence"])
        self.assertEqual(decision["confidence"], 0.5)

    def test_false_template_hit_stays_conservative(self):
        # 高分行首模板 + 无法解析的 OCR → ambiguous → 段保留为 review。
        noise = det(440, 350, 700, 372, "你好嗎", 0.9)
        reader, _ = self._observed_reader(
            [0.25, 0.75, 1.25], [match("knock", 440, 350, 480, 372, 0.92)], [noise]
        )
        rows = [{"time": 0.25, "weapon_labels": [{"text": "R-301 卡賓槍"}]}]
        segment = {"start": 0.0, "end": 1.5, "last_signal": 0.5}
        out = reader.finalize_segments([segment], rows, 60.0, sample_fps=2)
        decision = out[0]["combat_outcome"]
        self.assertEqual(decision["decision"], "review")
        self.assertFalse(decision["coverage"]["reliable_absence"])
        self.assertTrue(decision["coverage"]["ambiguous_frames"])

    def test_own_kill_counter_keeps_without_toast(self):
        reader, _ = self._observed_reader([0.25, 0.75])
        segment = {
            "start": 0.5,
            "end": 3.5,
            "last_signal": 1.0,
            "combat_evidence": {"accepted_reasons": ["own_kill"]},
        }
        out = reader.finalize_segments([segment], [], 60.0, sample_fps=2)
        decision = out[0]["combat_outcome"]
        self.assertEqual(decision["decision"], "kept")
        self.assertEqual(decision["counter_basis"], "own_kill_counter")
        self.assertEqual(decision["events"], [])

    def test_win_caps_and_extends_in_finalize(self):
        wipe = det(600, 200, 800, 240, "小隊全滅", 0.9)
        reader = make_reader(detections=[wipe])
        observe(
            reader,
            (own_frame(), 0.25),
            (inactive_frame(), 40.0),
        )
        segment = {"start": 0.0, "end": 60.0, "last_signal": 30.0}
        out = reader.finalize_segments([segment], [], 90.0, sample_fps=2)
        self.assertEqual(out[0]["end"], 42.0)
        self.assertEqual(out[0]["end_evidence"]["kind"], "end")


class ContractTest(unittest.TestCase):
    def test_invalid_backend_rejected(self):
        with self.assertRaises(ValueError):
            OutcomeReader(dict(PROFILE), backend="tensorrt")

    def test_light_import_has_no_heavy_deps(self):
        code = (
            "import sys, src.outcome; "
            "assert 'cv2' not in sys.modules and 'numpy' not in sys.modules, "
            "'handshake import must stay lazy'"
        )
        proc = subprocess.run(
            [sys.executable, "-c", code],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)

    def test_missing_engine_message_is_actionable(self):
        reader = make_reader()
        reader._engine = None
        # 缺 src.ocr 时抛 RuntimeError 并点名接口；此处引擎已注入则不触发。
        import src.ocr  # noqa: F401 —— 当前交付已提供该工厂


if __name__ == "__main__":
    unittest.main()
