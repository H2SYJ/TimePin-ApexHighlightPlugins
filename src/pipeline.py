"""Cancellable metadata traversal plus the AHC post-extraction finalize chain."""

from __future__ import annotations

import json
import threading
from collections.abc import Callable, Mapping, Sequence

from .downed import downed_intervals
from .errors import Cancelled
from .events import (
    Mark,
    classify_combat_candidates,
    detect_events,
    event_to_mark,
    segments_from_events,
)
from .frames import Frame

PROGRESS_INTERVAL = 32

# 与 manifest parametersSchema 对应的宿主策略参数；其余字段原样透传。
_PARAM_FLOATS = (
    "gap_seconds",
    "min_segment_seconds",
    "max_segment_seconds",
)

_CONFIRMED_COLOR = "#F6C343"
_REVIEW_COLOR = "#8A8F98"
_DOWNED_COLOR = "#5BD48A"


def _label_for(segment: Mapping[str, object]) -> str:
    """按段内最显著证据生成中文标签；击杀>倒地>命中>射击。"""
    kinds = [
        event.get("kind")
        for event in (segment.get("combat_outcome", {}) or {}).get("events", [])
    ]
    reasons = (segment.get("combat_evidence", {}) or {}).get("accepted_reasons", [])
    outcome = segment.get("combat_outcome", {}) or {}
    if outcome.get("counter_basis") == "own_kill_counter" or "own_kill" in reasons:
        return "击杀交战"
    if "elimination" in kinds:
        return "击杀交战"
    if "knock" in kinds:
        return "击倒交战"
    if "assist" in kinds:
        return "助攻交战"
    if segment.get("protected_reasons"):
        return "倒地救援"
    if "repeated_hit_feedback" in reasons or "strong_hit_with_shot" in reasons:
        return "命中交战"
    return "交战片段"


def _metadata(segment: Mapping[str, object], sample_fps: float) -> dict[str, str]:
    """把可复核证据序列化进 Mark.metadata（全部是 string）。"""
    evidence = segment.get("combat_evidence", {}) or {}
    outcome = segment.get("combat_outcome", {}) or {}
    coverage = outcome.get("coverage", {}) or {}
    reasons = evidence.get("accepted_reasons", [])
    metadata = {
        "classification": str(segment.get("classification", "review")),
        "decision": str(outcome.get("decision", "review")),
        "sample_fps": f"{sample_fps:g}",
        "first_signal": f"{segment.get('first_signal', 0):.3f}",
        "last_signal": f"{segment.get('last_signal', 0):.3f}",
        "shoot_samples": str(evidence.get("shoot_samples", 0)),
        "hit_updates": str(evidence.get("hit_updates", 0)),
        "checked_frames": str(coverage.get("checked_frames", 0)),
        "expected_frames": str(coverage.get("expected_frames", 0)),
    }
    if reasons:
        metadata["accepted_reasons"] = ",".join(str(r) for r in reasons)
    if outcome.get("reason"):
        metadata["outcome_reason"] = str(outcome["reason"])
    if outcome.get("counter_basis"):
        metadata["counter_basis"] = str(outcome["counter_basis"])
    if segment.get("review_reason"):
        metadata["review_reason"] = str(segment["review_reason"])
    if segment.get("protected_reasons"):
        metadata["protected_reasons"] = ",".join(
            str(r) for r in segment["protected_reasons"]
        )
    events = outcome.get("events") or []
    if events:
        metadata["outcome_kinds"] = ",".join(
            sorted({str(e.get("kind")) for e in events})
        )
    weapons = [e.get("weapon") for e in events if e.get("weapon")]
    if weapons:
        metadata["weapons"] = json.dumps(sorted(set(weapons)))
    timeline = [
        {
            "t": float(e["time"]),
            "weapon": e.get("weapon") or "未知枪",
            "kind": e.get("kind"),
        }
        for e in events
        if e.get("kind") == "shoot"
    ]
    if timeline:
        metadata["weapon_timeline"] = json.dumps(timeline, ensure_ascii=False)
    return metadata


def finalize_marks(
    rows: Sequence[dict],
    duration: float,
    sample_fps: float,
    *,
    outcome_reader=None,
    params: Mapping[str, object] | None = None,
    cancel_event: threading.Event | None = None,
) -> dict[str, object]:
    """把 HUD 行流合成为最终 Mark 列表，复刻 AHC 决策链。

    顺序（与上游 apex_clipper 主流程一致）：
    downed_intervals → detect_events → segments_from_events（含 protected）
    → classify_combat_candidates → OutcomeReader.finalize_segments
    → 宿主 min/max 裁剪 → event_to_mark。

    返回 ``{"marks": [...], "logs": [...], "segments": [...]}``。
    ``outcome_reader`` 为 ``None`` 时跳过战果提示决策，所有候选段按
    classification 直接落 mark（review 段低置信度保留）。
    """

    def cancelled() -> bool:
        return cancel_event is not None and cancel_event.is_set()

    if cancelled():
        raise Cancelled()

    params = params or {}
    # AHC 的 gap 是 35 秒事件间隔；宿主 gap_seconds 是 UI 暴露的片段间隙，
    # 这里作为 gap 的上界使用，默认沿用上游 35。
    gap = float(params.get("gap_seconds") or 35.0)
    if gap <= 0:
        gap = 35.0
    min_seg = float(params.get("min_segment_seconds") or 0.0)
    max_seg = float(params.get("max_segment_seconds") or 0.0)

    row_list = [dict(r) for r in rows]
    logs: list[str] = []

    protected = downed_intervals(row_list, duration, gap=35.0)
    if cancelled():
        raise Cancelled()
    events = detect_events(row_list)
    if cancelled():
        raise Cancelled()
    weapon_rows = []
    if params.get("enable_weapon_detection") is not False:
        for row in row_list:
            obs = row.get("weapon_observation")
            if obs and isinstance(obs, dict):
                obs_time = (
                    obs.get("actual_pts")
                    or row.get("time")
                    or row.get("t")
                )
                if obs_time is not None:
                    obs = dict(obs)  # shallow copy to avoid mutating caller rows
                    obs["actual_pts"] = float(obs_time)
                    weapon_rows.append(obs)
    if weapon_rows:
        from .weapon import annotate_events

        events = annotate_events(events, weapon_rows, cancel_event=cancel_event)
        if cancelled():
            raise Cancelled()
    segments = segments_from_events(
        events,
        duration,
        gap=max(gap, 1.0),
        protected_intervals=protected,
    )
    if cancelled():
        raise Cancelled()
    confirmed, review = classify_combat_candidates(events, segments)
    candidates = sorted(
        confirmed + review, key=lambda s: (s["start"], s["end"])
    )

    if outcome_reader is not None:
        decided = outcome_reader.finalize_segments(
            candidates, row_list, duration, sample_fps=sample_fps
        )
    else:
        decided = []
        for segment in candidates:
            item = dict(segment)
            item["combat_outcome"] = {
                "decision": "kept"
                if segment.get("classification") == "confirmed"
                else "review",
                "confidence": 1.0
                if segment.get("classification") == "confirmed"
                else 0.5,
                "events": [],
                "reason": "outcome_reader 未提供（无战果提示决策）",
                "coverage": {},
                "version": "own-center-result-v4",
            }
            decided.append(item)

    marks: list[Mark] = []
    kept = review_count = rejected = 0
    for segment in decided:
        if cancelled():
            raise Cancelled()
        outcome = segment["combat_outcome"]
        decision = outcome["decision"]
        if decision == "rejected":
            rejected += 1
            continue
        start = float(segment["start"])
        end = float(segment["end"])
        length = end - start
        if max_seg > 0 and length > max_seg:
            # 超出上限：保留以最后信号为中心的最大窗口。
            anchor = float(segment.get("last_signal", end))
            start = max(0.0, anchor - max_seg * 0.4)
            end = min(duration, start + max_seg)
            start = max(0.0, end - max_seg)
            length = end - start
        if min_seg > 0 and length < min_seg and not segment.get(
            "protected_reasons"
        ):
            # 过短的非受保护段降级为瞬时事件点而非区间。
            end = start
        is_confirmed = decision == "kept"
        confidence = float(outcome.get("confidence") or 0.0)
        if not is_confirmed:
            review_count += 1
        else:
            kept += 1
        marks.append(
            event_to_mark(
                start,
                _label_for(segment),
                end_time=end,
                confidence=min(1.0, max(0.0, confidence)),
                metadata=_metadata(segment, sample_fps),
            )
        )
        marks[-1]["color"] = _CONFIRMED_COLOR if is_confirmed else _REVIEW_COLOR

    # 倒地生命周期本身也产出独立 mark（与交战段并存，便于时间轴筛选）。
    for interval in protected:
        if cancelled():
            raise Cancelled()
        marks.append(
            event_to_mark(
                float(interval["start"]),
                "倒地",
                end_time=float(interval["end"]),
                confidence=0.9,
                metadata={
                    "classification": "protected",
                    "decision": "kept",
                    "reason": str(interval["reason"]),
                    "last_downed": f"{interval['last_downed']:.3f}",
                },
            )
        )
        marks[-1]["color"] = _DOWNED_COLOR

    marks.sort(key=lambda m: (m["startTime"], m["endTime"] or m["startTime"]))
    logs.append(
        f"finalize: rows={len(row_list)} events={len(events)} "
        f"segments={len(decided)} kept={kept} review={review_count} "
        f"rejected={rejected} downed={len(protected)} marks={len(marks)}"
    )
    return {"marks": marks, "logs": logs, "segments": decided}


def analyze(
    frames: Sequence[Frame],
    params: Mapping[str, object],
    *,
    cancel_event: threading.Event,
    on_progress: Callable[[int, int], None] | None = None,
) -> list[Mark]:
    """Visit metadata only, reporting counts every 32 frames and at completion.

    Parameters are reserved for future detectors. No image/SDK/model is loaded,
    no media is written, and no synthetic marks are emitted. Empty inputs report
    (0, 0); a final exact multiple of 32 is reported only once.
    """
    if cancel_event.is_set():
        raise Cancelled()
    total = len(frames)
    processed = 0
    for _frame in frames:
        if cancel_event.is_set():
            raise Cancelled()
        processed += 1
        if on_progress is not None and processed % PROGRESS_INTERVAL == 0:
            on_progress(processed, total)

    if cancel_event.is_set():
        raise Cancelled()
    if on_progress is not None and (
        processed == 0 or processed % PROGRESS_INTERVAL != 0
    ):
        on_progress(processed, total)
    # Also observe cancellation requested from the final progress callback.
    if cancel_event.is_set():
        raise Cancelled()
    return []
