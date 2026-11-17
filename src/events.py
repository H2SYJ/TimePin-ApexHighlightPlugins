"""Host-compatible marks and pure AHC combat event/segment algorithms."""

from __future__ import annotations

import base64
import itertools
import math
from typing import TypedDict
from uuid import uuid4


class Mark(TypedDict):
    id: str
    startTime: float
    endTime: float | None
    label: str
    color: str | None
    confidence: float | None
    sourcePlugin: str | None
    thumbnailPath: str | None
    metadata: dict[str, str]


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    try:
        number = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    return number


def event_to_mark(
    start_time: float,
    label: str,
    *,
    end_time: float | None = None,
    confidence: float | None = None,
    metadata: dict[str, str] | None = None,
) -> Mark:
    """Convert a real detected event, raising ValueError for invalid values.

    This helper does not detect events and is intentionally not called by the
    skeleton pipeline. Metadata is copied so later caller mutations are isolated.
    """
    start = _finite_number(start_time, "start_time")
    if start < 0:
        raise ValueError("start_time must be nonnegative")
    if not isinstance(label, str):
        raise ValueError("label must be a string")
    end = None if end_time is None else _finite_number(end_time, "end_time")
    if end is not None and end < start:
        raise ValueError("end_time must be greater than or equal to start_time")
    score = None if confidence is None else _finite_number(confidence, "confidence")
    if score is not None and not 0 <= score <= 1:
        raise ValueError("confidence must be between 0 and 1")
    if metadata is not None and (
        not isinstance(metadata, dict)
        or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in metadata.items()
        )
    ):
        raise ValueError("metadata must be a dictionary of strings to strings")

    return {
        "id": str(uuid4()),
        "startTime": start,
        "endTime": end,
        "label": label,
        "color": None,
        "confidence": score,
        "sourcePlugin": "apex-highlight",
        "thumbnailPath": None,
        "metadata": {} if metadata is None else dict(metadata),
    }


# Ported from Apex Highlight Clipper's apex_clipper.py (stable_counter,
# detect_events, segments_from_events, classify_combat_candidates). The numerical
# thresholds, ordering, counter-baseline rules and evidence strings are retained.
# This boundary only adapts host row names; it never reads media or loads models.
def _ahc_rows(rows: list[dict]) -> list[dict]:
    """Copy host rows into AHC names without changing any caller-owned data.

    Both AHC's time/inactive_view/current_weapon/active_weapon_slot and the
    sidecar's t/inactive/weapon/weapon_slot spellings are accepted. AHC keys win
    when both are supplied. Inputs must already be in global timestamp order;
    sorting or window-local resets would change the upstream index contract.
    """
    normalized = []
    for row in rows:
        item = dict(row)
        if "time" not in item:
            item["time"] = item["t"]
        if "inactive_view" not in item:
            item["inactive_view"] = item.get("inactive", False)
        if "current_weapon" not in item and item.get("weapon"):
            weapon = item["weapon"]
            item["current_weapon"] = (
                weapon if isinstance(weapon, dict) else {"id": weapon}
            )
        if "active_weapon_slot" not in item and "weapon_slot" in item:
            item["active_weapon_slot"] = item["weapon_slot"]
        normalized.append(item)
    return normalized


def icon_similarity(a: str, b: str) -> float:
    """AHC's Dice overlap of packed edge bits, with no NumPy dependency.

    Counting set bits in the packed bytes is exactly equivalent to unpackbits /
    count_nonzero. Empty edge maps deliberately score zero, even if identical.
    """
    left = base64.b64decode(a)
    right = base64.b64decode(b)
    if len(left) != len(right):
        raise ValueError("Weapon icon signatures must have equal packed lengths")
    x = int.from_bytes(left, "big")
    y = int.from_bytes(right, "big")
    return 2 * (x & y).bit_count() / max(x.bit_count() + y.bit_count(), 1)


def stable_counter(rows: list[dict], key: str) -> dict[int, int]:
    """Confirm counters with AHC's two votes in a +/-2-row, 1.1s window."""
    rows = _ahc_rows(rows)
    result = {}
    for i, row in enumerate(rows):
        value = None if row.get("inactive_view") else row.get(key)
        if value is None:
            continue
        neighbours = rows[max(0, i - 2) : min(len(rows), i + 3)]
        votes = sum(
            r.get(key) == value and abs(r["time"] - row["time"]) <= 1.1
            for r in neighbours
        )
        if votes >= 2:
            result[i] = value
    return result


def detect_events(rows: list[dict], ignore_weapon: bool = False) -> list[dict]:
    """Port AHC counter/shoot evidence, retaining global ``time`` timestamps.

    ``ignore_weapon`` is an explicit opt-out of the weapon-swap guard. It permits
    ammo-only callers without icon signatures; the default is upstream behavior.
    Run once on the complete row stream, not once per host extraction window.
    """
    rows = _ahc_rows(rows)
    events = []
    reborn = []
    for key, kind, max_step in [("damage", "hit", 1000), ("kills", "kill", 3)]:
        confirmed = stable_counter(rows, key)
        previous = None
        previous_t = None
        for i, value in confirmed.items():
            t = rows[i]["time"]
            if (
                previous is not None
                and value > previous
                and value - previous <= max_step
                and t - previous_t <= 20
            ):
                step = value - previous
                if kind == "kill" and step > 1:
                    for index in range(step):
                        events.append(
                            {
                                "time": t,
                                "kind": kind,
                                "before": previous + index,
                                "after": previous + index + 1,
                                "evidence": "own_" + key + "_counter",
                            }
                        )
                else:
                    events.append(
                        {
                            "time": t,
                            "kind": kind,
                            "before": previous,
                            "after": value,
                            "evidence": "own_" + key + "_counter",
                        }
                    )
            if previous is not None and t - previous_t > 20:
                between = [r for r in rows if previous_t < r["time"] < t]
                if value > 0 and any(r.get("round_ended", False) for r in between):
                    reborn.append(
                        {
                            "time": t,
                            "kind": kind,
                            "before": 0,
                            "after": value,
                            "evidence": "new_round_counter",
                        }
                    )
            # A rejected high jump must not poison the next accepted baseline.
            if (
                previous is None
                or t - previous_t > 20
                or (value >= previous and value - previous <= max_step)
            ):
                previous = value
                previous_t = t

    candidates = []
    previous = None
    for i, b in enumerate(rows):
        if b.get("inactive_view"):
            previous = None
            continue
        if b.get("ammo") is None:
            continue
        old = previous
        previous = i
        if old is None:
            continue
        a = rows[old]
        if b["time"] - a["time"] > 1.1:
            continue
        drop = a["ammo"] - b["ammo"]
        if not 1 <= drop <= 50:
            continue
        similarity = (
            0.0
            if ignore_weapon
            else icon_similarity(a["weapon_icon"], b["weapon_icon"])
        )
        same_weapon = ignore_weapon or similarity >= 0.65
        if not ignore_weapon:
            if a.get("current_weapon") and b.get("current_weapon"):
                same_weapon = (
                    same_weapon
                    and a["current_weapon"]["id"] == b["current_weapon"]["id"]
                )
            if (
                a.get("active_weapon_slot") is not None
                and b.get("active_weapon_slot") is not None
            ):
                same_weapon = (
                    same_weapon and a["active_weapon_slot"] == b["active_weapon_slot"]
                )
        candidates.append(
            {
                "time": b["time"],
                "kind": "shoot",
                "before": a["ammo"],
                "after": b["ammo"],
                "evidence": "ammo_decrease",
                "same_weapon": bool(same_weapon),
                "icon_similarity": round(similarity, 3),
                "index": i,
            }
        )
    for candidate in candidates:
        confirmed_hit = any(
            e["kind"] == "hit" and abs(e["time"] - candidate["time"]) <= 1.5
            for e in events
        )
        if candidate["same_weapon"] or confirmed_hit:
            events.append(candidate)

    # A newly visible positive counter needs nearby own shooting. A recording
    # that starts mid-round keeps its opening counters as the baseline instead.
    for key, kind in [("damage", "hit"), ("kills", "kill")]:
        confirmed = stable_counter(rows, key)
        if confirmed:
            i = next(iter(confirmed))
            value = confirmed[i]
            t = rows[i]["time"]
            if (
                value > 0
                and t - rows[0]["time"] >= 15
                and any(
                    e["kind"] == "shoot" and abs(e["time"] - t) <= 12 for e in events
                )
            ):
                events.append(
                    {
                        "time": t,
                        "kind": kind,
                        "before": None,
                        "after": value,
                        "evidence": "own_counter_became_visible",
                    }
                )
    for birth in reborn:
        if any(
            e["kind"] == "shoot" and abs(e["time"] - birth["time"]) <= 12
            for e in events
        ):
            events.append(birth)
    return sorted(events, key=lambda e: e["time"])


def segments_from_events(
    events: list[dict],
    duration: float,
    gap: float = 35,
    pre: float = 10,
    post: float = 15,
    protected_intervals: list[dict] | None = None,
) -> list[dict]:
    """Globally group AHC signals, merge padding and protect downed lifecycles.

    No min/max clip limits or fixed 60-second window boundaries are imposed.
    Those are host policy, to be applied after global segmentation.
    """
    if not all(math.isfinite(x) and x >= 0 for x in [duration, gap, pre, post]):
        raise ValueError("时间参数必须为非负有限数")
    events = _ahc_rows(events)
    times = sorted({e["time"] for e in events if 0 <= e["time"] < duration})
    if not times and not protected_intervals:
        return []
    groups = []
    first = last = times[0] if times else None
    for t in times[1:]:
        if t - last <= gap:
            last = t
        else:
            groups.append((first, last))
            first = last = t
    if times:
        groups.append((first, last))
    segments = []
    for first, last in groups:
        start = max(0, first - pre)
        end = min(duration, last + post)
        if segments and start <= segments[-1]["end"]:
            segments[-1]["end"] = max(segments[-1]["end"], end)
            segments[-1]["last_signal"] = last
        else:
            segments.append(
                {"start": start, "end": end, "first_signal": first, "last_signal": last}
            )
    for interval in protected_intervals or []:
        if not (0 <= interval["start"] < interval["end"] <= duration):
            raise ValueError("非法生命周期片段")
        segments.append(
            {
                "start": max(0, interval["start"] - pre),
                "end": interval["end"],
                "first_signal": interval["start"],
                "last_signal": interval.get("last_downed", interval["start"]),
                "protected_reasons": [interval["reason"]],
            }
        )
    merged = []
    for segment in sorted(segments, key=lambda s: s["start"]):
        if merged and segment["start"] <= merged[-1]["end"]:
            target = merged[-1]
            target["end"] = max(target["end"], segment["end"])
            target["first_signal"] = min(
                target["first_signal"], segment["first_signal"]
            )
            target["last_signal"] = max(target["last_signal"], segment["last_signal"])
            reasons = target.get("protected_reasons", []) + segment.get(
                "protected_reasons", []
            )
            if reasons:
                target["protected_reasons"] = sorted(set(reasons))
        else:
            merged.append(dict(segment))
    return merged


def classify_combat_candidates(
    events: list[dict],
    segments: list[dict],
    hit_window: float = 12,
) -> tuple[list[dict], list[dict]]:
    """AHC API: return ``(confirmed, review)``; do not discard weak candidates.

    Parameter order and the tuple return deliberately match AHC rather than
    collapsing review candidates into a single accepted-segment list.
    """
    events = _ahc_rows(events)
    confirmed = []
    review = []
    for segment in segments:
        relevant = [e for e in events if segment["start"] <= e["time"] < segment["end"]]
        hits = sorted(
            e["time"]
            for e in relevant
            if e["kind"] == "hit" and e.get("after", 0) > (e.get("before") or 0)
        )
        has_kill = any(
            e["kind"] == "kill" and e.get("after", 0) > (e.get("before") or 0)
            for e in relevant
        )
        continuing_downed = bool(segment.get("protected_reasons"))
        repeated_hits = any(
            0 < b - a <= hit_window for a, b in itertools.pairwise(hits)
        )
        strong_single = any(
            e["kind"] == "hit"
            and e.get("after", 0) - (e.get("before") or 0) >= 60
            and any(
                s["kind"] == "shoot" and abs(s["time"] - e["time"]) <= 1.5
                for s in relevant
            )
            for e in relevant
        )
        reasons = []
        if has_kill:
            reasons.append("own_kill")
        if continuing_downed:
            reasons.append("downed_continuation")
        if repeated_hits:
            reasons.append("repeated_hit_feedback")
        if strong_single:
            reasons.append("strong_hit_with_shot")
        candidate = dict(segment)
        candidate["combat_evidence"] = {
            "shoot_samples": sum(e["kind"] == "shoot" for e in relevant),
            "hit_updates": len(hits),
            "repeated_hit_window": hit_window,
            "accepted_reasons": reasons,
        }
        candidate["classification"] = "confirmed" if reasons else "review"
        if reasons:
            confirmed.append(candidate)
        else:
            candidate["review_reason"] = (
                "只有开枪或孤立命中，缺少连续命中/击杀/倒地延续证据"
            )
            review.append(candidate)
    return confirmed, review
