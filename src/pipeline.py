"""Cancellable metadata traversal plus the AHC post-extraction finalize chain with optional HUD info enrichment."""

from __future__ import annotations

import json
import threading
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

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
from .extra import (
    CombatStats,
    HUDInfo,
    PlayerNameInfo,
    TeammateInfo,
    create_empty_hud_info,
    hud_info_to_metadata,
)
from .hud_layout import detect_images, read_image
from .counter import counter_offset, inactive_view, numeric
from .rank_reader import RankReader
from .hero_reader import HeroReader
from .name_reader import (
    NameReader,
    canonical_name,
    consensus_name,
    names_to_infos,
)
from .stats import StatCounter, StatReader, find_counter_delta


PROGRESS_INTERVAL = 32
# Per-field sampling bounds — recognition is per-frame cheap but HUD state is
# stable, so a handful of well-spread frames carries the signal.
LAYOUT_SAMPLE_MAX = 8
FIELD_SAMPLE_MAX = 6
STAT_SAMPLE_MAX = 10

_DIGITS_MODEL = Path(__file__).resolve().parent.parent / "resources" / "hud_digits_v1.onnx"


def _even_sample(items: Sequence[Any], limit: int) -> list[Any]:
    if len(items) <= limit:
        return list(items)
    step = (len(items) - 1) / max(1, limit - 1)
    return [items[round(i * step)] for i in range(limit)]


def _consensus_single(
    values: list[dict[str, Any] | None], key: str, minimum: int = 2
) -> dict[str, Any] | None:
    """Two-frame-consensus for categorical fields (rank tier, hero id)."""
    from collections import Counter

    rows = [v for v in values if v and v.get(key)]
    if not rows:
        return None
    counts = Counter(v[key] for v in rows)
    common, n = counts.most_common(1)[0]
    if n < minimum:
        return None
    selected = [v for v in rows if v[key] == common]
    return {
        **min(selected, key=lambda v: -v.get("confidence", 0)),
        "confidence": min(v.get("confidence", 0) for v in selected),
    }


def _make_name_recognizer() -> Callable[[list[np.ndarray]], list[tuple[str, float]]] | None:
    """Batch recognizer over image crops via the SDK OCR engine (multilingual)."""
    try:
        from timepin_vision import OcrEngine
    except Exception:
        return None
    try:
        engine = OcrEngine()
    except Exception:
        return None

    def recognize(images: list[np.ndarray]) -> list[tuple[str, float]]:
        out: list[tuple[str, float]] = []
        for image in images:
            try:
                results, _ = engine.recognize(image)
            except Exception:
                out.append(("", 0.0))
                continue
            if not results:
                out.append(("", 0.0))
                continue
            text = " ".join(t for t, _, _ in results)
            conf = float(min(c for _, c, _ in results))
            out.append((text, conf))
        return out

    return recognize


def _make_digit_decoder() -> Callable[[np.ndarray], tuple[str, float] | None] | None:
    if not _DIGITS_MODEL.is_file():
        return None
    try:
        from timepin_vision import OnnxDigitDecoder

        decoder = OnnxDigitDecoder(_DIGITS_MODEL)
    except Exception:
        return None

    def decode(image: np.ndarray) -> tuple[str, float] | None:
        result = decoder(image)
        if isinstance(result, (tuple, list)) and len(result) >= 2:
            return str(result[0]), float(result[1])
        return None

    return decode


def _extract_hud_info(
    frames: Sequence[Frame],
    params: Mapping[str, object],
    *,
    cancel_event: threading.Event,
) -> HUDInfo:
    """Extract optional HUD info (rank, hero, player name, teammates, stats).

    Every field is best-effort: a failed read leaves None and never raises
    beyond cancellation.
    """
    if not params.get("hud_info_enabled", False):
        return create_empty_hud_info()

    hud_info = create_empty_hud_info()
    if not frames:
        return hud_info

    # Load a bounded spread of playable frames (decode once, reuse everywhere).
    sample_frames = _even_sample(list(frames), LAYOUT_SAMPLE_MAX)
    loaded: list[tuple[Frame, np.ndarray]] = []
    for frame in sample_frames:
        if cancel_event.is_set():
            raise Cancelled()
        image = read_image(frame.path)
        if image is not None and not inactive_view(image):
            loaded.append((frame, image))
    if not loaded:
        return hud_info

    images = [image for _, image in loaded]
    h, w = images[0].shape[:2]

    # Layout detection — the counter marker anchors every HUD crop.
    try:
        profile = detect_images(images, w, h)
    except Exception:
        profile = None
    if profile is None:
        return hud_info

    # --- Rank & hero (template matching, cheap) ---------------------------------
    field_frames = _even_sample(loaded, FIELD_SAMPLE_MAX)
    rank_reader = RankReader()
    hero_reader = HeroReader()

    rank_reads: list[dict[str, Any] | None] = []
    hero_reads: list[dict[str, Any] | None] = []
    for _, image in field_frames:
        if cancel_event.is_set():
            raise Cancelled()
        try:
            rank = rank_reader.inspect(image, profile)
        except Exception:
            rank = None
        rank_reads.append(
            {
                "tier": rank.tier,
                "name": rank.name,
                "division": rank.division,
                "confidence": rank.confidence or 0.0,
                "method": rank.method,
                "mode": rank.mode,
            }
            if rank is not None
            else None
        )
        try:
            hero = hero_reader.inspect(image, profile)
        except Exception:
            hero = None
        hero_reads.append(
            {
                "id": hero.id,
                "name": hero.name,
                "confidence": hero.confidence or 0.0,
                "method": hero.method,
            }
            if hero is not None
            else None
        )

    # Rank mode ('other' = non-BR mode icon seen) is itself a valid read.
    mode_rows = [r for r in rank_reads if r and r.get("mode") == "other"]
    if len(mode_rows) >= 2:
        best = max(mode_rows, key=lambda r: r["confidence"] or 0.0)
        hud_info.rank.tier = None
        hud_info.rank.mode = "other"
        hud_info.rank.method = best["method"]
        hud_info.rank.confidence = best["confidence"]
    else:
        rank = _consensus_single(rank_reads, "tier")
        if rank:
            # Division must also agree on two frames to be trusted.
            divisions = [r["division"] for r in rank_reads if r and r.get("tier") == rank["tier"] and r.get("division")]
            from collections import Counter

            div_counts = Counter(divisions)
            division = (
                div_counts.most_common(1)[0][0] if div_counts and div_counts.most_common(1)[0][1] >= 2 else None
            )
            hud_info.rank.tier = rank["tier"]
            hud_info.rank.name = rank["name"]
            hud_info.rank.division = division
            hud_info.rank.confidence = rank["confidence"]
            hud_info.rank.method = rank["method"]

    hero = _consensus_single(hero_reads, "id")
    if hero:
        hud_info.hero.id = hero["id"]
        hud_info.hero.name = hero["name"]
        hud_info.hero.confidence = hero["confidence"]
        hud_info.hero.method = hero["method"]

    # --- Player & teammate names (OCR, gated on engine availability) -----------
    recognizer = _make_name_recognizer()
    if recognizer is not None:
        reader = NameReader(recognizer)
        player_reads: list[dict[str, Any] | None] = []
        teammate_reads: list[list[dict[str, Any] | None]] = [[], []]
        for _, image in field_frames:
            if cancel_event.is_set():
                raise Cancelled()
            rh, rw = profile["reference_size"]
            patches = []
            rois = [profile["account_roi"], *profile.get("teammate_rois", [])]
            for x1, y1, x2, y2 in rois:
                patch = image[
                    round(y1 * image.shape[0] / rh) : round(y2 * image.shape[0] / rh),
                    round(x1 * image.shape[1] / rw) : round(x2 * image.shape[1] / rw),
                ]
                patches.append(patch if patch.size else np.zeros((4, 4, 3), np.uint8))
            try:
                values = reader(patches)
            except Exception:
                values = [None] * len(patches)
            player_reads.append(values[0] if values else None)
            for slot in range(min(2, len(values) - 1)):
                teammate_reads[slot].append(values[slot + 1])

        player = consensus_name(player_reads)
        teammates = [consensus_name(rows) for rows in teammate_reads]
        # Recording-wide reference: account name is the canonical self spelling.
        player = canonical_name(player, player)
        player_info, teammate_infos = names_to_infos(player, teammates)
        hud_info.player_name = player_info or PlayerNameInfo()
        hud_info.teammates = teammate_infos

    # --- Combat stats (digit recognition, per-frame point-in-time) -------------
    decode = _make_digit_decoder()
    if decode is not None:
        try:
            marker = read_image(profile["counter_marker"], 0)
            stat_frames = _even_sample(loaded, STAT_SAMPLE_MAX)
            delta, _score = find_counter_delta([img for _, img in stat_frames], profile)
            if delta is not None:
                counter = StatCounter()
                stat_reader = StatReader(decode, counter)
                best = CombatStats()
                for _, image in stat_frames:
                    if cancel_event.is_set():
                        raise Cancelled()
                    try:
                        stats = stat_reader.read_frame(image, profile, delta)
                    except Exception:
                        continue
                    # The strip reads a live counter; peak values are the stable ones.
                    for key in ("damage", "assists"):
                        current = getattr(stats, key)
                        if current is not None and (
                            getattr(best, key) is None or current > getattr(best, key)
                        ):
                            setattr(best, key, current)
                    try:
                        regions = counter.regions(image, profile, delta)
                        patch = regions.get("kills")
                        if patch is not None:
                            decoded = stat_reader.decode(patch)
                            if decoded is not None:
                                value = numeric(decoded[0], decoded[1], 99)
                                if value is not None and (best.kills is None or value > best.kills):
                                    best.kills = value
                    except Exception:
                        pass
                hud_info.stats = best
        except Exception:
            pass

    return hud_info


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