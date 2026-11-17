"""Rank recognition using bounded template matching on the badge region.

Ported from apex-highlight-clipper rank_reader.py (hud-rank-v6). Only the
self-view HUD badge region is searched — never the whole frame.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .extra import RankInfo
from .hud_layout import read_image

RESOURCES = Path(__file__).resolve().parent.parent / "resources"

RANK_NAMES = {
    "rookie": "菜鸟",
    "bronze": "青铜",
    "silver": "白银",
    "gold": "黄金",
    "platinum": "白金",
    "diamond": "钻石",
    "master": "大师",
    "predator": "猎杀",
    "unranked": "匹配",
}


def empty_badge_region(roi: np.ndarray) -> dict[str, float] | None:
    """Treat only a clearly empty region as badge-absent; an unknown-shaped
    badge stays unrecognized rather than mislabeled."""
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    contrast = float(np.percentile(gray, 95) - np.percentile(gray, 5))
    edges = float(np.count_nonzero(cv2.Canny(gray, 40, 100))) / gray.size
    return {"contrast": round(contrast, 3), "edge_fraction": round(edges, 4)} if contrast < 24 and edges < 0.02 else None


def shifted_squad_panel(roi: np.ndarray) -> dict[str, float] | None:
    """With no rank badge the squads-remaining panel shifts right into the badge
    slot (observed on real HUDs); with a badge the area is a dark badge."""
    h, w = roi.shape[:2]
    box = roi[round(h * 48 / 108) : round(h * 82 / 108), round(w * 5 / 120) : round(w * 65 / 120)]
    if not box.size:
        return None
    hsv = cv2.cvtColor(box, cv2.COLOR_BGR2HSV)
    white = float(((hsv[:, :, 1] < 45) & (hsv[:, :, 2] > 215)).mean())
    dark = float((hsv[:, :, 2] < 70).mean())
    return {"panel_white": round(white, 3), "panel_glyphs": round(dark, 3)} if white >= 0.5 and 0.05 <= dark <= 0.4 else None


_MODE_ICONS: list[tuple[np.ndarray, str, float]] | None = None
# Non-BR modes (control/mixtape clock, green heart etc.) put a mode icon where
# the rank badge sits. Thresholds measured on real recordings: clock 0.48–0.78,
# heart 0.72; all badge/empty regions stay below 0.32 on both.
MODE_ICONS = (("mode-clock.png", "clock", 0.42), ("mode-heart.png", "heart", 0.55))


def mode_clock_evidence(roi: np.ndarray) -> dict[str, Any] | None:
    global _MODE_ICONS
    if _MODE_ICONS is None:
        _MODE_ICONS = [
            (read_image(RESOURCES / "modes" / name, cv2.IMREAD_COLOR), icon, threshold)
            for name, icon, threshold in MODE_ICONS
            if (RESOURCES / "modes" / name).is_file()
        ]
    if roi.size == 0:
        return None
    best = None
    for template, icon, threshold in _MODE_ICONS:
        if template is None:
            continue
        th, tw = template.shape[:2]
        if roi.shape[0] < th or roi.shape[1] < tw:
            continue
        _, score, _, point = cv2.minMaxLoc(cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED))
        if score >= threshold and (best is None or score - threshold > best["score"] - best["threshold"]):
            best = {"score": round(float(score), 3), "point": [int(point[0]), int(point[1])], "icon": icon, "threshold": threshold}
    if best:
        best.pop("threshold")
    return best


def palette_matches(template: np.ndarray, patch: np.ndarray) -> bool:
    """Same-shape badges must still match their palette; shape threshold alone
    is used when color evidence is insufficient."""
    reference = cv2.cvtColor(template, cv2.COLOR_BGR2HSV)
    observed = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    mask = (reference[:, :, 1] > 60) & (reference[:, :, 2] > 110)
    colored = mask & (observed[:, :, 1] > 60) & (observed[:, :, 2] > 110)
    if np.count_nonzero(colored) < max(8, np.count_nonzero(mask) * 0.25):
        bright = mask & (observed[:, :, 2] > 110)
        if (
            np.count_nonzero(mask) >= 16
            and np.count_nonzero(bright) >= np.count_nonzero(mask) * 0.5
            and float(np.median(observed[:, :, 1][bright])) < 30
        ):
            return False
        return True

    def hue(image: np.ndarray, selection: np.ndarray) -> float:
        angles = image[:, :, 0][selection].astype(float) * np.pi / 90
        weights = image[:, :, 1][selection].astype(float) * image[:, :, 2][selection]
        return float(np.arctan2(np.sum(np.sin(angles) * weights), np.sum(np.cos(angles) * weights)) * 90 / np.pi % 180)

    difference = abs(hue(reference, mask) - hue(observed, colored))
    return min(difference, 180 - difference) <= 18


@dataclass
class _RankTemplates:
    key: str
    center_image: np.ndarray
    scaled_templates: list[np.ndarray]


class RankReader:
    def __init__(self) -> None:
        self.templates: list[_RankTemplates] = []
        self.native_centers: dict[str, np.ndarray] = {}
        self._load_templates()

    def _load_templates(self) -> None:
        for key in RANK_NAMES:
            template_path = RESOURCES / "ranks" / f"{key}.png"
            if not template_path.exists():
                continue
            image = read_image(template_path, cv2.IMREAD_UNCHANGED)
            if image is None:
                continue
            h, w = image.shape[:2]
            # Match only the badge center, avoiding the outer ring, division
            # text and animated background.
            center = image[round(h * 0.27) : round(h * 0.67), round(w * 0.28) : round(w * 0.72), :3]
            if center.size == 0:
                continue
            self.native_centers[key] = center.copy()
            scaled = [
                cv2.resize(center, (round(center.shape[1] / center.shape[0] * size), size))
                for size in range(11, 31)
            ]
            self.templates.append(_RankTemplates(key=key, center_image=center, scaled_templates=scaled))

    def inspect(self, image: np.ndarray, profile: dict[str, Any] | None = None) -> RankInfo | None:
        h, w = image.shape[:2]
        if profile and profile.get("rank_roi"):
            rw, rh = profile["reference_size"]
            x1, y1, x2, y2 = profile["rank_roi"]
            roi = image[round(y1 * h / rh) : round(y2 * h / rh), round(x1 * w / rw) : round(x2 * w / rw)]
        else:
            if abs(w / h - 2560 / 1080) > 0.03:
                return None
            roi = image[: round(h * 0.091), round(w * 0.953) :]
        if not roi.size:
            return None
        native = roi

        confirmed = bool(profile) and profile.get("layout_status") == "confirmed"
        clock = mode_clock_evidence(native) if confirmed else None
        if clock is not None:
            return RankInfo(
                tier=None, name=None, division=None,
                confidence=clock["score"], method="mode_clock_detected", mode="other",
            )
        absence = empty_badge_region(native) if confirmed else None
        if absence is None and confirmed and profile.get("ui_layout") != "compact":
            absence = shifted_squad_panel(native)
        if absence is not None:
            return RankInfo(
                tier="unranked", name=RANK_NAMES["unranked"], division=None,
                confidence=0.8, method="no_badge_matching_rule", mode=None,
            )

        if profile and profile.get("hud_scale"):
            sx, sy = profile["hud_scale"]
            rw, rh = profile["reference_size"]
            roi = cv2.resize(roi, None, fx=0.5 / sx * rw / w, fy=0.5 / sy * rh / h)
        else:
            roi = cv2.resize(roi, (60, 49))

        scores: list[tuple[float, str, tuple[int, int] | None, tuple[int, int] | None]] = []
        for tmpl in self.templates:
            best: tuple[float, tuple[int, int] | None, tuple[int, int] | None] = (-1.0, None, None)
            for template in tmpl.scaled_templates:
                th, tw = template.shape[:2]
                if th > roi.shape[0] or tw > roi.shape[1]:
                    continue
                _, score, _, point = cv2.minMaxLoc(cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED))
                x, y = point
                if not palette_matches(template, roi[y : y + th, x : x + tw]):
                    continue
                if score > best[0]:
                    best = (score, (x, y), (tw, th))
            scores.append((best[0], tmpl.key, best[1], best[2]))
        scores.sort(reverse=True)
        if len(scores) < 2:
            return None
        if scores[0][0] < 0.80 or scores[0][0] - scores[1][0] < 0.12:
            fallback = self.inspect_native(native, (roi.shape[1] * 2, roi.shape[0] * 2))
            return fallback if scores[0][0] >= 0.70 and fallback is not None else None

        score, key, point, size = scores[0]
        division = None
        if key not in ("rookie", "master", "predator", "unranked") and point is not None and size is not None:
            x, y = point
            tw, th = size
            cx = x + tw / 2
            # Roman numerals must be read at native pixel scale — downscaling
            # collapses the thin strokes.
            sx = native.shape[1] / roi.shape[1]
            sy = native.shape[0] / roi.shape[0]
            patch = native[
                round((y + th + 2) * sy) : min(native.shape[0], round((y + th + 12) * sy)),
                max(0, round((cx - 7) * sx)) : min(native.shape[1], round((cx + 7) * sx)),
            ]
            division = read_division(patch)
            if division is None:
                refined = self.inspect_native(native, (roi.shape[1] * 2, roi.shape[0] * 2))
                if refined and refined.tier == key:
                    division = refined.division

        return RankInfo(
            tier=key, name=RANK_NAMES.get(key, key), division=division,
            confidence=round(float(score), 3), method="hud_badge_template", mode=None,
        )

    def inspect_native(self, native: np.ndarray, size: tuple[int, int]) -> RankInfo | None:
        """Re-check at native resolution when the downscaled pass was marginal."""
        roi = cv2.resize(native, size)
        scores: list[tuple[float, str, tuple[int, int] | None, tuple[int, int] | None]] = []
        for key, center in self.native_centers.items():
            best: tuple[float, tuple[int, int] | None, tuple[int, int] | None] = (-1.0, None, None)
            for height in range(22, 61):
                template = cv2.resize(center, (round(center.shape[1] / center.shape[0] * height), height))
                th, tw = template.shape[:2]
                if th > roi.shape[0] or tw > roi.shape[1]:
                    continue
                _, score, _, (x, y) = cv2.minMaxLoc(cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED))
                if score > best[0] and palette_matches(template, roi[y : y + th, x : x + tw]):
                    best = (score, (x, y), (tw, th))
            scores.append((best[0], key, best[1], best[2]))
        scores.sort(reverse=True)
        if len(scores) < 2 or scores[0][0] < 0.80 or scores[0][0] - scores[1][0] < 0.12:
            return None
        score, key, point, size_ = scores[0]
        division = None
        if key not in ("rookie", "master", "predator", "unranked") and point is not None and size_ is not None:
            x, y = point
            tw, th = size_
            sx = native.shape[1] / roi.shape[1]
            sy = native.shape[0] / roi.shape[0]
            cx = x + tw / 2
            patch = native[
                round((y + th + 4) * sy) : min(native.shape[0], round((y + th + 24) * sy)),
                max(0, round((cx - 14) * sx)) : min(native.shape[1], round((cx + 14) * sx)),
            ]
            division = read_division(patch)
        return RankInfo(
            tier=key, name=RANK_NAMES.get(key, key), division=division,
            confidence=round(float(score), 3), method="hud_badge_native_template", mode=None,
        )


def read_division(patch: np.ndarray) -> str | None:
    """Accept only clearly read Roman divisions; no inference from cup rims or
    blurry text."""
    if not patch.size:
        return None
    if patch.shape[0] < 16:
        patch = cv2.resize(patch, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
    readings = []
    for threshold in [120, 155, 190]:
        value = _division_components((np.min(patch, 2) > threshold).astype(np.uint8))
        if value:
            readings.append(value)
    counts = Counter(readings)
    return next(iter(counts)) if len(counts) == 1 and next(iter(counts.values())) >= 2 else None


def _division_components(mask: np.ndarray) -> str | None:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    components = []
    glyph_boxes = []
    fragment_mask = np.zeros_like(mask)
    for label, (x, y, w, h, area) in enumerate(stats[1:], 1):
        if x == 0 or x + w == mask.shape[1] or y == 0:
            continue
        if h < 5 or h > mask.shape[0] * 0.8 or w > mask.shape[1] * 0.8 or area < 8:
            if h >= 2 and w <= max(3, h * 0.65) and area >= 3:
                fragment_mask[labels == label] = 1
            continue
        glyph_boxes.append((x, y, w, h))
        glyph = mask[y : y + h, x : x + w]
        trim = max(1, round(h * 0.23))
        body = glyph[trim : h - trim]
        spans = []
        for line in body:
            points = np.flatnonzero(line)
            if len(points):
                spans.append((points[0], points[-1]))
        if not spans:
            return None
        if w <= h * 1.15 and w >= 3 and spans[0][1] - spans[0][0] >= spans[-1][1] - spans[-1][0] + 2:
            components.append((x, "V"))
            continue
        # Serifs can connect II/III at the top; split only in the stem body.
        _, _, parts, _ = cv2.connectedComponentsWithStats(body)
        letters = []
        for px, py, pw, ph, pa in parts[1:]:
            if ph < body.shape[0] * 0.7 or pa < 3:
                return None
            piece = body[py : py + ph, px : px + pw]
            centers = [np.flatnonzero(line).mean() for line in piece if line.any()]
            if pw <= max(3, h * 0.45) and max(centers) - min(centers) <= max(1, h * 0.13):
                letters.append((px, "I"))
            else:
                widths = [np.ptp(np.flatnonzero(line)) for line in piece if line.any()]
                if pw >= 3 and widths[0] >= widths[-1] + 1:
                    letters.append((px, "V"))
                else:
                    return None
        components.extend((x + px, letter) for px, letter in letters)
    # High thresholds can cut dark stems; fragments next to a glyph invalidate
    # that binarization instead of letting III read as I.
    fragments = cv2.connectedComponentsWithStats(
        cv2.morphologyEx(fragment_mask, cv2.MORPH_CLOSE, np.ones((3, 1), np.uint8))
    )[2][1:]
    for x, y, w, h, area in fragments:
        for gx, gy, gw, gh in glyph_boxes:
            overlap = max(0, min(y + h, gy + gh) - max(y, gy))
            gap = max(gx - (x + w), x - (gx + gw), 0)
            if h >= gh * 0.45 and overlap >= h * 0.7 and gap <= gh:
                return None
    text = "".join(v for _, v in sorted(components))
    return text if text in ["I", "II", "III", "IV"] else None
