"""Counter-panel helpers: anchor matching and numeric HUD crops."""

from __future__ import annotations

import re
from typing import Any

import cv2
import numpy as np

from .hud_layout import template_for


def crop(frame: np.ndarray, box: list[int], reference: list[int]) -> np.ndarray:
    """Crop a profile box out of `frame` (which may differ from reference size)."""
    h, w = frame.shape[:2]
    rw, rh = reference
    x1, y1, x2, y2 = box
    region = frame[round(y1 * h / rh) : round(y2 * h / rh), round(x1 * w / rw) : round(x2 * w / rw)]
    if not region.size:
        raise ValueError("HUD region is empty")
    return cv2.resize(region, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)


def numeric(text: str, score: float, limit: int, minimum: float = 0.88) -> int | None:
    text = text.strip()
    if score < minimum or not re.fullmatch(r"\d+", text):
        return None
    value = int(text)
    return value if value <= limit else None


def counter_offset(
    frame: np.ndarray, profile: dict[str, Any], template: np.ndarray
) -> tuple[tuple[int, int] | None, float]:
    """Locate the counter icon; Apex shifts the counter HUD with animations and
    digit widths, so the icon is the anchor. Returns (dx, dy) offset vs origin."""
    template = template_for(profile, template)
    ref = cv2.resize(frame, (profile["reference_size"][0] // 2, profile["reference_size"][1] // 2))
    x1, y1, x2, y2 = [round(v / 2) for v in profile["counter_marker_search"]]
    region = cv2.cvtColor(ref[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    _, score, _, location = cv2.minMaxLoc(cv2.matchTemplate(region, template, cv2.TM_CCOEFF_NORMED))
    if score < 0.78:
        return None, score
    origin = profile["counter_marker_origin"]
    return ((location[0] + x1) * 2 - origin[0], (location[1] + y1) * 2 - origin[1]), score


def inactive_view(frame: np.ndarray) -> bool:
    """Spectator and post-match panels show a bottom black control bar; this
    check blocks treating another player's HUD signals as the player's own."""
    h, w = frame.shape[:2]
    bottom = frame[round(h * 0.985) :, round(w * 0.12) : round(w * 0.88)]
    return float(np.mean(np.max(bottom, axis=2) < 14)) > 0.72


def translate(box: list[int], dx: int, dy: int) -> list[int]:
    return [box[0] + dx, box[1] + dy, box[2] + dx, box[3] + dy]


def counter_box(profile: dict[str, Any], key: str) -> list[int]:
    box = list(profile[key])
    if key == "damage":
        sx = profile.get("hud_scale", [1, 1])[0]
        # Four-digit damage does not fit the original three-digit crop width;
        # keep the gap to the rank badge.
        box[2] = max(
            box[2],
            min(
                box[2] + round(24 * sx),
                profile.get("rank_roi", [profile["reference_size"][0]])[0] - round(8 * sx),
            ),
        )
    return box


def without_counter_divider(image: np.ndarray) -> np.ndarray | None:
    """Remove only the right diagonal separator that spans the crop height;
    shorter digit strokes are kept."""
    if image is None or not image.size:
        return None
    height, width = image.shape[:2]
    edges = cv2.Canny(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 80, 180)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180,
        threshold=max(8, round(height * 0.3)),
        minLineLength=round(height * 0.75),
        maxLineGap=round(height * 0.08),
    )
    candidates = []
    for x1, y1, x2, y2 in [] if lines is None else lines.reshape(-1, 4):
        dy = abs(int(y2) - int(y1))
        dx = abs(int(x2) - int(x1))
        if (
            dy >= height * 0.75
            and 0.2 * dy <= dx <= 0.8 * dy
            and min(y1, y2) <= height * 0.25
            and max(y1, y2) >= height * 0.85
            and min(x1, x2) > width * 0.35
        ):
            candidates.append(int(min(x1, x2)))
    return image[:, : max(1, min(candidates) - 2)].copy() if candidates else None


def digit_view(patch: np.ndarray) -> np.ndarray | None:
    """Foreground-separated view of a digit crop for recognition."""
    if patch is None or not patch.size:
        return None
    clean = without_counter_divider(patch)
    if clean is not None:
        patch = clean
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    count, labels, stats, _ = cv2.connectedComponentsWithStats((gray > 180).astype(np.uint8))
    mask = np.zeros(gray.shape, np.uint8)
    for i, (x, y, w, h, area) in enumerate(stats[1:], 1):
        # Exclude full-height white background, bottom edge and icon fragments;
        # keep independent digit strokes.
        if 4 <= h <= gray.shape[0] - 2 and area >= 4 and w <= h * 1.6:
            mask[labels == i] = 255
    yy, xx = np.where(mask > 0)
    if not len(xx):
        return None
    glyph = 255 - mask[yy.min() : yy.max() + 1, xx.min() : xx.max() + 1]
    glyph = cv2.resize(glyph, None, fx=4, fy=4)
    glyph = cv2.copyMakeBorder(glyph, 8, 8, 8, 8, cv2.BORDER_CONSTANT, value=255)
    return cv2.cvtColor(glyph, cv2.COLOR_GRAY2BGR)


def digit_span(patch: np.ndarray) -> tuple[int, int] | None:
    """Left/right bounds of counter digits in the window; skips icon edges
    touching the left, the '/' separator and right-side white blocks."""
    if patch is None or not patch.size:
        return None
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    level = max(150, int(np.percentile(gray, 99) * 0.75))
    _, labels, stats, _ = cv2.connectedComponentsWithStats((gray >= level).astype(np.uint8), connectivity=8)
    glyphs = []
    others = []
    for index, (x, y, w, h, area) in enumerate(stats[1:], 1):
        if area < 3:
            continue
        digit = x > 0 and 0.4 * height <= h <= 0.95 * height and w <= h * 0.9 and not (
            area / (w * h) < 0.38 and w >= h * 0.25
        )
        if digit:
            ys, xs = np.where(labels[y : y + h, x : x + w] == index)
            top = xs[ys < h * 0.3]
            bottom = xs[ys > h * 0.7]
            # '/' leans right at the top; '1' is basically vertical.
            digit = not (len(top) and len(bottom) and top.mean() - bottom.mean() > max(1.5, h * 0.18))
        (glyphs if digit else others).append((x, x + w, h))
    if not glyphs:
        return None
    glyphs.sort()
    first = glyphs[0]
    right = first[1]
    for x, x2, h in glyphs[1:]:
        if x - right > first[2] * 0.5 or abs(h - first[2]) > first[2] * 0.25:
            break
        right = x2
    pad = max(1, round(first[2] * 0.25))
    left_wall = max([x2 for x, x2, _ in others if x2 <= first[0]], default=0)
    right_wall = min([x for x, _, _ in others if x >= right], default=width)
    return max(left_wall, first[0] - pad), min(right_wall, right + pad)
