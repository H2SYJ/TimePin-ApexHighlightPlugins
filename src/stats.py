"""Combat statistics extraction from the counter strip.

Main path only (per TPIN-10): locate the counter panel via the damage-marker
icon, crop the digit regions, and decode with the hud_digits CNN+CTC model.
Values that fail to read stay None — no fallback cascade.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .counter import counter_box, counter_offset, digit_span, numeric, translate
from .extra import CombatStats
from .hud_layout import read_image, template_for

RESOURCES = Path(__file__).resolve().parent.parent / "resources"
DIGITS_MODEL = RESOURCES / "hud_digits_v1.onnx"

# HUD counter cannot legitimately exceed these; larger reads are OCR noise.
LIMITS = {"damage": 99999, "kills": 99, "assists": 99, "participation": 99}


class StatCounter:
    """Icon-anchored digit crops for the counter strip (kills/assists/damage)."""

    def __init__(self) -> None:
        root = RESOURCES / "templates" / "statistics"
        self.templates = {
            key: read_image(root / f"{key}-gray.png", cv2.IMREAD_GRAYSCALE)
            for key in ("kills", "assists", "participation")
        }
        crown = read_image(root / "kills-crown-gray.png", cv2.IMREAD_GRAYSCALE)
        self.extra_templates = {"kills": [crown]} if crown is not None else {}

    def regions(
        self, image: np.ndarray, profile: dict[str, Any], delta: tuple[int, int]
    ) -> dict[str, np.ndarray]:
        """Per-stat digit crops anchored to their icons; damage uses the fixed
        box shifted by the counter offset."""
        rw, rh = profile["reference_size"]
        fx, fy = image.shape[1] / rw, image.shape[0] / rh
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        dx, dy = delta
        anchors: dict[str, list[int]] = {}
        anchor_scores: dict[str, float] = {}
        sx, sy = profile.get("hud_scale", [1.0, 1.0])
        origin = profile["counter_marker_origin"]
        marker_x = round((origin[0] + dx) * fx)

        for key, reference in (
            ("kills", profile["kills"]),
            ("assists", profile.get("assists", [2202, 94, 2228, 124])),
            ("participation", profile.get("participation", [2260, 94, 2300, 124])),
        ):
            # Icons must follow the counter marker; the numeric box alone is not
            # reliable because participation can be absent on some HUDs.
            x1 = max(0, marker_x - round(300 * sx * fx))
            x2 = min(image.shape[1], marker_x)
            y1 = max(0, round((reference[1] + dy) * fy) - 8)
            y2 = min(image.shape[0], round((reference[3] + dy) * fy) + 8)
            region = gray[y1:y2, x1:x2]
            for template in [self.templates[key], *self.extra_templates.get(key, [])]:
                if template is None:
                    continue
                template = template_for(profile, template)
                for factor in (0.75, 1.0, 1.25):
                    scaled = cv2.resize(template, None, fx=fx * 2 * factor, fy=fy * 2 * factor)
                    if region.shape[0] < scaled.shape[0] or region.shape[1] < scaled.shape[1]:
                        continue
                    _, score, _, point = cv2.minMaxLoc(cv2.matchTemplate(region, scaled, cv2.TM_CCOEFF_NORMED))
                    if score >= 0.84 and score > anchor_scores.get(key, 0):
                        anchor_scores[key] = score
                        anchors[key] = [x1 + point[0], y1 + point[1], x1 + point[0] + scaled.shape[1], y1 + point[1] + scaled.shape[0]]

        result: dict[str, np.ndarray] = {}
        for key in ("kills", "assists"):
            if key not in anchors:
                continue
            a = anchors[key]
            next_x = min([point[0] for name, point in anchors.items() if name != key and point[0] > a[2]] + [marker_x])
            padding = max(1, round(sx * fx))
            left = a[2]
            right = min(next_x - padding, left + round(40 * sx * fx))
            if right - left < 3:
                continue
            top = max(0, a[1] - round(2 * sy * fy))
            bottom = min(image.shape[0], a[3] + round(2 * sy * fy))
            # Icon match edges shift with scaling: grow from the icon center to
            # the real digit glyphs instead of trusting the template's edge.
            start = (a[0] + a[2]) // 2
            end = min(next_x - padding, a[2] + round(48 * sx * fx))
            span = digit_span(image[top:bottom, start:end]) if end - start >= 3 else None
            if span:
                left, right = start + span[0], start + span[1]
            result[key] = image[top:bottom, left:right].copy()

        box = counter_box(profile, "damage")
        x1, y1, x2, y2 = [
            round((v + (dx if i % 2 == 0 else dy)) * (fx if i % 2 == 0 else fy))
            for i, v in enumerate(box)
        ]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
        if x2 - x1 >= 3 and y2 - y1 >= 3:
            result["damage"] = image[y1:y2, x1:x2].copy()
        return result


class StatReader:
    """Decode counter crops with the hud_digits CNN+CTC model.

    `decoder` is any callable(np.ndarray) -> (text, confidence); wire to
    ``timepin_vision.OnnxDigitDecoder`` for the bundled hud_digits_v1 model.
    """

    def __init__(self, decoder: Any, counter: StatCounter | None = None) -> None:
        self.decoder = decoder
        self.counter = counter or StatCounter()

    def decode(self, patch: np.ndarray) -> tuple[str, float] | None:
        try:
            result = self.decoder(patch)
        except Exception:
            return None
        if isinstance(result, (tuple, list)) and len(result) >= 2:
            text, conf = result[0], float(result[1])
        elif isinstance(result, str):
            text, conf = result, 1.0
        else:
            return None
        return (str(text), conf) if text else None

    def read_frame(
        self, image: np.ndarray, profile: dict[str, Any], delta: tuple[int, int] | None
    ) -> CombatStats:
        """Read the counter strip on one frame; delta from counter_offset."""
        stats = CombatStats()
        if delta is None:
            return stats
        regions = self.counter.regions(image, profile, delta)
        for key in ("kills", "assists", "damage"):
            patch = regions.get(key)
            if patch is None or not patch.size:
                continue
            decoded = self.decode(patch)
            if decoded is None:
                continue
            value = numeric(decoded[0], decoded[1], LIMITS.get(key, 99))
            if value is None:
                continue
            setattr(stats, key, value)
        return stats


def find_counter_delta(
    frames: list[np.ndarray], profile: dict[str, Any]
) -> tuple[tuple[int, int] | None, float]:
    """Best counter-marker offset across the given frames."""
    marker = read_image(RESOURCES / profile["counter_marker"], cv2.IMREAD_GRAYSCALE)
    if marker is None:
        return None, 0.0
    best_delta: tuple[int, int] | None = None
    best_score = 0.0
    for image in frames:
        try:
            delta, score = counter_offset(image, profile, marker)
        except cv2.error:
            continue
        if score > best_score:
            best_delta, best_score = delta, float(score)
    return best_delta, best_score
