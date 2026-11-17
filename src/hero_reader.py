"""Hero recognition from ability icons in self HUD."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any
import json
import threading

import cv2
import numpy as np

from .extra import HeroInfo, LEGENDS_DIR


HERO_NAMES = {
    "bloodhound": "寻血猎犬",
    "gibraltar": "直布罗陀",
    "lifeline": "命脉",
    "pathfinder": "探路者",
    "wraith": "恶灵",
    "bangalore": "班加罗尔",
    "caustic": "侵蚀",
    "mirage": "幻象",
    "octane": "动力小子",
    "wattson": "沃特森",
    "crypto": "密客",
    "revenant": "亡灵",
    "loba": "罗芭",
    "rampart": "兰伯特",
    "horizon": "地平线",
    "fuse": "暴雷",
    "valkyrie": "瓦尔基里",
    "seer": "希尔",
    "ash": "艾许",
    "mad-maggie": "疯玛吉",
    "newcastle": "纽卡斯尔",
    "vantage": "万蒂奇",
    "catalyst": "催化姬",
    "ballistic": "弹道",
    "conduit": "导管",
    "alter": "变幻",
    "sparrow": "Sparrow",
    "axle": "艾克赛尔",
}


_TEMPLATE_LOCK = threading.RLock()


@dataclass
class HeroTemplate:
    hero_id: str
    slot: str  # 'tactical' or 'ultimate'
    glyph: np.ndarray
    is_contrast: bool


@lru_cache(maxsize=2)
def _load_template_bank(signature: tuple) -> tuple[list[HeroTemplate], dict]:
    """Load hero ability glyph templates from assets."""
    templates = []
    contrast_templates = []

    # Load from legends abilities
    data_path = LEGENDS_DIR / "sources.json"
    if data_path.exists():
        data = json.loads(data_path.read_text(encoding="utf-8"))
        for hero in data.get("legends", []):
            for kind, asset in hero.get("abilities", {}).items():
                image_path = LEGENDS_DIR / asset["file"]
                if not image_path.exists():
                    continue
                image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
                if image is None:
                    continue
                mask = image[:, :, 3] if image.shape[2] == 4 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                ys, xs = np.where(mask > 127)
                if not len(xs):
                    continue
                glyph = (mask[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1] > 127).astype(np.uint8) * 255
                templates.append(HeroTemplate(hero_id=hero["id"], slot=kind, glyph=glyph, is_contrast=False))

    # Load contrast templates (for ultimate charge states)
    hud_data_path = LEGENDS_DIR / "hud" / "sources.json"
    if hud_data_path.exists():
        hud_data = json.loads(hud_data_path.read_text(encoding="utf-8"))
        for item in hud_data.get("templates", []):
            glyph_path = hud_data_path.parent / item["file"]
            if not glyph_path.exists():
                continue
            glyph = cv2.imread(str(glyph_path), cv2.IMREAD_GRAYSCALE)
            if glyph is not None:
                contrast_templates.append(HeroTemplate(hero_id=item["id"], slot=item["slot"], glyph=glyph, is_contrast=True))

    return templates, contrast_templates


class HeroReader:
    def __init__(self) -> None:
        root = LEGENDS_DIR
        signature = tuple(
            (str(path), path.stat().st_size, path.stat().st_mtime_ns)
            for path in sorted(root.rglob("*"))
            if path.is_file() and path.suffix.lower() in {".png", ".json"}
        )
        self.templates, self.contrast_templates = _load_template_bank(signature)
        self._scaled_cache: dict[Any, list] = {}
        self._matches_cache: dict[Any, dict] = {}
        self.last_scores: list[tuple[str, float]] = []
        self.last_slots: dict[str, dict[str, float]] = {}

    def _prepare_templates(
        self, kind: str, image_shape: tuple[int, int], profile: dict[str, Any], roi_shape: tuple[int, int]
    ) -> list[tuple[str, bool, list[tuple[np.ndarray, tuple[int, int, int, int]]]]]:
        """Prepare scaled templates for matching."""
        with _TEMPLATE_LOCK:
            return self._prepare_templates_locked(kind, image_shape, profile, roi_shape)

    def _prepare_templates_locked(
        self, kind: str, image_shape: tuple[int, int], profile: dict[str, Any], roi_shape: tuple[int, int]
    ) -> list[tuple[str, bool, list[tuple[np.ndarray, tuple[int, int, int, int]]]]]:
        rw, rh = profile["reference_size"]
        sx, sy = profile.get("hud_scale", [1.0, 1.0])
        key = (kind, tuple(image_shape[:2]), rw, rh, sx, sy, tuple(roi_shape[:2]))

        if key not in self._scaled_cache:
            prepared = []
            all_templates = [(t.hero_id, t.slot, t.glyph, t.is_contrast) for t in self.templates] + [
                (t.hero_id, t.slot, t.glyph, t.is_contrast) for t in self.contrast_templates
            ]

            for hero_id, slot, glyph, is_contrast in all_templates:
                if slot != kind:
                    continue
                variants = []
                sizes = set()
                # Different height ranges for contrast vs regular
                if is_contrast and kind == "ultimate":
                    height_range = range(20, 101)
                elif is_contrast:
                    height_range = range(20, 81)
                else:
                    height_range = range(20, 69, 3)

                for height in height_range:
                    h = max(4, round(height * sy * image_shape[0] / rh))
                    w = max(4, round(height * glyph.shape[1] / glyph.shape[0] * sx * image_shape[1] / rw))
                    if h >= roi_shape[0] or w >= roi_shape[1] or (w, h) in sizes:
                        continue
                    sizes.add((w, h))
                    # Center search region
                    x1 = max(0, int(np.ceil(roi_shape[1] * 0.4 - w / 2)))
                    x2 = min(roi_shape[1] - w + 1, int(np.floor(roi_shape[1] * 0.6 - w / 2)) + 1)
                    y1 = max(0, int(np.ceil(roi_shape[0] * 0.3 - h / 2)))
                    y2 = min(roi_shape[0] - h + 1, int(np.floor(roi_shape[0] * 0.7 - h / 2)) + 1)
                    if x1 >= x2 or y1 >= y2:
                        continue
                    variants.append((cv2.resize(glyph, (w, h), interpolation=cv2.INTER_AREA), (x1, x2, y1, y2)))

                if variants:
                    prepared.append((hero_id, is_contrast, variants))

            self._scaled_cache[key] = prepared
            # LRU eviction
            while len(self._scaled_cache) > 4:
                self._scaled_cache.pop(next(iter(self._scaled_cache)))

        self._scaled_cache.move_to_end(key)
        return self._scaled_cache[key]

    def inspect(self, image: np.ndarray, profile: dict[str, Any]) -> HeroInfo | None:
        """Inspect image for hero from tactical and ultimate ability icons."""
        rw, rh = profile["reference_size"]
        sx, sy = profile.get("hud_scale", [1.0, 1.0])

        scores: dict[str, float] = {}
        slot_scores: dict[str, dict[str, float]] = {}
        native_scores: dict[str, float] = {}

        for kind in ["tactical", "ultimate"]:
            if f"hero_{kind}_roi" not in profile:
                continue
            x1, y1, x2, y2 = profile[f"hero_{kind}_roi"]
            roi = image[
                round(y1 * image.shape[0] / rh) : round(y2 * image.shape[0] / rh),
                round(x1 * image.shape[1] / rw) : round(x2 * image.shape[1] / rw),
            ]
            if not roi.size:
                continue

            # White HUD glyphs mask
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            masks = [((hsv[:, :, 1] < 70) & (hsv[:, :, 2] > level)).astype(np.uint8) * 255 for level in [155, 85]]
            contrast_masks = list(masks)

            if any(t.slot == kind for t in self.contrast_templates):
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).astype(np.float32)
                difference = np.maximum(0, gray - np.median(gray, axis=1)[:, None])
                contrast_masks.append(
                    np.where(hsv[:, :, 1] < 70, np.clip((difference - 4) * 8, 0, 255), 0).astype(np.uint8)
                )

            # Deduplicate masks
            unique_masks = []
            unique_contrast = []
            for source, target in [(masks, unique_masks), (contrast_masks, unique_contrast)]:
                for mask in source:
                    if not any(np.array_equal(mask, other) for other in target):
                        target.append(mask)

            prepared = self._prepare_templates(kind, image.shape, profile, roi.shape)

            # Check cache
            cache_key = (
                kind,
                tuple(image.shape),
                tuple(roi.shape),
                tuple(profile["reference_size"]),
                tuple(profile.get("hud_scale", [1.0, 1.0])),
                tuple(mask.tobytes() for mask in unique_masks),
                tuple(mask.tobytes() for mask in unique_contrast),
            )
            cached = self._matches_cache.get(cache_key)
            if cached is not None:
                for hero_id, (best, native) in cached.items():
                    scores[hero_id] = max(scores.get(hero_id, 0.0), best)
                    slot_scores.setdefault(hero_id, {})[kind] = best
                    if native is not None:
                        native_scores[hero_id] = max(native_scores.get(hero_id, 0.0), native)
                self._matches_cache.move_to_end(cache_key)
                continue

            matched = {}
            for hero_id, is_contrast, variants in prepared:
                best = 0.0
                for template, (x1, x2, y1, y2) in variants:
                    for mask in unique_contrast if is_contrast else unique_masks:
                        response = cv2.matchTemplate(mask, template, cv2.TM_CCOEFF_NORMED)
                        score = float(cv2.minMaxLoc(response[y1:y2, x1:x2])[1])
                        best = max(best, score)

                scores[hero_id] = max(scores.get(hero_id, 0.0), best)
                slot_scores.setdefault(hero_id, {})[kind] = max(
                    slot_scores.get(hero_id, {}).get(kind, 0.0), best
                )
                if is_contrast:
                    native_scores[hero_id] = max(native_scores.get(hero_id, 0.0), best)

                previous = matched.get(hero_id, (0.0, None))
                matched[hero_id] = (
                    max(previous[0], best),
                    max(previous[1] or 0.0, best) if is_contrast else previous[1],
                )

            self._matches_cache[cache_key] = matched
            while len(self._matches_cache) > 8:
                self._matches_cache.popitem(last=False)

        # Sort by score
        raw_ordered = sorted(scores.items(), key=lambda v: -v[1])
        self.last_scores = raw_ordered[:5]
        self.last_slots = slot_scores

        # Filter candidates
        filtered_scores = {
            hero_id: score
            for hero_id, score in scores.items()
            if native_scores.get(hero_id, 0.0) >= 0.78
            or min(slot_scores[hero_id].get(k, 0.0) for k in ["tactical", "ultimate"]) >= 0.65
            or score >= 0.90
            and score - max((other for key, other in scores.items() if key != hero_id), default=0) >= 0.08
        }

        ordered = sorted(filtered_scores.items(), key=lambda v: -v[1])
        if not ordered or ordered[0][1] < 0.70:
            return None

        margin = 0.025 if ordered[0][1] >= 0.80 else 0.055
        if len(ordered) > 1 and ordered[0][1] - ordered[1][1] < margin:
            return None

        hero_id, score = ordered[0]
        return HeroInfo(
            id=hero_id,
            name=HERO_NAMES.get(hero_id, hero_id.title()),
            confidence=round(score, 3),
            method="self_ability_glyph",
        )