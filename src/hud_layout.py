"""HUD layout detection: anchored box adaptation across resolutions and UI modes.

Ported from apex-highlight-clipper hud_layout.py (anchored-hud-v2). Standard and
compact (游戏「界面配置模式：精简」) UI layouts, native and stretched renders,
plus pillarboxed 4:3 content recorded inside widescreen files.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

BASE = Path(__file__).resolve().parent.parent / "resources"
VERSION = "anchored-hud-v2"

# Reference coordinates 2560x1080; each box anchored to its screen edge.
STANDARD_LAYOUT: dict[str, Any] = {
    "assists": [2202, 94, 2228, 124],
    "participation": [2260, 94, 2300, 124],
    "rank_roi": [2440, 0, 2560, 108],
    "account_roi": [175, 958, 337, 989],
    "teammate_rois": [[136, 780, 292, 808], [136, 845, 292, 873]],
    "hero_tactical_roi": [555, 967, 653, 1035],
    "hero_ultimate_roi": [1243, 928, 1323, 1028],
}

# Compact UI: HUD scaled ~1.25x, counter strip merged; boxes measured on a real
# 1280x720 (4:3 stretched) recording, expressed in enlarged reference units.
COMPACT_SCALE = 1.25
COMPACT_LAYOUT: dict[str, Any] = {
    "counter_marker_origin": [2313, 134],
    "damage": [2347, 134, 2387, 162],
    "kills": [2047, 132, 2083, 162],
    "assists": [2104, 130, 2134, 164],
    "participation": [2160, 132, 2200, 162],
    "ammo": [2385, 955, 2465, 1010],
    "weapon": [2350, 1046, 2510, 1078],
    "weapon_icon": [2215, 955, 2380, 1003],
    "weapon_name_boxes": [[2185, 1046, 2345, 1078], [2350, 1046, 2510, 1078]],
    "weapon_name_metrics": [[2200, 1048, 2330, 1078], [2365, 1048, 2495, 1078]],
    "rank_roi": [2410, 10, 2530, 170],
    "account_roi": [133, 986, 296, 1020],
    "teammate_rois": [[124, 798, 280, 828], [124, 876, 280, 906]],
    "hero_tactical_roi": [555, 967, 653, 1035],
    "hero_ultimate_roi": [1228, 930, 1335, 1035],
}

# Common stretch renders seen in recorded footage. HUD sits in the 16:9 safe area.
STRETCH_RENDERS = ((16, 9), (16, 10), (4, 3))
DETECTOR_VERSION = "fit-scale-candidates-v1"


def read_image(path: str | Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray | None:
    """Unicode-safe image read (cv2.imread fails on non-ASCII paths)."""
    try:
        data = Path(path).read_bytes()
        return cv2.imdecode(np.frombuffer(data, dtype=np.uint8), flags) if data else None
    except (OSError, cv2.error):
        return None


def point(
    profile: dict[str, Any],
    x: float,
    y: float,
    horizontal: str = "right",
    vertical: str = "top",
) -> list[int]:
    left, top, width, height = profile.get("content_box") or [0, 0, *profile["reference_size"]]
    sx, sy = profile.get("hud_scale", [height / 1080, height / 1080])
    px = x * sx if horizontal == "left" else width - (2560 - x) * sx if horizontal == "right" else width / 2 + (x - 1280) * sx
    py = y * sy if vertical == "top" else height - (1080 - y) * sy if vertical == "bottom" else height / 2 + (y - 540) * sy
    return [round(left + px), round(top + py)]


def pillarbox_content(images: Sequence[np.ndarray], width: int, height: int) -> list[int] | None:
    """Symmetric unlit side bars -> actual game area [x, y, w, h]; None if none."""
    lit = None
    for image in images:
        if image.shape[:2] != (height, width):
            continue
        bright = image.max(axis=2) > 24
        if bright.mean() < 0.2:
            continue  # loading black screens don't prove boundaries
        columns = bright.mean(axis=0)
        lit = columns if lit is None else np.maximum(lit, columns)
    if lit is None:
        return None
    used = np.nonzero(lit > 0.01)[0]
    if not len(used):
        return None
    x1, x2 = int(used[0]), int(used[-1]) + 1
    if x1 < width * 0.03 or abs(x1 - (width - x2)) > max(4, width * 0.01):
        return None
    return [x1, 0, x2 - x1, height]


def box(profile: dict[str, Any], values: Sequence[float], horizontal: str = "right", vertical: str = "top") -> list[int]:
    return point(profile, *values[:2], horizontal, vertical) + point(profile, *values[2:], horizontal, vertical)


def load_base(base: Path | None = None) -> dict[str, Any]:
    return json.loads((base or BASE / "profile-ultrawide.json").read_text("utf-8"))


def make_profile(
    width: int,
    height: int,
    vertical_scale: float = 1.0,
    horizontal_scale: float = 1.0,
    base: dict[str, Any] | None = None,
    ui: str = "standard",
    layout: str | None = None,
    content_box: Sequence[int] | None = None,
) -> dict[str, Any]:
    original = base or load_base()
    compact = ui == "compact"
    source = {**original, **STANDARD_LAYOUT, **(COMPACT_LAYOUT if compact else {})}
    k = COMPACT_SCALE if compact else 1.0
    profile = deepcopy(original)
    profile["reference_size"] = [width, height]
    if content_box:
        profile["content_box"] = list(content_box)
    scale_height = content_box[3] if content_box else height
    profile["hud_scale"] = [scale_height / 1080 * horizontal_scale * k, scale_height / 1080 * vertical_scale * k]
    profile["layout_version"] = VERSION
    profile["layout"] = layout or ("4:3-stretched" if abs(vertical_scale - 0.75) < 0.05 else "native")
    if content_box:
        profile["layout"] = "pillarbox/" + profile["layout"]
    if compact:
        profile["ui_layout"] = "compact"
        profile["layout"] += "/compact"
    profile["name"] = f"Auto {width}x{height} / {profile['layout']}"
    for key in ("damage", "kills"):
        profile[key] = box(profile, source[key])
    # Counter panels can move vertically independently of the stretched HUD.
    # Locate the icon first, then counter_offset translates the numeric boxes.
    search = original["counter_marker_search"]
    profile["counter_marker_search"] = box(
        profile, [search[0] - 40, max(0, search[1] - 64), search[2] + 60, search[3] + 80]
    )
    profile["assists"] = box(profile, source["assists"])
    profile["participation"] = box(profile, source["participation"])
    profile["counter_marker_origin"] = point(profile, *source["counter_marker_origin"])
    for key in ("ammo", "weapon", "weapon_icon"):
        profile[key] = box(profile, source[key], vertical="bottom")
    for key in ("weapon_name_boxes", "weapon_name_metrics"):
        profile[key] = [box(profile, b, vertical="bottom") for b in source[key]]
    for key in ("spectator", "downed_marker_search"):
        profile[key] = box(profile, original[key], horizontal="center", vertical="bottom")
    profile["banner"] = box(profile, original["banner"], horizontal="center", vertical="center")
    left, top, inner_w, inner_h = content_box or [0, 0, width, height]
    cx = round((left + inner_w * 0.75) / 2) * 2
    cy = round((top + inner_h * 5 / 6) / 2) * 2
    profile["weapon_native_crop"] = [
        cx,
        cy,
        (round((left + inner_w) / 2) * 2 if content_box else width) - cx,
        height - cy,
    ]
    profile["rank_roi"] = box(profile, source["rank_roi"], vertical="top")
    profile["account_roi"] = box(profile, source["account_roi"], horizontal="left", vertical="bottom")
    profile["teammate_rois"] = [
        box(profile, b, horizontal="left", vertical="bottom") for b in source["teammate_rois"]
    ]
    profile["hero_tactical_roi"] = box(profile, source["hero_tactical_roi"], horizontal="left", vertical="bottom")
    profile["hero_ultimate_roi"] = box(profile, source["hero_ultimate_roi"], horizontal="center", vertical="bottom")
    return profile


def layout_candidates(
    width: int,
    height: int,
    base: dict[str, Any] | None = None,
    content_box: Sequence[int] | None = None,
) -> list[dict[str, Any]]:
    """All plausible HUD geometries: render ratio (native/stretched) x UI mode."""
    candidates: list[dict[str, Any]] = []
    seen: set[tuple[str, float, float]] = set()
    aspect = (content_box[2] / content_box[3]) if content_box else width / height

    def add(vertical: float, horizontal: float, ui: str, layout: str | None = None) -> None:
        profile = make_profile(
            width, height, vertical, horizontal, base=base, ui=ui, layout=layout, content_box=content_box
        )
        key = (ui, round(profile["hud_scale"][0], 3), round(profile["hud_scale"][1], 3))
        if key not in seen:
            seen.add(key)
            candidates.append(profile)

    for ui in ("standard", "compact"):
        add(1.0, 1.0, ui)
        add(0.75, 1.0, ui)
        if aspect < 1.55:
            add(0.75, 0.75, ui)
        for render in [None, *STRETCH_RENDERS]:
            ratio = aspect if render is None else render[0] / render[1]
            if ratio > aspect + 0.01:
                continue  # wider image in a narrower file leaves bars -> not a stretch
            unit = min(1.0, ratio / (16 / 9))
            label = "native" if render is None or abs(ratio - aspect) < 0.01 else f"{render[0]}:{render[1]}-stretched"
            add(unit, unit * aspect / ratio, ui, label)
    return candidates


def template_for(profile: dict[str, Any], template: np.ndarray) -> np.ndarray:
    sx, sy = profile.get("hud_scale", [1.0, 1.0])
    if abs(sx - 1) < 0.001 and abs(sy - 1) < 0.001:
        return template
    return cv2.resize(
        template,
        (max(1, round(template.shape[1] * sx)), max(1, round(template.shape[0] * sy))),
        interpolation=cv2.INTER_CUBIC,
    )


def detect_images(
    images: Sequence[np.ndarray],
    width: int,
    height: int,
    base: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Pick the best-fitting profile by counter-marker match score."""
    from .counter import counter_offset

    original = base or load_base()
    marker = read_image(BASE / original["counter_marker"], cv2.IMREAD_GRAYSCALE)
    candidates = layout_candidates(width, height, base=original)
    content = pillarbox_content(images, width, height)
    if content:
        candidates += layout_candidates(width, height, base=original, content_box=content)
    evidence = []
    for profile in candidates:
        scores = []
        for image in images:
            try:
                _, score = counter_offset(image, profile, marker)
            except cv2.error:
                score = 0.0
            scores.append(float(score))
        evidence.append((max(scores, default=0.0), profile, scores))
    evidence.sort(key=lambda entry: entry[0], reverse=True)
    score, profile, scores = evidence[0]
    if score < 0.78 or (len(evidence) > 1 and score - evidence[1][0] < 0.025):
        # Do not silently select stretched geometry when there was no in-game HUD.
        profile = candidates[0]
        score = next(entry[0] for entry in evidence if entry[1] is profile)
        profile["layout_status"] = "unconfirmed"
    else:
        profile["layout_status"] = "confirmed"
    profile["layout_score"] = round(score, 4)
    return profile


def confirm_by_name_row(
    images: Sequence[np.ndarray], profile: dict[str, Any], minimum: int = 2
) -> dict[str, Any] | None:
    """Voice-icon check on the account name row when the counter strip is occluded.

    Accepts only standard UI, unstretched candidates; icon must land on the name
    row's height in at least `minimum` frames.
    """
    if profile.get("ui_layout") == "compact" or any(
        abs(v - profile["reference_size"][1] / 1080) > 0.001 for v in profile.get("hud_scale", [1, 1])
    ):
        return None
    voice = read_image(BASE / "templates/voice.png", cv2.IMREAD_GRAYSCALE)
    if voice is None:
        return None
    width, height = profile["reference_size"]
    x1, y1, x2, y2 = profile["account_roi"]
    hits = []
    for image in images:
        h, w = image.shape[:2]
        if (w, h) != (width, height):
            continue
        top = max(0, y1 - 8)
        band = cv2.cvtColor(image[top : min(h, y2 + 8), x1 : min(w, x2 + 260)], cv2.COLOR_BGR2GRAY)
        best = 0.0
        for factor in (0.75, 1.0, 1.25):
            template = cv2.resize(voice, None, fx=factor * h / 1080, fy=factor * h / 1080)
            if band.shape[0] < template.shape[0] or band.shape[1] < template.shape[1]:
                continue
            _, score, _, (_, y) = cv2.minMaxLoc(cv2.matchTemplate(band, template, cv2.TM_CCOEFF_NORMED))
            if abs(top + y + template.shape[0] / 2 - (y1 + y2) / 2) <= 6 * h / 1080:
                best = max(best, float(score))
        hits.append(round(best, 3))
    return {"name_row_scores": hits} if sum(s >= 0.85 for s in hits) >= minimum else None
