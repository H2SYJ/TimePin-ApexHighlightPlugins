"""Player name recognition via caption-view OCR and cross-frame consensus.

Ported from apex-highlight-clipper name_reader.py / player_names.py /
weapon_reader.caption_views. The latin-glyph and Japanese review paths are
dropped — their models are not bundled with the plugin; the multilingual rec
model is the main path.
"""

from __future__ import annotations

from collections import Counter
import re
from statistics import median
from typing import Any, Callable
import unicodedata

import cv2
import numpy as np

from .extra import PlayerNameInfo, TeammateInfo


# ---------------------------------------------------------------------------
# Name normalization (AHC player_names.py)
# ---------------------------------------------------------------------------

def strip_clan_tag(value: str) -> str:
    name = unicodedata.normalize("NFKC", value or "").strip()
    name = re.sub(r"^\[?[^\[\]\s]{1,6}\]\s*", "", name)
    name = re.sub(r"^\[[A-Za-z0-9]{2,6}(?=[぀-ヿ㐀-鿿])", "", name)
    return re.sub(r"^\[[A-Za-z0-9]{2,6}\s+", "", name).strip()


def name_key(value: str) -> str:
    return "".join(c for c in strip_clan_tag(value).casefold() if c.isalnum())


def close_name(left: str, right: str, minimum_length: int = 6) -> bool:
    left, right = name_key(left), name_key(right)
    if left == right:
        return bool(left)
    length = min(len(left), len(right))
    limit = 2 if length >= 12 or (length >= 10 and left.isascii() and right.isascii()) else 1
    if min(len(left), len(right)) < minimum_length or abs(len(left) - len(right)) > limit:
        return False
    previous = list(range(len(right) + 1))
    for i, a in enumerate(left, 1):
        row = [i]
        for j, b in enumerate(right, 1):
            row.append(min(row[-1] + 1, previous[j] + 1, previous[j - 1] + (a != b)))
        previous = row
    return previous[-1] <= limit


def player_name(text: Any, *, allow_rendered_square: bool = False) -> str:
    """Keep only plausibly-typed player names from OCR output."""
    if not isinstance(text, str):
        return ""
    name = strip_clan_tag(re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text))).strip(" ()（）♫♪◀▶□")
    key = name_key(name)
    # A fixed name box can be a numeric account; digits with separators (811<3)
    # are also real. Pure digits still need enough length to not be a counter.
    if not 2 <= len(name) <= 40 or not (
        any(c.isalpha() for c in name)
        or re.fullmatch(r"\d{5,20}", name)
        or (re.fullmatch(r"\d+(?:[<>]+\d+)+", name) and len(name) >= 4)
    ):
        return ""
    if re.search(r"[\x00-\x1f�:=+!?！？。]", name) or ("□" in name and not allow_rendered_square):
        return ""
    # Mid-name < > are real characters; at the edges they are icon misreads.
    if re.search(r"(?<!\w)[<>]|[<>](?!\w)", name):
        return ""
    if key.upper() in {
        "MAXLEVEL", "DEATHBOXRECOVERY", "RECOVERBANNER", "SPECTATING",
        "DAMAGEINFLICTED", "KNOCKEDDOWN", "ELIMINATED", "ASSIST",
    }:
        return ""
    if re.search(r"傷害|伤害|造成|擊倒|击倒|消滅|消灭|助攻|進化|进化|等級|等级|下個|下个|回收|旗幟|旗帜|觀戰|观战", name):
        return ""
    return name


def consensus_name(values: list[dict[str, Any] | None], minimum: int = 2) -> dict[str, Any] | None:
    """Cross-frame name vote; two preprocessed views of one frame are not votes."""
    values = [
        {**v, "name": strip_clan_tag(v["name"])}
        for v in values
        if v and name_key(v.get("name")) and v.get("confidence", 0) >= 0.80
    ]
    counts = Counter(name_key(v["name"]) for v in values)
    reliable = [key for key, n in counts.items() if n >= minimum]
    if not reliable:
        return None
    key = max(
        reliable,
        key=lambda k: (counts[k], sum(v["confidence"] for v in values if name_key(v["name"]) == k)),
    )
    if any(counts[k] == counts[key] and not close_name(k, key) for k in reliable if k != key):
        return None
    selected = [v for v in values if name_key(v["name"]) == key]
    spelling = Counter(v["name"] for v in selected)
    best = max(selected, key=lambda v: (spelling[v["name"]], v["confidence"]))
    return {
        **best,
        "confidence": min(v["confidence"] for v in selected),
        "consensus_confidence": median(v["confidence"] for v in selected),
        "votes": len(selected),
    }


def canonical_name(
    value: dict[str, Any] | None, reference: dict[str, Any] | None
) -> dict[str, Any] | None:
    """Adopt the recording-wide spelling only when evidence supports it."""
    value = {**value, "name": strip_clan_tag(value["name"])} if value else None
    reference = {**reference, "name": strip_clan_tag(reference["name"])} if reference else None
    if not value or not reference or not close_name(value["name"], reference["name"], 6):
        return value
    # Majority vote cannot override two-view clear reads of a different spelling.
    if name_key(value["name"]) != name_key(reference["name"]) and value.get("view_agreement", 0) >= 2 and value.get("confidence", 0) >= 0.92:
        return value
    if name_key(value["name"]) != name_key(reference["name"]) and (
        value.get("confidence", 0) >= 0.92
        or reference.get("confidence", 0) < 0.86
        or reference.get("votes", 0) < 3
    ):
        return value
    return {**value, "name": reference["name"], "method": "recording_name_consensus"}


# ---------------------------------------------------------------------------
# Caption views (AHC weapon_reader.caption_views + caption_image)
# ---------------------------------------------------------------------------

def caption_image(patch: np.ndarray) -> np.ndarray:
    """Foreground-separated binary view; keeps anti-aliased strokes elsewhere."""
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    values = hsv[:, :, 2][hsv[:, :, 1] < 100]
    if values.size < 8:
        return cv2.resize(patch, None, fx=2, fy=2)
    threshold = max(80, float(np.percentile(values, 95)) * 0.67)
    mask = (hsv[:, :, 1] < 100) & (hsv[:, :, 2] > threshold)
    ys, xs = np.nonzero(mask)
    if len(xs) < 8:
        return cv2.resize(patch, None, fx=2, fy=2)
    image = np.where(mask, 0, 255).astype(np.uint8)
    image = image[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    image = cv2.copyMakeBorder(image, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=255)
    return cv2.cvtColor(cv2.resize(image, None, fx=3, fy=3), cv2.COLOR_GRAY2BGR)


def caption_views(patch: np.ndarray) -> list[np.ndarray]:
    masked = caption_image(patch)
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    values = hsv[:, :, 2][hsv[:, :, 1] < 100]
    if values.size < 8:
        return [masked, cv2.resize(patch, None, fx=2, fy=2)]
    mask = (hsv[:, :, 1] < 100) & (hsv[:, :, 2] > max(80, float(np.percentile(values, 95)) * 0.67))
    ys, xs = np.nonzero(mask)
    if len(xs) < 8:
        return [masked, cv2.resize(patch, None, fx=2, fy=2)]
    # Keep anti-aliasing — binarization damages fine strokes in complex glyphs.
    raw = patch[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    color = tuple(int(c) for c in np.median(patch.reshape(-1, 3), axis=0))
    raw = cv2.copyMakeBorder(raw, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=color)
    return [masked, cv2.resize(raw, None, fx=3, fy=3)]


def rendered_square(patch: np.ndarray) -> bool:
    """Verify an actual drawn hollow square in the name row; an OCR placeholder
    char is not glyph evidence."""
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    height = patch.shape[0]
    for level in (100, 120, 140):
        mask = ((hsv[:, :, 1] < 90) & (hsv[:, :, 2] > level)).astype(np.uint8) * 255
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is None:
            continue
        for index, contour in enumerate(contours):
            child = hierarchy[0, index, 2]
            if child < 0:
                continue
            _, _, width, tall = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            if (
                5 <= tall <= height * 0.8
                and 0.8 <= width / tall <= 1.2
                and area >= 0.85 * (width - 1) * (tall - 1)
                and cv2.contourArea(contours[child]) >= area * 0.25
            ):
                return True
    return False


def name_value(
    views: list[tuple[str, float]], *, allow_rendered_square: bool = False
) -> dict[str, Any] | None:
    """Pick the best name among OCR views of one patch."""
    choices = []
    for text, confidence in views:
        name = player_name(text, allow_rendered_square=allow_rendered_square)
        if confidence >= 0.80 and name:
            choices.append({"name": name, "confidence": float(confidence)})
    for value in choices:
        value["view_agreement"] = sum(name_key(other["name"]) == name_key(value["name"]) for other in choices)
    return max(choices, key=lambda row: (row["view_agreement"], row["confidence"])) if choices else None


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------

class NameReader:
    """Reads player-name patches with a text recognizer.

    `recognizer` is a callable: list[np.ndarray] -> list[tuple[str, float]].
    Wire it to ``timepin_vision.OcrEngine``'s rec stage or any compatible
    batch recognizer.
    """

    def __init__(self, recognizer: Callable[[list[np.ndarray]], list[tuple[str, float]]]) -> None:
        self.recognizer = recognizer

    def __call__(self, patches: list[np.ndarray]) -> list[dict[str, Any] | None]:
        views: list[np.ndarray] = []
        for patch in patches:
            masked, color = caption_views(patch)
            gray = cv2.cvtColor(cv2.cvtColor(color, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
            views.extend([masked, color, gray])
        if not views:
            return []
        texts = self.recognizer(views)
        results = [name_value(texts[i : i + 3]) for i in range(0, len(texts), 3)]
        for index, value in enumerate(results):
            if value is not None:
                continue
            candidate = name_value(texts[index * 3 : index * 3 + 3], allow_rendered_square=True)
            if (
                candidate
                and "□" in candidate["name"]
                and candidate["confidence"] >= 0.85
                and candidate["view_agreement"] >= 2
                and rendered_square(patches[index])
            ):
                results[index] = {**candidate, "raw_name": candidate["name"], "rendered_placeholder": True}
        return results


def names_to_infos(
    player: dict[str, Any] | None, teammates: list[dict[str, Any] | None]
) -> tuple[PlayerNameInfo | None, list[TeammateInfo]]:
    """Convert consensus dicts into plugin info dataclasses."""
    player_info = None
    if player:
        player_info = PlayerNameInfo(
            name=player["name"],
            confidence=round(float(player.get("confidence", 0.0)), 3),
            method=player.get("method", "caption_view_ocr"),
        )
    teammate_infos = [
        TeammateInfo(
            name=t["name"],
            confidence=round(float(t.get("confidence", 0.0)), 3),
            method=t.get("method", "caption_view_ocr"),
            slot=i,
        )
        for i, t in enumerate(teammates)
        if t
    ]
    return player_info, teammate_infos
