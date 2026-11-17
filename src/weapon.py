"""Native HUD weapon names, calibrated icon fallback and output-local learning.

Port of AHC weapon_reader.py / event_weapon_reader.py and apex_clipper's weapon
annotation. No decoder, downloads, subprocesses, global caches or video writes.
The bundled catalog is an unmodified upstream snapshot (29 base + 3 supplemental
weapons); only 3 of those weapons have real-video icon prototypes.

Public API
----------
WeaponReader(profile, *, backend='cpu', output_dir=None, cancel_event=None,
             text_reader=None, catalog=None, learn=True)
    text_reader(crops) -> ([(text, confidence), ...], elapsed), injectable without
    any OCR runtime. The default lazily uses src.ocr and the already bundled
    PP-OCRv6 rec model, NOT upstream's unavailable/downloaded PP-OCRv5 model.
read(frame_or_image, timestamp=None, *, row=None) -> dict
    BGR uint8 image + seconds, or frames.Frame (uses its PTS/path). Name crops
    retain native pixels until caption preprocessing. row supplies view exclusions
    and ammo evidence; it is NOT mutated. Unknown/ambiguous stays weapon=None.
sample(frames, times=None, *, rows=None) -> list[dict]
    Ordered, cancellable reads; Frame inputs need no separate times. read_batch
    is an alias. Results include both AHC row names and event observation names.
save_learning() -> Path | None
    Explicit atomic save ONLY to output_dir/learned-weapons.json. Nothing is
    written during read/sample. At least two distinct high-confidence, two-view
    OCR observations with consistent icons are required; pending samples are NOT
    activated in this reader. A new reader merges validated saved samples.
annotate_events(events, observations, *, cancel_event=None) -> list[dict]
    Pure copy: shoot events get weapon fields; confirmed self combat outcomes get
    weapon_context, which always has kill_attribution=False. Temporal proximity
    is never presented as the weapon that caused a kill.

Cancellation raises src.errors.Cancelled between frames, OCR groups, matching
loops and before writes. A native inference already in flight is not preempted.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import re
import stat
import tempfile
import time
import unicodedata
from collections.abc import Mapping
from copy import deepcopy
from difflib import SequenceMatcher
from itertools import zip_longest
from numbers import Real
from pathlib import Path

import cv2
import numpy as np

from .errors import Cancelled
from .frames import Frame
from .hud import crop, icon_signature, inactive_view
from .resources import RESOURCE_DIR, require_file

CATALOG_SHA256 = "f7b91888615f7200d1b475f23325390ea2389f35c0a53d540fc59c94abf80994"
VERSION = "timepin-weapon-reader-v1"
LEARNING_SCHEMA = "timepin-weapon-learning-v1"
LEARNING_FILE = "learned-weapons.json"
ICON_BYTES = 96 * 28 // 8
MAX_SAMPLES = 32
MAX_LEARNED_BYTES = 2 * 1024 * 1024
NAME_CONFIDENCE = .75
ICON_MINIMUM = .86
ICON_MARGIN = .10
TRANSLATION = str.maketrans(
    "敵轉換衝鋒槍電專輕機噴獵獸擊長萊連發賓雙幫復仇標記莫彈徑",
    "敌转换冲锋枪电专轻机喷猎兽击长莱连发宾双帮复仇标记莫弹径",
)


def _check_cancel(event):
    if event is not None and event.is_set():
        raise Cancelled("Weapon recognition cancelled")


def _number(value, *, minimum=0., maximum=math.inf):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("Expected a finite number")
    try:
        number = float(value)
    except OverflowError as exc:
        raise ValueError("Expected a finite number") from exc
    if not math.isfinite(number) or not minimum <= number <= maximum:
        raise ValueError("Number outside permitted range")
    return number


def _signature(value):
    if not isinstance(value, str) or len(value) != ICON_BYTES * 4 // 3:
        raise ValueError("Invalid 96x28 packed icon signature")
    try:
        raw = base64.b64decode(value, validate=True)
    except (ValueError, base64.binascii.Error) as exc:
        raise ValueError("Invalid icon base64") from exc
    if len(raw) != ICON_BYTES:
        raise ValueError("Invalid icon length")
    bits = np.unpackbits(np.frombuffer(raw, dtype=np.uint8)).astype(bool)
    # Empty/full masks contain no identifying geometry. Never learn/match them.
    if not 4 <= np.count_nonzero(bits) <= bits.size * .8:
        raise ValueError("Icon signature has no usable geometry")
    return bits


def icon_similarity(a, b):
    """AHC Dice similarity, with malformed/blank signatures rejected safely."""
    try:
        x, y = _signature(a), _signature(b)
    except ValueError:
        return 0.
    return float(2 * np.count_nonzero(x & y) / (np.count_nonzero(x) + np.count_nonzero(y)))


def _validate_catalog(value):
    if not isinstance(value, dict) or not isinstance(value.get("weapons"), list):
        raise ValueError("Weapon catalog must contain a weapons array")
    supplements = value.get("supplemental_weapons", [])
    if not isinstance(supplements, list):
        raise ValueError("supplemental_weapons must be an array")
    seen = set()
    for entry in value["weapons"] + supplements:
        if not isinstance(entry, dict):
            raise ValueError("Invalid weapon entry")
        ident = entry.get("id")
        if not isinstance(ident, str) or re.fullmatch(r"[a-z0-9_]{1,48}", ident) is None or ident in seen:
            raise ValueError("Invalid or duplicate weapon id")
        seen.add(ident)
        if any(not isinstance(entry.get(k), str) or not entry[k] for k in ("name", "category")):
            raise ValueError("Weapon name/category must be nonempty strings")
        aliases = entry.get("aliases")
        if not isinstance(aliases, list) or not aliases or any(not isinstance(a, str) or not a or len(a) > 128 for a in aliases):
            raise ValueError("Invalid weapon aliases")
        variants = entry.get("variants", [])
        if not isinstance(variants, list) or any(not isinstance(v, str) or not v or len(v) > 32 for v in variants):
            raise ValueError("Invalid weapon variants")
        prototypes = entry.get("prototypes", [])
        if not isinstance(prototypes, list):
            raise ValueError("Invalid icon prototypes")
        for prototype in prototypes:
            _signature(prototype)
    return deepcopy(value)


def load_catalog():
    """Read-only, hash-verified upstream snapshot; no runtime catalog mutation."""
    path = require_file("weapon-catalog.json")
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != CATALOG_SHA256:
        raise RuntimeError("Bundled weapon catalog failed SHA-256 verification")
    return _validate_catalog(json.loads(data.decode("utf-8")))


def normalized(text):
    text = unicodedata.normalize("NFKC", text).translate(TRANSLATION).upper()
    return "".join(c for c in text if c.isalnum())


def resolve_weapon_text(text, confidence=1., catalog=None):
    """Upstream multilingual aliases, ambiguity margin and model-digit repair."""
    confidence = _number(confidence, maximum=1.)
    if not isinstance(text, str) or confidence < NAME_CONFIDENCE:
        return None
    catalog = load_catalog() if catalog is None else catalog
    value = normalized(text)
    if len(value) < 2:
        return None
    akimbo = "AKIMBO" in value or "双持" in value
    clean = value.replace("AKIMBO", "").replace("双持", "")
    if re.fullmatch(r"[RP][0-9OI]+", clean):
        clean = clean[0] + clean[1:].replace("O", "0").replace("I", "1")
    scores = []
    for weapon in catalog["weapons"] + catalog.get("supplemental_weapons", []):
        score = 0.
        for alias in {normalized(a) for a in weapon["aliases"]}:
            if clean == alias:
                score = 1.
                break
            if len(alias) >= 3 and alias in clean and len(alias) / max(len(clean), 1) >= .70:
                score = max(score, .96)
            if min(len(alias), len(clean)) >= 4:
                similarity = SequenceMatcher(None, clean, alias).ratio()
                if similarity >= .84:
                    score = max(score, similarity)
        if score:
            scores.append((score, weapon))
    scores.sort(key=lambda pair: pair[0], reverse=True)
    if not scores or len(scores) > 1 and scores[0][0] - scores[1][0] < .08:
        return None
    score, weapon = scores[0]
    if score < .95 and confidence < .86:
        return None
    if akimbo and weapon["id"] not in ("p2020", "mozambique"):
        return None
    variant = "akimbo" if akimbo else None
    if weapon["id"] == "re45" and "BURST" in clean:
        variant = "burst"
    if weapon["id"] == "hemlok" and "BREACH" in clean:
        variant = "breach"
    return {"id": weapon["id"], "name": ("双持" if akimbo else "") + weapon["name"],
            "category": weapon["category"], "variant": variant, "text_match_score": score}


def active_slot(highlights, margin=25):
    """Select by highlight BEFORE OCR; never pick the other slot's clearer name."""
    if len(highlights) != 2 or abs(highlights[0] - highlights[1]) < margin:
        return None
    return 0 if highlights[0] > highlights[1] else 1


def caption_image(patch):
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    values = hsv[:, :, 2][hsv[:, :, 1] < 100]
    if values.size < 8:
        return cv2.resize(patch, None, fx=2, fy=2)
    mask = (hsv[:, :, 1] < 100) & (hsv[:, :, 2] > max(80, float(np.percentile(values, 95)) * .67))
    ys, xs = np.nonzero(mask)
    if len(xs) < 8:
        return cv2.resize(patch, None, fx=2, fy=2)
    image = np.where(mask, 0, 255).astype(np.uint8)
    image = image[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    image = cv2.copyMakeBorder(image, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=255)
    return cv2.cvtColor(cv2.resize(image, None, fx=3, fy=3), cv2.COLOR_GRAY2BGR)


def caption_views(patch):
    """Binary and antialiased native-name views (AHC preprocessing)."""
    masked = caption_image(patch)
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    values = hsv[:, :, 2][hsv[:, :, 1] < 100]
    if values.size < 8:
        return [masked, cv2.resize(patch, None, fx=2, fy=2)]
    mask = (hsv[:, :, 1] < 100) & (hsv[:, :, 2] > max(80, float(np.percentile(values, 95)) * .67))
    ys, xs = np.nonzero(mask)
    if len(xs) < 8:
        return [masked, cv2.resize(patch, None, fx=2, fy=2)]
    raw = patch[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    color = tuple(int(c) for c in np.median(patch.reshape(-1, 3), axis=0))
    raw = cv2.copyMakeBorder(raw, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=color)
    return [masked, cv2.resize(raw, None, fx=3, fy=3)]


def highlight_score(patch):
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    values = hsv[:, :, 2][hsv[:, :, 1] < 70]
    if not values.size:
        return 0.
    foreground = values[values > max(80, float(np.percentile(values, 25)) + 25)]
    return float(np.percentile(foreground, 90)) if foreground.size >= 4 else 0.


def _identity(weapon):
    return (weapon["id"], weapon.get("variant")) if weapon else None


def recognize_text_views(views, catalog=None):
    catalog = load_catalog() if catalog is None else catalog
    if not views:
        return {"text": "", "confidence": 0.}, None
    matches = []
    for text, confidence in views:
        match = resolve_weapon_text(text, confidence, catalog)
        if match:
            matches.append((confidence * match["text_match_score"], text, confidence, match))
    if matches and len({_identity(m[3]) for m in matches}) == 1:
        _, text, confidence, match = max(matches, key=lambda m: m[0])
        return {"text": text, "confidence": float(confidence)}, match
    text, confidence = max(views, key=lambda item: item[1])
    return {"text": text, "confidence": float(confidence)}, None


def _view_supports(text, confidence, weapon, catalog):
    match = resolve_weapon_text(text, confidence, catalog)
    if _identity(match) == _identity(weapon):
        return True
    if not weapon or confidence < .6:
        return False
    clean = normalized(text)
    if not clean:
        return False
    entry = next(w for w in catalog["weapons"] + catalog.get("supplemental_weapons", []) if w["id"] == weapon["id"])
    aliases = {normalized(a) for a in entry["aliases"]}
    stripped = clean[1:] if len(clean) >= 3 and clean[0].isascii() else clean
    if clean in aliases or stripped in aliases:
        return True
    return len(stripped) == 1 and not stripped.isascii() and any(
        2 <= len(a) <= 3 and not a.isascii() and a.startswith(stripped) for a in aliases)


def ctc_label_score(probabilities, tokens):
    """Original CTC forward probability; no fabricated characters/probabilities."""
    if not tokens or probabilities.ndim != 2 or not len(probabilities):
        return 0.
    sequence = [0]
    for token in tokens:
        sequence.extend([token, 0])
    log = np.log(np.maximum(probabilities[:, sequence], 1e-12))
    state = np.full(len(sequence), -np.inf)
    state[0], state[1] = log[0, 0], log[0, 1]
    skip = np.array([False, False] + [sequence[i] != 0 and sequence[i] != sequence[i - 2] for i in range(2, len(sequence))])
    for row in log[1:]:
        one = np.r_[-np.inf, state[:-1]]
        two = np.r_[-np.inf, -np.inf, state[:-2]]
        state = np.logaddexp(np.logaddexp(state, one), np.where(skip, two, -np.inf)) + row
    return float(np.exp(np.logaddexp(state[-1], state[-2])))


class WeaponTextReader:
    """AHC small-width batches + evidence-backed closed-vocabulary CTC repair."""

    def __init__(self, recognizer, catalog=None, cancel_event=None):
        self.recognizer = recognizer
        self.catalog = load_catalog() if catalog is None else catalog
        self.cancel_event = cancel_event
        self.labels = []
        dictionary = recognizer.postprocess_op.dict
        for weapon in self.catalog["weapons"] + self.catalog.get("supplemental_weapons", []):
            for alias in sorted(set(weapon["aliases"])):
                for text in sorted({alias, alias.upper()}):
                    if text and all(c in dictionary for c in text):
                        self.labels.append((weapon["id"], text, [dictionary[c] for c in text]))

    def constrained(self, probabilities, greedy):
        text, confidence = greedy
        if resolve_weapon_text(text, confidence, self.catalog) and confidence >= .85:
            return greedy
        clean = normalized(text)
        if not clean:
            return greedy
        peaks = probabilities.max(axis=0)
        scores = {}
        for ident, alias, tokens in self.labels:
            _check_cancel(self.cancel_event)
            label = normalized(alias)
            one_off = (len(clean) == len(label) == 2 and not clean.isascii() and not label.isascii()
                       and sum(a == b for a, b in zip(clean, label)) == 1)
            similar = min(len(clean), len(label)) >= 2 and SequenceMatcher(None, clean, label).ratio() >= .70
            if not (clean == label or similar or one_off or len(clean) == 1 and not clean.isascii() and len(label) == 2 and clean in label):
                continue
            if any(peaks[token] < .08 for token in tokens):
                continue
            probability = ctc_label_score(probabilities, tokens)
            if probability > scores.get(ident, (0., ""))[0]:
                scores[ident] = (probability, alias)
        ranked = sorted(scores.values(), reverse=True)
        if ranked and ranked[0][0] >= .12 and (len(ranked) == 1 or ranked[0][0] >= ranked[1][0] * 8):
            return ranked[0][1], .75
        return greedy

    def __call__(self, images):
        started = time.perf_counter()
        result = []
        reader = self.recognizer
        for offset in range(0, len(images), 4):
            _check_cancel(self.cancel_event)
            group = images[offset:offset + 4]
            ratio = max(reader.rec_image_shape[2] / reader.rec_image_shape[1], max(im.shape[1] / im.shape[0] for im in group))
            batch = np.stack([reader.resize_norm_img(im, ratio) for im in group]).astype(np.float32)
            probabilities = reader.session(batch)[0]
            greedy = reader.postprocess_op(probabilities)
            result.extend(self.constrained(p, g) for p, g in zip(probabilities, greedy))
        _check_cancel(self.cancel_event)
        return result, time.perf_counter() - started


def _native_patch(image, box, reference):
    h, w = image.shape[:2]
    rw, rh = reference
    x1, y1, x2, y2 = [round(v * scale) for v, scale in zip(box, (w / rw, h / rh, w / rw, h / rh))]
    if not 0 <= x1 < x2 <= w or not 0 <= y1 < y2 <= h:
        raise ValueError("Weapon ROI is empty or outside the frame")
    return image[y1:y2, x1:x2]


def icon_mask(image, profile):
    patch = cv2.resize(crop(image, profile["weapon_icon"], profile["reference_size"]), (96, 28))
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    mask = ((hsv[:, :, 1] < 80) & (hsv[:, :, 2] > 170)) | ((hsv[:, :, 1] >= 80) & (hsv[:, :, 2] > 200))
    return base64.b64encode(np.packbits(mask).tobytes()).decode("ascii")


def _similar_observations(a, b):
    edge = icon_similarity(a.get("weapon_icon"), b.get("weapon_icon"))
    if icon_similarity(a.get("weapon_icon_mask"), b.get("weapon_icon_mask")) >= .88:
        return max(edge, .80)
    return edge


def _regular(path):
    """Reject symlinks, junctions/reparse points and special files, even dangling."""
    value = path.lstat()
    if stat.S_ISLNK(value.st_mode) or getattr(value, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0):
        raise ValueError("Learning paths must not contain symlinks/reparse points")
    return value


def _output_root(output_dir):
    path = Path(output_dir)
    if not path.is_absolute():
        raise ValueError("output_dir must be an existing absolute directory")
    # Check lexical ancestors as well as the resolved root: a pre-existing link
    # must not turn the authorization into a write outside the supplied directory.
    for part in [path, *path.parents]:
        _regular(part)
    root = path.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("output_dir must be an existing directory")
    if root == RESOURCE_DIR.resolve() or RESOURCE_DIR.resolve() in root.parents:
        raise ValueError("Learning must not modify bundled resources")
    return root


class WeaponReader:
    def __init__(self, profile, *, backend="cpu", output_dir=None, cancel_event=None,
                 text_reader=None, catalog=None, learn=True):
        if backend not in ("cpu", "dml", "cuda"):
            raise ValueError("backend must be cpu/dml/cuda")
        self.profile = deepcopy(profile)
        reference = self.profile.get("reference_size")
        if not isinstance(reference, (tuple, list)) or len(reference) != 2:
            raise ValueError("Missing weapon reference_size")
        rw, rh = [_number(v, minimum=1.) for v in reference]
        for key in ("weapon_name_boxes", "weapon_name_metrics"):
            boxes = self.profile.get(key)
            if not isinstance(boxes, (tuple, list)) or len(boxes) != 2:
                raise ValueError(f"Missing two {key}")
            for box in boxes:
                self._validate_box(box, rw, rh)
        self._validate_box(self.profile.get("weapon_icon"), rw, rh)
        self.backend = backend
        self.cancel_event = cancel_event
        self.catalog = load_catalog() if catalog is None else _validate_catalog(catalog)
        self._entries = {w["id"]: w for w in self.catalog["weapons"] + self.catalog.get("supplemental_weapons", [])}
        self._text_reader = text_reader
        self.learn = bool(learn)
        self._root = _output_root(output_dir) if output_dir is not None else None
        root_stat = self._root.stat() if self._root is not None else None
        self._root_identity = (root_stat.st_dev, root_stat.st_ino) if root_stat else None
        self._loaded = self._load_learning() if self._root is not None else {}
        self._pending = {}

    @staticmethod
    def _validate_box(box, width, height):
        if not isinstance(box, (tuple, list)) or len(box) != 4:
            raise ValueError("Weapon box must have four coordinates")
        x1, y1, x2, y2 = [_number(v) for v in box]
        if not 0 <= x1 < x2 <= width or not 0 <= y1 < y2 <= height:
            raise ValueError("Weapon box outside reference_size")

    @property
    def text_reader(self):
        if self._text_reader is None:
            from .ocr import create_ocr_engine

            _check_cancel(self.cancel_event)
            engine = create_ocr_engine(backend=self.backend,
                                       model_path=require_file("models/ppocrv6_rec_small.onnx"),
                                       rec_batch_num=4)
            self._text_reader = WeaponTextReader(engine.text_rec, self.catalog, self.cancel_event)
        return self._text_reader

    def _safe_target(self):
        if self._root is None:
            raise ValueError("Learning requires output_dir")
        current = _output_root(self._root)
        value = current.stat()
        if current != self._root or (value.st_dev, value.st_ino) != self._root_identity:
            raise ValueError("Authorized output directory changed")
        target = current / LEARNING_FILE
        if os.path.lexists(target):
            value = _regular(target)
            if not stat.S_ISREG(value.st_mode) or value.st_nlink != 1:
                raise ValueError("Learning target must be an unlinked regular file")
        return target

    def _validate_learning(self, data):
        if not isinstance(data, dict) or data.get("version") != 1 or data.get("schema") != LEARNING_SCHEMA:
            raise ValueError("Unsupported learned weapon schema")
        entries = data.get("weapons")
        if not isinstance(entries, list) or len(entries) > 128:
            raise ValueError("Invalid learned weapons array")
        result = {}
        for entry in entries:
            _check_cancel(self.cancel_event)
            if not isinstance(entry, dict) or entry.get("id") not in self._entries:
                raise ValueError("Unknown learned weapon id")
            ident, variant = entry["id"], entry.get("variant")
            if variant is not None and (not isinstance(variant, str) or variant not in self._entries[ident].get("variants", [])):
                raise ValueError("Invalid learned weapon variant")
            key = (ident, variant)
            samples = entry.get("samples")
            if key in result or not isinstance(samples, list) or not 2 <= len(samples) <= MAX_SAMPLES:
                raise ValueError("Learning requires 2..32 samples per unique weapon/variant")
            validated = []
            seen_pts, seen_hashes = set(), set()
            for sample in samples:
                if not isinstance(sample, dict):
                    raise ValueError("Invalid learned sample")
                _signature(sample.get("icon"))
                mask = sample.get("mask")
                if mask is not None:
                    _signature(mask)
                confidence = _number(sample.get("confidence"), minimum=.90, maximum=1.)
                timestamp = _number(sample.get("timestamp"))
                digest = sample.get("frame_sha256")
                if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
                    raise ValueError("Invalid learned frame digest")
                if timestamp in seen_pts or digest in seen_hashes:
                    raise ValueError("Learning samples must be independent frames")
                seen_pts.add(timestamp)
                seen_hashes.add(digest)
                validated.append({"icon": sample["icon"], "mask": mask, "confidence": confidence,
                                  "timestamp": timestamp, "frame_sha256": digest})
            if any(not any(i != j and icon_similarity(a["icon"], b["icon"]) >= ICON_MINIMUM
                           for j, b in enumerate(validated)) for i, a in enumerate(validated)):
                raise ValueError("Learned icons lack two-frame consensus")
            result[key] = validated
        return result

    def _load_learning(self):
        target = self._safe_target()
        if not target.exists():
            return {}
        if target.stat().st_size > MAX_LEARNED_BYTES:
            raise ValueError("Learned weapons file exceeds 2 MiB")
        # Never follow a final-component link on platforms supporting O_NOFOLLOW.
        fd = os.open(target, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        with os.fdopen(fd, "rb") as stream:
            payload = stream.read(MAX_LEARNED_BYTES + 1)
        if len(payload) > MAX_LEARNED_BYTES:
            raise ValueError("Learned weapons file exceeds 2 MiB")
        try:
            data = json.loads(payload.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError("Malformed learned weapons JSON") from exc
        return self._validate_learning(data)

    def _icon_match(self, signature):
        ranked = []
        for entry in self._entries.values():
            _check_cancel(self.cancel_event)
            scores = [icon_similarity(signature, p) for p in entry.get("prototypes", [])]
            if scores:
                ranked.append((max(scores), (entry["id"], None), "calibrated_icon"))
        for key, samples in self._loaded.items():
            _check_cancel(self.cancel_event)
            scores = sorted((icon_similarity(signature, s["icon"]) for s in samples), reverse=True)
            ranked.append((scores[1], key, "learned_icon_consensus"))
        # Multiple templates for the SAME base/variant are one candidate, not a
        # fabricated ambiguity. Different variants remain distinct candidates.
        candidates = {}
        for score, key, method in ranked:
            if key not in candidates or score > candidates[key][0]:
                candidates[key] = (score, key, method)
        ranked = sorted(candidates.values(), key=lambda value: value[0], reverse=True)
        if not ranked or ranked[0][0] < ICON_MINIMUM or len(ranked) > 1 and ranked[0][0] - ranked[1][0] < ICON_MARGIN:
            return None, 0., "unknown"
        score, (ident, variant), method = ranked[0]
        entry = self._entries[ident]
        return {"id": ident, "name": ("双持" if variant == "akimbo" else "") + entry["name"],
                "category": entry["category"], "variant": variant}, score, method

    def read(self, frame_or_image, timestamp=None, *, row=None):
        _check_cancel(self.cancel_event)
        if row is not None and not isinstance(row, Mapping):
            raise TypeError("row must be a mapping")
        row = {} if row is None else row
        if isinstance(frame_or_image, Frame):
            if timestamp is None:
                timestamp = frame_or_image.pts_sec
            data = np.frombuffer(frame_or_image.path.read_bytes(), dtype=np.uint8)
            image = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Cannot decode weapon frame: {frame_or_image.path}")
        else:
            image = frame_or_image
        second = _number(timestamp)
        if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 2:
            raise ValueError("Weapon frames must be nonempty uint8 BGR arrays")
        result = {"version": VERSION, "t": second, "time": second, "actual_pts": second,
                  "weapon": None, "current_weapon": None, "weapon_id": None,
                  "weapon_name": "未知枪", "weapon_variant": None, "confidence": 0.,
                  "current_weapon_confidence": 0., "weapon_method": "unknown",
                  "slot": None, "active_weapon_slot": None, "weapon_labels": [],
                  "name_views": [], "name_view_agreement": 0, "excluded_view": False,
                  "inactive_view": False, "inactive": False,
                  "ammo": deepcopy(row.get("ammo")), "ammo_score": row.get("ammo_score", 0.)}
        if any(row.get(key) for key in ("inactive_view", "inactive", "friend_spectate", "spectator", "excluded_view", "downed", "round_ended")) or inactive_view(image):
            return {**result, "excluded_view": True, "inactive_view": True, "inactive": True,
                    "reason": "view_change_or_inactive"}
        if self.profile.get("layout_status") == "unconfirmed":
            return {**result, "reason": "unconfirmed_layout"}
        reference = self.profile["reference_size"]
        patches = [_native_patch(image, box, reference) for box in self.profile["weapon_name_boxes"]]
        highlights = [highlight_score(_native_patch(image, box, reference)) for box in self.profile["weapon_name_metrics"]]
        slot = active_slot(highlights)
        result["highlights"] = highlights
        if slot is None:
            backgrounds = [float(np.median(cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)[:, :, 2])) for patch in patches]
            result["slot_backgrounds"] = backgrounds
            if abs(backgrounds[0] - backgrounds[1]) >= 30 and max(highlights) > 0:
                slot = 0 if backgrounds[0] > backgrounds[1] else 1
                result["slot_basis"] = "name_panel_background"
        if slot is None:
            return {**result, "reason": "ambiguous_active_slot"}
        result.update(slot=slot, active_weapon_slot=slot)
        _native_patch(image, self.profile["weapon_icon"], reference)  # strict bounds, no negative slicing
        result["weapon_icon"] = icon_signature(image, self.profile)
        result["weapon_icon_mask"] = icon_mask(image, self.profile)
        views = [view for patch in patches for view in caption_views(patch)]
        values, _ = self.text_reader(views)
        _check_cancel(self.cancel_event)
        if len(values) != len(views):
            raise ValueError("Weapon OCR result count does not match crops")
        parsed = []
        for value in values:
            if not isinstance(value, (tuple, list)) or len(value) != 2 or not isinstance(value[0], str):
                raise ValueError("Invalid weapon OCR result")
            parsed.append((value[0], _number(value[1], maximum=1.)))
        labels, matches = [], []
        for index in range(2):
            label, match = recognize_text_views(parsed[index * 2:index * 2 + 2], self.catalog)
            labels.append({**label, "highlight": highlights[index]})
            matches.append(match)
        result["weapon_labels"] = labels
        name_views = parsed[slot * 2:slot * 2 + 2]
        result["name_views"] = [{"text": text, "confidence": confidence} for text, confidence in name_views]
        match = matches[slot]
        confidence, method = 0., "unknown"
        if match:
            confidence = labels[slot]["confidence"] * match["text_match_score"]
            method = "native_hud_text"
            result["name_view_agreement"] = sum(_view_supports(text, confidence, match, self.catalog) for text, confidence in name_views)
        elif not any(resolve_weapon_text(text, confidence, self.catalog) for text, confidence in name_views):
            # Conflicting readable names are NOT replaced by an icon guess.
            match, confidence, method = self._icon_match(result["weapon_icon"])
        result.update(weapon=match, current_weapon=deepcopy(match), confidence=confidence,
                      current_weapon_confidence=confidence, weapon_method=method,
                      weapon_id=match["id"] if match else None,
                      weapon_name=match["name"] if match else "未知枪",
                      weapon_variant=match.get("variant") if match else None)
        if not match:
            result["reason"] = "insufficient_or_conflicting_hud_evidence"
        if self.learn and self._root is not None and method == "native_hud_text" and confidence >= .90:
            # Partial-character support is useful for event context, but not for
            # persistent training: BOTH views must independently resolve the name.
            if all(_identity(resolve_weapon_text(text, score, self.catalog)) == _identity(match) and score >= .90 for text, score in name_views):
                self._collect(result, image)
        return result

    def _collect(self, result, image):
        try:
            _signature(result["weapon_icon"])
        except ValueError:
            return
        mask = result["weapon_icon_mask"]
        try:
            _signature(mask)
        except ValueError:
            mask = None
        key = _identity(result["weapon"])
        digest = hashlib.sha256(image.tobytes()).hexdigest()
        sample = {"icon": result["weapon_icon"], "mask": mask, "timestamp": result["actual_pts"],
                  "frame_sha256": digest, "confidence": result["confidence"]}
        samples = self._pending.setdefault(key, [])
        if any(s["frame_sha256"] == digest or s["timestamp"] == sample["timestamp"] for s in samples):
            return
        samples.append(sample)
        del samples[:-MAX_SAMPLES]

    def sample(self, frames, times=None, *, rows=None):
        """Read a bounded host batch. Missing/mismatched parallel inputs fail."""
        _check_cancel(self.cancel_event)
        missing = object()
        result = []
        if times is None and rows is None:
            return [self.read(frame) for frame in frames]
        iterators = [iter(frames)]
        if times is not None:
            iterators.append(iter(times))
        if rows is not None:
            iterators.append(iter(rows))
        for group in zip_longest(*iterators, fillvalue=missing):
            _check_cancel(self.cancel_event)
            if any(item is missing for item in group):
                raise ValueError("frames/times/rows lengths must match")
            index = 1
            second = group[index] if times is not None else None
            index += times is not None
            row = group[index] if rows is not None else None
            result.append(self.read(group[0], second, row=row))
        return result

    read_batch = sample

    def save_learning(self):
        _check_cancel(self.cancel_event)
        if self._root is None or not self.learn or not self._pending:
            return None
        merged = self._load_learning()
        added = False
        for key, samples in self._pending.items():
            _check_cancel(self.cancel_event)
            accepted = [a for a in samples if any(a["timestamp"] != b["timestamp"] and a["frame_sha256"] != b["frame_sha256"] and icon_similarity(a["icon"], b["icon"]) >= ICON_MINIMUM for b in samples)]
            if not accepted:
                continue
            combined = list(merged.get(key, []))
            for sample in accepted:
                if not any(s["frame_sha256"] == sample["frame_sha256"] or s["timestamp"] == sample["timestamp"] for s in combined):
                    combined.append(sample)
                    added = True
            merged[key] = combined[-MAX_SAMPLES:]
        if not added:
            return None
        data = {"version": 1, "schema": LEARNING_SCHEMA, "weapons": [
            {"id": ident, "variant": variant, "samples": samples}
            for (ident, variant), samples in sorted(merged.items(), key=lambda pair: (pair[0][0], pair[0][1] or ""))]}
        self._validate_learning(data)
        payload = json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8")
        if len(payload) > MAX_LEARNED_BYTES:
            raise ValueError("Learned weapons file exceeds 2 MiB")
        target = self._safe_target()
        fd, temporary = tempfile.mkstemp(prefix=".learned-weapons-", suffix=".tmp", dir=self._root)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            _check_cancel(self.cancel_event)
            self._safe_target()
            os.replace(temporary, target)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        # Deliberately do not alter self._loaded: learned fallback starts NEXT run.
        self._pending.clear()
        return target


def _event_time(row, *, name="row"):
    """Observation timestamp: read() actual_pts wins; plain time/t rows accepted."""
    if not isinstance(row, Mapping):
        raise TypeError(f"{name} must be a mapping")
    for key in ("actual_pts", "time", "t"):
        if key in row:
            return _number(row[key])
    raise ValueError(f"{name} is missing a timestamp (actual_pts/time/t)")


def _event_kind(event, *kinds):
    kind = event.get("kind", event.get("type"))
    return kind in kinds


def _usable(row, *, agreement=2):
    return (not row.get("excluded_view") and row.get("weapon") and row.get("slot") in (0, 1)
            and row.get("confidence", 0.) >= NAME_CONFIDENCE and row.get("name_view_agreement", 0) >= agreement)


def _same_weapon(a, b, *, agreement=2):
    return (_usable(a, agreement=agreement) and _usable(b, agreement=agreement)
            and a["slot"] == b["slot"] and _identity(a["weapon"]) == _identity(b["weapon"]))


def _ammo(row):
    value = row.get("ammo")
    confidence = row.get("ammo_score", 0.)
    if isinstance(value, dict):
        confidence, value = value.get("confidence", 0.), value.get("value")
    return value if type(value) is int and 0 <= value <= 1000 and confidence >= .88 else None


def context_from_observations(event, observations):
    """AHC nearby firing/held evidence; NEVER a causal kill-weapon attribution."""
    if not isinstance(event, Mapping):
        raise TypeError("event must be a mapping")
    second = _number(event.get("time", event.get("t")))
    timed = [(_event_time(row), row) for row in observations]
    rows = [row for _, row in sorted(
        {stamp: (stamp, row) for stamp, row in timed if second - 1.5 <= stamp <= second + .25}.values(),
        key=lambda item: item[0])]
    base = {"version": VERSION, "status": "unknown", "scope": "unknown", "label": "相关武器未确认",
            "weapon": None, "kill_attribution": False, "evidence": deepcopy(rows)}
    if any(row.get("excluded_view") or row.get("inactive_view") or row.get("inactive") for row in rows):
        return {**base, "reason": "view_change_or_inactive"}
    shots = []
    for before, after in zip(rows, rows[1:]):
        before_pts, after_pts = _event_time(before), _event_time(after)
        interval = after_pts - before_pts
        if after_pts > second + .04 or not 0 < interval <= .75 or not _same_weapon(before, after):
            continue
        similarity = _similar_observations(before, after)
        a, b = _ammo(before), _ammo(after)
        if a is None or b is None or not 0 < b < a <= 1000 or a - b > 50 or similarity < .80:
            continue
        shots.append({"weapon": deepcopy(after["weapon"]), "before_pts": before_pts,
                      "after_pts": after_pts, "slot": after["slot"], "ammo_before": a,
                      "ammo_after": b, "ammo_drop": a - b, "icon_similarity": round(similarity, 4)})
    if len({_identity(shot["weapon"]) for shot in shots}) > 1:
        return {**base, "reason": "multiple_firing_weapons", "shots": shots}
    if shots:
        weapon = shots[-1]["weapon"]
        return {**base, "status": "observed", "scope": "nearby_firing", "weapon": weapon,
                "label": "战果前射击：" + weapon["name"], "shots": shots,
                "reason": "same_slot_name_icon_and_ammo_decrease"}
    at_event = [row for row in rows if second - .04 <= _event_time(row) <= second + .25]
    if len(at_event) >= 2 and all(_same_weapon(at_event[0], row) and _similar_observations(at_event[0], row) >= .80 for row in at_event[1:]):
        weapon = deepcopy(at_event[0]["weapon"])
        return {**base, "status": "observed", "scope": "held_at_event", "weapon": weapon,
                "label": "当时持枪：" + weapon["name"], "reason": "two_frame_active_hud_slot"}
    usable = [row for row in rows if _usable(row)]
    if len(usable) >= 2 and all(_same_weapon(usable[-1], row) and _similar_observations(usable[-1], row) >= .80 for row in usable):
        weapon = deepcopy(usable[-1]["weapon"])
        # A readable intervening different weapon (even single-view) breaks the
        # chain; ignoring a weak but contradictory read would invent continuity.
        if all(not _usable(row, agreement=1) or _same_weapon(usable[-1], row, agreement=1) for row in rows):
            return {**base, "status": "observed", "scope": "held_before_event", "weapon": weapon,
                    "label": "战果前持枪：" + weapon["name"], "reason": "two_frame_active_hud_slot_before_event"}
    readable = [row for row in rows if _usable(row, agreement=1)]
    if len(readable) >= 3 and all(_same_weapon(readable[-1], row, agreement=1) and _similar_observations(readable[-1], row) >= .80 for row in readable):
        weapon = deepcopy(readable[-1]["weapon"])
        return {**base, "status": "observed", "scope": "held_before_event", "weapon": weapon,
                "label": "战果前持枪：" + weapon["name"], "reason": "three_frame_single_view_consensus"}
    return {**base, "reason": "insufficient_or_conflicting_hud_evidence"}


def annotate_events(events, observations, *, cancel_event=None):
    """Copy events; annotate shooting, and non-causal context on self outcomes."""
    _check_cancel(cancel_event)
    results = deepcopy(events)
    rows = list(observations)
    by_time = {round(_event_time(row), 6): row for row in rows}
    for event in results:
        _check_cancel(cancel_event)
        if not isinstance(event, Mapping):
            raise TypeError("event must be a mapping")
        second = _number(event.get("time", event.get("t")))
        if _event_kind(event, "shoot"):
            row = by_time.get(round(second, 6))
            match = None
            if row and not row.get("excluded_view") and row.get("weapon"):
                trusted = row.get("weapon_method") in ("calibrated_icon", "learned_icon_consensus") or row.get("confidence", 0.) >= .90
                if not trusted:
                    trusted = any(_event_time(other) != _event_time(row) and abs(_event_time(other) - second) <= 1.1
                                  and _same_weapon(row, other, agreement=1) and _similar_observations(row, other) >= .80 for other in rows)
                if trusted:
                    match = row["weapon"]
            event.update(weapon=match["name"] if match else "未知枪", weapon_id=match["id"] if match else None,
                         weapon_variant=match.get("variant") if match else None,
                         weapon_confidence=round(row["confidence"], 3) if match else 0.,
                         weapon_method=row["weapon_method"] if match else "unknown")
        elif _event_kind(event, "knock", "elimination", "assist", "kill"):
            if event.get("ownership") == "self" and event.get("confirmed") is not False:
                event["weapon_context"] = context_from_observations(event, rows)
            else:
                event["weapon_context"] = {"version": VERSION, "status": "unknown", "scope": "unknown",
                                           "weapon": None, "kill_attribution": False, "evidence": [],
                                           "reason": "unconfirmed_or_unsupported_event"}
    return results
