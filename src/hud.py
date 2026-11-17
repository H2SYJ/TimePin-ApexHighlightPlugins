"""HUD 帧识别：计分栏数字（双模型一致协议）+ 弹药 OCR + 武器图标签名 + 视野/倒地状态。

纯几何/像素函数从 apex-highlight-clipper-codex/apex_clipper.py 原样移植：
crop, numeric, ammo_number, icon_signature, icon_similarity, inactive_view,
downed_score（counter_offset 与模板缩放逻辑在 src/layout.py）。

HudReader 是帧批读入口：输入 BGR ndarray + 时间戳，输出与 AHC read_hud
行结构一致的 dict（t/ammo/damage/kills/inactive/downed/downed_score/
counter_marker_score/weapon_icon，另保留 *_text/*_score/kills_source/
counter_offset 供事件合并）。所有模型/模板按需加载并缓存；缺失资源
立即报错，绝不静默降级。
"""

import base64
import re

import cv2
import numpy as np

from .layout import counter_offset  # re-export: 布局检测与帧读共用同一锚点逻辑

_NUMERIC_LIMITS = {"damage": 20000, "kills": 60}
_NUMERIC_MINIMUM = 0.88


def crop(frame, box, reference):
    """按参考分辨率比例裁出 HUD 区域并放大 3 倍（与上游 crop 完全一致）。"""
    h, w = frame.shape[:2]
    rw, rh = reference
    x1, y1, x2, y2 = box
    region = frame[
        round(y1 * h / rh) : round(y2 * h / rh), round(x1 * w / rw) : round(x2 * w / rw)
    ]
    if not region.size:
        raise ValueError("HUD 识别区域为空")
    return cv2.resize(region, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)


def numeric(text, score, limit, minimum=_NUMERIC_MINIMUM):
    text = text.strip()
    if score < minimum or not re.fullmatch(r"\d+", text):
        return None
    value = int(text)
    return value if value <= limit else None


def ammo_number(text, score):
    # Apex 的斜杠零被中文识别模型稳定识别成“日”；只在弹药小区域修正。
    if "日" in text and score >= 0.67:
        text = text.replace("日", "0")
        return numeric(text, score, 1000, 0.67)
    if score >= 0.80 and re.fullmatch(r"[0-9O〇Ø]+", text):
        return numeric(
            text.translate(str.maketrans({"O": "0", "〇": "0", "Ø": "0"})),
            score,
            1000,
            0.80,
        )
    return numeric(text, score, 1000, 0.86)


def icon_signature(frame, profile):
    """武器图标边缘签名（base64 packbits，与上游逐位一致，可跨模块直接比对）。"""
    image = crop(frame, profile["weapon_icon"], profile["reference_size"])
    image = cv2.resize(image, (96, 28))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 65, 130) > 0
    return base64.b64encode(np.packbits(edges).tobytes()).decode("ascii")


def icon_similarity(a, b):
    """两个图标签名的 Dice 相似度；纯位运算，events 模块可直接复用本函数。"""
    x = np.unpackbits(np.frombuffer(base64.b64decode(a), dtype=np.uint8)).astype(bool)
    y = np.unpackbits(np.frombuffer(base64.b64decode(b), dtype=np.uint8)).astype(bool)
    intersection = np.count_nonzero(x & y)
    return 2 * intersection / max(np.count_nonzero(x) + np.count_nonzero(y), 1)


def inactive_view(frame):
    """观战/赛后视角：底部黑色控制栏存在时弹药信号不可信。"""
    h, w = frame.shape[:2]
    bottom = frame[round(h * 0.985) :, round(w * 0.12) : round(w * 0.88)]
    return float(np.mean(np.max(bottom, axis=2) < 14)) > 0.72


def downed_score(frame, profile, template):
    """倒地护盾模板在半分辨率搜索区的匹配分（与上游 downed_score 一致）。"""
    from .layout import template_for

    template = template_for(profile, template)
    rw, rh = profile["reference_size"]
    x1, y1, x2, y2 = [round(v / 2) for v in profile["downed_marker_search"]]
    ref = cv2.resize(frame, (rw // 2, rh // 2))
    region = cv2.cvtColor(ref[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    return float(
        cv2.minMaxLoc(cv2.matchTemplate(region, template, cv2.TM_CCOEFF_NORMED))[1]
    )


def _load_gray(relative):
    from .layout import RESOURCE_DIR

    path = RESOURCE_DIR / relative
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"缺少 HUD 模板资源: {path}")
    return image


def _check_cancel(cancel_event):
    if cancel_event is not None and cancel_event.is_set():
        raise InterruptedError("HUD 识别已取消")


class HudReader:
    """按帧批量读取 HUD 行数据。所有重资源（模板、模型、OCR 引擎）懒加载并缓存。

    HudReader(profile, backend='cpu', cancel_event=None)
        profile: detect_images 返回的布局配置（含 reference_size/counter_marker 等）。
        backend: 数字/弹药 OCR 后端 cpu|dml|cuda（计分主模型与复核模型按校准
                 约定固定 CPU，不受此参数影响——backend 只作用于 ammo 及
                 kills 图标补读的通用 rec 引擎）。
        cancel_event: threading.Event；每帧/每批检查一次，置位即抛 InterruptedError。

    read_batch(images, times) -> list[dict]
        images: BGR ndarray 列表；times: 等长秒时间戳（float/int）。
        空批不加载任何模型直接返回 []。
    """

    def __init__(self, profile, backend="cpu", cancel_event=None):
        if "reference_size" not in profile or "counter_marker_search" not in profile:
            raise ValueError(
                "profile 不是 detect_images/make_profile 产物：缺少 reference_size/counter_marker_search"
            )
        self.profile = profile
        self.backend = backend
        self.cancel_event = cancel_event
        self._marker = None
        self._downed = None
        self._digit_reader = None
        self._counter_fallback = None
        self._ammo_engine = None

    # -- 懒加载资源 -------------------------------------------------------
    @property
    def marker(self):
        if self._marker is None:
            self._marker = _load_gray(self.profile["counter_marker"])
        return self._marker

    @property
    def downed_marker(self):
        if self._downed is None and self.profile.get("downed_marker"):
            self._downed = _load_gray(self.profile["downed_marker"])
        return self._downed

    @property
    def digit_reader(self):
        """计分数字双模型读数器（主模型 + PP-OCRv6 复核，固定 CPU）。"""
        if self._digit_reader is None:
            from .digits import HudDigitReader

            self._digit_reader = HudDigitReader()
        return self._digit_reader

    @property
    def counter_fallback(self):
        """图标锚定的击杀补读（KILLS_FALLBACK 协议必需，缺模板即报错）。"""
        if self._counter_fallback is None:
            from .stat_counter_fallback import CounterFallback

            self._counter_fallback = CounterFallback()
        return self._counter_fallback

    @property
    def ammo_engine(self):
        """弹药/图标补读通用 rec-only 引擎（RapidOCR 打包默认模型）。"""
        if self._ammo_engine is None:
            from .ocr import create_ocr_engine

            self._ammo_engine = create_ocr_engine(
                backend=self.backend, rec_batch_num=24
            )
        return self._ammo_engine

    # -- 帧读 -------------------------------------------------------------
    def _prepare_frame(self, image):
        """单帧特征：锚点偏移、视野/倒地状态、武器签名、各 ROI 裁片。"""
        profile = self.profile
        delta, score = counter_offset(image, profile, self.marker)
        downed = None
        downed_value = None
        if self.downed_marker is not None:
            downed_value = downed_score(image, profile, self.downed_marker)
            downed = downed_value >= profile["downed_marker_threshold"]
        inactive = inactive_view(image)
        if downed:
            inactive = False  # 倒地视角保留弹药信号（与上游 enrich_lifecycle 一致）
        rois = {}
        for key in ("ammo", "damage", "kills"):
            box = profile[key]
            if key in ("damage", "kills"):
                if delta is None:
                    rois[key] = None
                    continue
                dx, dy = delta
                box = [box[0] + dx, box[1] + dy, box[2] + dx, box[3] + dy]
            rois[key] = crop(image, box, profile["reference_size"])
        # 固定击杀框读不到时的图标锚定补读裁片（已 3x 放大）。
        rois["kills_icon"] = self.counter_fallback.kills_crop(image, profile, delta)
        return {
            "counter_offset": delta,
            "counter_marker_score": float(score),
            "inactive_view": bool(inactive),
            "downed_score": None if downed_value is None else float(downed_value),
            "downed": None if downed is None else bool(downed),
            "weapon_icon": icon_signature(image, profile),
        }, rois

    def read_batch(self, images, times):
        """批量读取；times 必须与 images 等长。取消时抛出 InterruptedError。"""
        _check_cancel(self.cancel_event)
        if not images:
            return []
        if len(images) != len(times):
            raise ValueError("images 与 times 长度不一致")
        prepared = []
        for image in images:
            _check_cancel(self.cancel_event)
            if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
                raise ValueError("HUD 帧必须是 8-bit BGR ndarray")
            prepared.append(self._prepare_frame(image))
        _check_cancel(self.cancel_event)

        # 计分数字：双模型一致协议（damage/kills）；弹药用通用 rec 引擎。
        counter_slots = []  # (frame_index, key) -> crop index
        counter_crops = []
        ammo_slots = []
        ammo_crops = []
        for index, (_, rois) in enumerate(prepared):
            for key in ("damage", "kills"):
                if rois[key] is not None:
                    counter_slots.append((index, key))
                    counter_crops.append(rois[key])
            if rois["ammo"] is not None:
                ammo_slots.append(index)
                ammo_crops.append(rois["ammo"])
        counter_reads = [("", 0.0)] * len(counter_crops)
        if counter_crops:
            counter_reads, _ = self.digit_reader(counter_crops)
        ammo_reads = [("", 0.0)] * len(ammo_crops)
        if ammo_crops:
            ammo_reads, _ = self.ammo_engine.text_rec(ammo_crops)
        _check_cancel(self.cancel_event)

        counter_map = dict(zip(counter_slots, counter_reads))
        ammo_map = dict(zip(ammo_slots, ammo_reads))
        rows = []
        for index, (features, rois) in enumerate(prepared):
            row = {"t": float(times[index]), **features}
            for key in ("damage", "kills"):
                text, score = counter_map.get((index, key), ("", 0.0))
                row[key + "_text"] = text
                row[key + "_score"] = float(score)
                row[key] = numeric(text, score, _NUMERIC_LIMITS[key])
            text, score = ammo_map.get(index, ("", 0.0))
            row["ammo_text"] = text
            row["ammo_score"] = float(score)
            row["ammo"] = ammo_number(text, score)
            rows.append(row)

        # 固定击杀框读不到时：图标锚定补读；骷髅图标旁单个细体 0 读成
        # 同形字（D/口/O…）且像素拓扑是居中闭合环时按 0 计。
        retry = [
            (index, rois["kills_icon"])
            for index, (_, rois) in enumerate(prepared)
            if rows[index]["kills"] is None and rois.get("kills_icon") is not None
        ]
        if retry:
            _check_cancel(self.cancel_event)
            retry_reads, _ = self.ammo_engine.text_rec([crop_ for _, crop_ in retry])
            from .stat_counter_fallback import ZERO_ALIASES, zero_glyph_proof

            for (index, icon), (text, score) in zip(retry, retry_reads):
                row = rows[index]
                value = numeric(text, score, 60)
                if value is not None:
                    row.update(
                        kills=value,
                        kills_text=text,
                        kills_score=float(score),
                        kills_source="icon_anchor",
                    )
                elif (
                    (text or "").strip() in ZERO_ALIASES
                    and row.get("kills_text", "").strip() in ZERO_ALIASES
                    and zero_glyph_proof(icon) is not None
                ):
                    row.update(
                        kills=0,
                        kills_text=text,
                        kills_score=float(score),
                        kills_source="icon_anchor_zero_ring",
                    )
        return rows
