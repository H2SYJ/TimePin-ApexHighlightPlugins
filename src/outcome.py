"""本人中心战果提示（toast）与赛后全灭/夺冠确认 —— 移植自 AHC。

来源与版本（保持与上游一致，便于 parity 比对）：
- ``combat_outcome_reader.py`` 前缀逻辑：``RULE_VERSION='own-center-result-v4'``、
  ``CONFIRMATION_VERSION='prefix-confirmation-v2'``、
  ``TOAST_GRAMMAR_VERSION='toast-grammar-v1'``。
- ``outcome_reader.py`` 尾部逻辑：``SELF_VIEW_VERSION='self-hud-view-v3'``，
  以及 ``apex_clipper.inactive_view`` 的底部黑栏判定。

适配差异（相对上游）：
- 输入为宿主解码好的 BGR ndarray 帧与标量行（``row['time']`` 秒），不做视频
  解码、不读采样文件、无任务缓存/检查点。
- OCR 引擎经 ``src.ocr.create_full_ocr_engine(backend)`` 惰性创建并复用；
  det/cls 固定 CPU、rec 按 backend 指定 provider。引擎创建失败抛出
  带 backend 信息的 RuntimeError，绝不静默回退 CPU。
- 不携带上游原生目标名增强（CTC 裁剪/独立模型复核/词证据）与视频指纹校验；
  ``read_target_span``/``native_*`` 家族按任务约定省略，事件保留
  ``target_text`` 整行语法读数。
- 两帧确认与段级筛选集中在 :meth:`OutcomeReader.finalize_segments`；
  ``observe`` 只收集每帧 gate/OCR 观测与行级状态键。

``observe`` 写入的 row 键（宿主在调用 detect_events 前可获得）：
- ``inactive_view``: bool —— 底部黑色控制栏判定（观战/赛后面板）。
- ``friend_spectate``: bool —— inactive 帧底部绿色横幅（观战队友视角）。
- ``round_ended``: bool —— 赛后界面识别到全灭/夺冠文字。
- ``outcome_kind``: 'win'|'end'|None —— round_ended 时的文字分类。
- ``outcome_status``: 'self_event'|'ambiguous'|'negative'|'excluded_view'。
- ``outcome_toast``: list[dict] —— 该帧解析出的本人战果候选/确认事件。
- ``outcome_gate``: list[dict] —— 前缀图形模板命中（kind/action/score/box）。
"""

from __future__ import annotations

import math
import re
import threading
import unicodedata
from collections.abc import Sequence
from difflib import SequenceMatcher
from typing import Any

from .errors import Cancelled

KINDS = {"knock", "assist", "elimination"}
RULE_VERSION = "own-center-result-v4"
CONFIRMATION_VERSION = "prefix-confirmation-v2"
PARSER_VERSION = "target-name-v15"
TOAST_GRAMMAR_VERSION = "toast-grammar-v1"
GRAMMAR_LOCALES = {"zh-CN", "ru"}
SELF_VIEW_VERSION = "self-hud-view-v3"

VALID_BACKENDS = ("cpu", "dml", "cuda")

TARGET_PREFIX = re.compile(
    r"^(?:ASSIST[,.:\s]*)?(?:KN[O0]CKED\s*D[O0]WN|ELIMINATED)\s*"
    r"|^(?:助攻[,，:：\s]*)?(?:[擊击撃擎驛驿]倒|(?:已)?消[滅灭減减]|淘汰)\s*",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# 纯解析函数（与上游逐行一致；不改事件种类）
# ---------------------------------------------------------------------------


def clean_target_name(text: object, *, allow_rendered_square: bool = False) -> str:
    """只保留姓名；OCR 的固定提示、积分、对白不能作为玩家名展示。"""
    if not isinstance(text, str):
        return ""
    target = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip()
    # 也处理旧缓存曾把完整动作前缀写进 target_text 的情况。
    prefix = (
        r"^(?:(?:ASSIST|助攻)\s*[,.:，：]*\s*|KN[O0]CK(?:ED)?\s*D[O0]WN\s*"
        r"|ELIMINAT(?:ED|ION)\s*|(?:已)?[擊击擎驛驿]倒\s*|(?:已)?消[滅灭減减]\s*|淘汰\s*)"
    )
    for _ in range(3):
        stripped = re.sub(prefix, "", target, flags=re.IGNORECASE)
        if stripped == target:
            break
        target = stripped
    target = re.sub(
        r"\s*\+\s*[0-9OoIiLlSsBbZz]{1,5}(?:\s|$|[^A-Za-z0-9]).*$", "", target
    ).strip(" ,:;，：；")
    compact = re.sub(r"\s+", "", target).upper()
    if (
        not target
        or len(target) > 48
        or not (any(c.isalpha() for c in target) or re.fullmatch(r"\d{5,20}", target))
    ):
        return ""
    if (
        re.search(r"[\x00-\x1f�<>:=+!?！？。]", target)
        or "□" in target
        and not allow_rendered_square
    ):
        return ""
    if re.search(
        r"DAMAGE|INFLICT|SHIELDBROKEN|YOUSPOTTED|SPOTTEDANENEMY|PRESS|HOLD|TOGGLE"
        r"|RELOAD|ELIMINAT|KNOCK|KN0CK|ASSIST|SQUAD|擊倒|击倒|造成|傷害|伤害"
        r"|護盾|护盾|按住|按下|切換|切换|你發現|你发现|我被|你被",
        compact,
        re.IGNORECASE,
    ):
        return ""
    # 被截断的动作词（真实缓存有 ASSIST.KNO）没有可用姓名。
    if re.fullmatch(
        r"KN[O0]?(?:C(?:K(?:ED)?)?)?|D[O0]WN|ELIM|ELIMIN|ELIMINA|ELIMINATE|XP|RP",
        compact,
    ):
        return ""
    return target


# 俄语界面：默认识别模型不含西里尔字母，会把大写提示读成形近的拉丁字母（НОКАУТ→HOKAYT）。
# 只认真实录像里出现过的本人提示行首，且区分大小写，避免「Нокаут-щит」等物品名误触发。
_CYRILLIC_LOOKALIKE = str.maketrans("АВЕКМНОРСТУХЗДИЙЛПЯЬЫ", "ABEKMHOPCTYX3ANNANRbb")
RU_ASSIST_KNOCK = re.compile(
    r"^C[O0]A[EF][MNH][CG]T[BV][NM][EF]\s*[BV]\s*H[O0]KA[YV]T[EF]\s*"
)
RU_KNOCK = re.compile(
    r"^H[O0]KA[YV]T(?![-—–])\s*"
)  # 识别常把名字直接连在后面；「-щит」是物品名
RESULT_CUE = r"擊倒|击倒|消滅|消灭|淘汰|助攻|KNOCK|ELIMINAT|H[O0]KA[YV]T|C[O0]A[EF][MNH][CG]T[BV]"


def result_locale(text: str) -> str:
    """只按行首动作词判断提示语言，用于选择确认方式；不改变事件种类。"""
    if re.match(r"^(?:ASSIST|KNOCKED DOWN|ELIMINATED)\b", text, re.IGNORECASE):
        return "en"
    if RU_ASSIST_KNOCK.match(text) or RU_KNOCK.match(text):
        return "ru"
    if re.search(r"^(?:助攻\s*[,，:：]?\s*)?(?:擊倒|已消滅)", text):
        return "zh-TW"
    if re.search(r"^(?:助攻\s*[,，:：]?\s*)?(?:击倒|已消灭|淘汰)", text):
        return "zh-CN"
    return "zh"


def parse_result_line(text: str) -> dict[str, Any] | None:
    """只接受本人toast的行首语法；玩家名拼错不影响事件种类。"""
    clean = unicodedata.normalize("NFKC", text).strip()
    clean = re.sub(r"\s+", " ", clean)
    russian = clean.translate(_CYRILLIC_LOOKALIKE)
    for kind, action, pattern in [
        ("assist", "knock", RU_ASSIST_KNOCK),
        ("knock", "knock", RU_KNOCK),
    ]:
        match = pattern.match(russian)
        if match:
            target = clean[match.end() :].strip()
            return {
                "kind": kind,
                "action": action,
                "target_text": clean_target_name(target),
                "target_raw": target,
                "text": clean,
                "locale": "ru",
            }
    # OCR 常把同形的繁体“擊”读作日文“撃”；仅归一行首动作，不改昵称字符。
    clean = re.sub(
        r"^(助攻\s*[,，:：]?\s*)?撃倒",
        lambda match: (match[1] or "") + "擊倒",
        clean,
    )
    # Apex英文HUD常被OCR连成一词，且DOWN的O会读成0；仅修正行首固定语法。
    clean = re.sub(
        r"^KN[O0]CKED\s*D[O0]WN\s*", "KNOCKED DOWN ", clean, flags=re.IGNORECASE
    )
    clean = re.sub(
        r"^(ELIMINATED|ASSIST)\s*",
        lambda match: match.group(1).upper() + " ",
        clean,
        flags=re.IGNORECASE,
    )
    clean = re.sub(
        r"^(ASSIST)\s*[,.:]*\s*KN[O0]CK(?:ED)?\s*D[O0]WN\s*",
        r"ASSIST KNOCKED DOWN ",
        clean,
        flags=re.IGNORECASE,
    )
    clean = re.sub(
        r"^(ASSIST)\s*[,.:]*\s*ELIMINAT(?:ION|ED)\s*",
        r"ASSIST ELIMINATED ",
        clean,
        flags=re.IGNORECASE,
    )
    patterns = [
        ("assist", r"^(?:助攻\s*[,，:]?\s*|ASSIST\s*[: ,]*\s*)\s*(.*)$"),
        ("knock", r"^(?:(?:已)?(?:擊倒|击倒)|KNOCKED DOWN)\s*(.*)$"),
        ("elimination", r"^(?:已消滅|已消灭|ELIMINATED)\s*(.*)$"),
    ]
    for kind, pattern in patterns:
        match = re.match(pattern, clean, re.IGNORECASE)
        if not match:
            continue
        target = match.group(1).strip()
        action = kind if kind != "assist" else "unknown"
        if kind == "assist":
            # 只读助攻之后的完整动作前缀；昵称里出现动作词不能改变动作。
            # 简体界面的助攻消灭显示为「助攻，淘汰 <名字>」。
            for observed, syntax in [
                ("knock", r"^(?:[擊击撃擎驛驿]倒|KNOCK(?:ED)? DOWN)\s*"),
                ("elimination", r"^(?:(?:已)?消[滅灭減减]|淘汰|ELIMINAT(?:ION|ED))\s*"),
            ]:
                prefix = re.match(syntax, target, re.IGNORECASE)
                if prefix:
                    action = observed
                    target = target[prefix.end() :]
                    break
        return {
            "kind": kind,
            "action": action,
            "target_text": clean_target_name(target),
            "target_raw": target,
            "text": clean,
            "locale": result_locale(clean),
        }
    return None


def is_known_dialogue(text: str) -> bool:
    text = unicodedata.normalize("NFKC", text)
    if ":" in text and not re.match(
        r"^(?:助攻|ASSIST|已消滅|已消灭|擊倒|击倒|KNOCKED|ELIMINATED)",
        text,
        re.IGNORECASE,
    ):
        return True
    cue = re.search(RESULT_CUE + r"|撃倒", text, re.IGNORECASE)
    if not cue:
        return False
    prefix = text[: cue.start()]
    return bool(
        ":" in prefix
        or re.search(
            r"我被|你被|被人|I(?: AM| WAS)? |YOU(?: ARE| WERE)? ", prefix, re.IGNORECASE
        )
        or (not parse_result_line(text) and re.search(r"[。!?！？]", text))
    )


def group_lines(detections: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """合并同一行的不同颜色/分词框，保留字幕说话者和否定前缀。"""
    lines: list[dict[str, Any]] = []
    for item in sorted(detections, key=lambda x: (x["top"], x["left"])):
        cy = (item["top"] + item["bottom"]) / 2
        height = max(1, item["bottom"] - item["top"])
        possible = [
            line
            for line in lines
            if abs(cy - line["cy"]) <= min(height, line["height"]) * 0.65
        ]
        if possible:
            line = min(possible, key=lambda line: abs(cy - line["cy"]))
            line["items"].append(item)
            line["cy"] = sum((i["top"] + i["bottom"]) / 2 for i in line["items"]) / len(
                line["items"]
            )
            line["height"] = max(height, line["height"])
        else:
            lines.append({"cy": cy, "height": height, "items": [item]})
    result = []
    for line in lines:
        items = sorted(line["items"], key=lambda x: x["left"])
        groups = []
        for item in items:
            if groups and item["left"] - groups[-1][-1]["right"] <= 80:
                groups[-1].append(item)
            else:
                groups.append([item])
        for items in groups:
            result.append(
                {
                    "text": " ".join(i["text"] for i in items),
                    "confidence": min(i["confidence"] for i in items),
                    "prefix_confidence": items[0]["confidence"],
                    "box": [
                        min(i["left"] for i in items),
                        min(i["top"] for i in items),
                        max(i["right"] for i in items),
                        max(i["bottom"] for i in items),
                    ],
                    "parts": items,
                }
            )
    return result


def roi_image(image, profile: dict[str, Any]):
    h, w = image.shape[:2]
    rw, rh = profile["reference_size"]
    x1, y1, x2, y2 = profile["toast_roi"]
    box = [
        round(x1 * w / rw),
        round(y1 * h / rh),
        round(x2 * w / rw),
        round(y2 * h / rh),
    ]
    return image[box[1] : box[3], box[0] : box[2]], box


def text_line_gate(image, profile: dict[str, Any]) -> bool:
    """未知语言只用廉价文字行筛选触发OCR，不把几何命中当战果。"""
    import cv2
    import numpy as np

    region, _ = roi_image(image, profile)
    if not region.size:
        return False
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    mask = ((hsv[:, :, 2] > 180) & (hsv[:, :, 1] < 90)).astype(np.uint8)
    _, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    glyphs = [
        (x, y, w, h)
        for x, y, w, h, area in stats[1:]
        if 3 <= h <= 26 and 1 <= w <= 30 and 4 <= area <= 350
    ]
    return any(
        sum(abs(y + h / 2 - (yy + hh / 2)) <= 3 for _, yy, _, hh in glyphs) >= 8
        for _, y, _, h in glyphs
    )


def short_text_hint(image, profile: dict[str, Any]) -> bool:
    """「已消灭」这类三四个字的短提示达不到 text_line_gate 的字数；按提示区高度找一排浅色字形。"""
    import cv2
    import numpy as np

    region, _ = roi_image(image, profile)
    if not region.size:
        return False
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    mask = ((hsv[:, :, 2] > 185) & (hsv[:, :, 1] < 80)).astype(np.uint8)
    _, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    height = region.shape[0]
    low = max(4, int(height * 0.10))
    high = max(low + 2, int(height * 0.35))
    glyphs = [
        (y, h)
        for x, y, w, h, area in stats[1:]
        if low <= h <= high and 2 <= w <= high * 1.5 and area >= 6
    ]
    return any(
        sum(
            abs(y + h / 2 - (yy + hh / 2)) <= max(3, h * 0.5)
            and abs(hh - h) <= max(2, h * 0.4)
            for yy, hh in glyphs
        )
        >= 3
        for y, h in glyphs
    )


def same_toast_slot(left: dict[str, Any], right: dict[str, Any]) -> bool:
    """非拉丁名字的识别常逐帧不同；同种提示停在同一行位置即视为同一条。"""
    a, b = left.get("box"), right.get("box")
    if not a or not b:
        return False
    height = max(1, a[3] - a[1])
    return (
        abs(a[0] - b[0]) <= height * 1.5
        and abs((a[1] + a[3]) / 2 - (b[1] + b[3]) / 2) <= height * 0.6
    )


def grammar_confirmable(event: dict[str, Any]) -> bool:
    """无像素模板时：必须是行首语法、不是紧随对白的换行；名字读不清时，余下部分也不能像一句字幕。"""
    if event.get("after_dialogue"):
        return False
    if event.get("target_text"):
        return True
    raw = (event.get("target_raw") or "").strip()
    return not raw or len(raw) <= 24 and not re.search(r"[。！？!?，,：:；;、]", raw)


def same_english_target(left: str, right: str) -> bool:
    left = "".join(c for c in left.upper() if c.isalnum())
    right = "".join(c for c in right.upper() if c.isalnum())
    return bool(
        left
        and right
        and (
            left == right
            or min(len(left), len(right)) >= 4
            and SequenceMatcher(None, left, right).ratio() >= 0.82
        )
    )


def _same_line_gate(event: dict[str, Any], matches: Sequence[dict[str, Any]]) -> float:
    box = event["box"]
    cy = (box[1] + box[3]) / 2
    # 助攻优先；不能把“助攻，擊倒”中的擊倒当另一个本人独立击倒。
    relevant = [
        m
        for m in matches
        if m["kind"] == event["kind"]
        and abs((m["box"][1] + m["box"][3]) / 2 - cy) <= 10
        and box[0] - 5 <= m["box"][0] <= box[2] + 5
    ]
    return max((m["score"] for m in relevant), default=0.0)


def calibrated_sample_size(profile: dict[str, Any]) -> list[int] | None:
    """像素模板在标准界面上按 HUD 缩放；2Hz 采样宽 1280，高度随录像比例。精简界面尚未校准。"""
    if profile.get("ui_layout") == "compact":
        return None
    if not profile.get("layout_version"):
        return [1280, 540]
    width, height = profile["reference_size"]
    return [1280, round(1280 * height / width / 2) * 2]


def candidate_indices(candidate: dict[str, Any], fps: float) -> list[int]:
    first = max(1, math.ceil(candidate["start"] * fps + 0.5))
    last = math.ceil(candidate["end"] * fps + 0.5) - 1
    return list(range(first, last + 1))


# ---------------------------------------------------------------------------
# 视图判定（apex_clipper.inactive_view + outcome_reader 尾部）
# ---------------------------------------------------------------------------


def inactive_view(frame) -> bool:
    """观战和赛后面板都有底部黑色控制栏；本样本用来阻断他人的弹药信号。"""
    import numpy as np

    h, w = frame.shape[:2]
    bottom = frame[round(h * 0.985) :, round(w * 0.12) : round(w * 0.88)]
    if not bottom.size:
        return False
    return float(np.mean(np.max(bottom, axis=2) < 14)) > 0.72


def compact(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).upper()
    return "".join(c for c in text if c.isalnum())


def result_kind(text: str, inactive: bool) -> str | None:
    value = compact(text)
    if any(
        x in value
        for x in [
            "YOUARETHECHAMPION",
            "你已成为冠军",
            "你已成為冠軍",
            "你是冠军",
            "你是冠軍",
        ]
    ):
        return "win"
    if inactive and any(
        x in value
        for x in [
            "小队全灭",
            "小隊全滅",
            "游戏结束",
            "遊戲結束",
            "SQUADELIMINATED",
            "GAMEOVER",
        ]
    ):
        return "end"
    return None


def friend_view(image, inactive: bool) -> bool:
    import cv2
    import numpy as np

    if not inactive:
        return False
    h, w = image.shape[:2]
    banner = image[round(h * 0.93) : round(h * 0.98), round(w * 0.36) : round(w * 0.64)]
    hsv = cv2.cvtColor(banner, cv2.COLOR_BGR2HSV)
    green = (
        (hsv[:, :, 0] >= 24)
        & (hsv[:, :, 0] <= 95)
        & (hsv[:, :, 1] > 100)
        & (hsv[:, :, 2] > 100)
    )
    _, xx = np.nonzero(green)
    return bool(len(xx) >= 80 and xx.max() - xx.min() >= banner.shape[1] * 0.40)


def spectator_view(image, inactive: bool) -> bool:
    """本人 HUD 复核排除队友和敌人观战；不改变旧的对局结束判定。"""
    import cv2
    import numpy as np

    if not inactive:
        return False
    if friend_view(image, True):
        return True
    h, w = image.shape[:2]
    banner = image[round(h * 0.93) : round(h * 0.98), round(w * 0.36) : round(w * 0.64)]
    hsv = cv2.cvtColor(banner, cv2.COLOR_BGR2HSV)
    red = (
        ((hsv[:, :, 0] <= 15) | (hsv[:, :, 0] >= 170))
        & (hsv[:, :, 1] > 120)
        & (hsv[:, :, 2] > 100)
    )
    _, xx = np.nonzero(red)
    if len(xx) < 80 or xx.max() - xx.min() < banner.shape[1] * 0.25:
        return False
    # 观战同时具有顶部蓝/红色导航栏和底部姓名条；普通彩色场景不能仅凭颜色排除。
    top = image[: max(1, round(h * 0.055)), round(w * 0.1) : round(w * 0.9)]
    hsv = cv2.cvtColor(top, cv2.COLOR_BGR2HSV)
    navigation = (
        (
            (hsv[:, :, 0] <= 15)
            | (hsv[:, :, 0] >= 170)
            | ((hsv[:, :, 0] >= 100) & (hsv[:, :, 0] <= 135))
        )
        & (hsv[:, :, 1] > 120)
        & (hsv[:, :, 2] > 30)
        & (hsv[:, :, 2] < 200)
    )
    return float(np.mean(navigation)) > 0.35


def cap_segment_ends(
    segments: Sequence[dict[str, Any]],
    outcomes: Sequence[dict[str, Any]],
    tail: float = 2,
) -> list[dict[str, Any]]:
    result = []
    for s in segments:
        item = dict(s)
        endings = [e for e in outcomes if s["last_signal"] <= e["time"] < s["end"]]
        if endings:
            end = min(endings, key=lambda e: e["time"])
            item["end"] = min(s["end"], end["time"] + tail)
            item["end_evidence"] = end
        result.append(item)
    return result


def preserve_team_continuation(
    segments: Sequence[dict[str, Any]],
    rows: Sequence[dict[str, Any]],
    outcomes: Sequence[dict[str, Any]],
    duration: float,
    tail: float = 2,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for segment in segments:
        item = dict(segment)
        transitions = [
            r
            for r in rows
            if segment["last_signal"] <= r["time"] <= segment["end"] + 2
            and r.get("friend_spectate")
        ]
        if transitions:
            start = transitions[0]["time"]
            end_event = next((e for e in outcomes if e["time"] >= start), None)
            exit_row = next(
                (
                    r
                    for r in rows
                    if r["time"] > start + 1 and not r.get("inactive_view")
                ),
                None,
            )
            # 先返回本人视角时，不能跨越新局/复活后的长空档等待远处的结束画面。
            if end_event and (
                exit_row is None or end_event["time"] <= exit_row["time"] + 3
            ):
                item["end"] = max(item["end"], min(duration, end_event["time"] + tail))
                item["team_end_evidence"] = end_event
            else:
                # 录像边界内没有全灭/夺冠证据，保留友方视角到本次观战退出或文件末。
                item["end"] = max(
                    item["end"],
                    min(duration, exit_row["time"] + 15 if exit_row else duration),
                )
                item["team_end_needs_review"] = True
            item.setdefault("protected_reasons", []).append("friend_team_continuation")
        if result and item["start"] <= result[-1]["end"]:
            result[-1]["end"] = max(result[-1]["end"], item["end"])
            result[-1]["last_signal"] = max(
                result[-1]["last_signal"], item["last_signal"]
            )
            result[-1]["protected_reasons"] = list(
                set(
                    result[-1].get("protected_reasons", [])
                    + item.get("protected_reasons", [])
                )
            )
        else:
            result.append(item)
    return result


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------


class PrefixGate:
    """本人提示的行首图形模板匹配；助攻两张模板只含共用「助攻」字形。"""

    def __init__(self, profile: dict[str, Any]) -> None:
        import cv2

        self.profile = profile
        self.templates: list[tuple[dict[str, Any], Any]] = []
        from .resources import require_file

        for entry in profile.get("prefix_templates", []):
            path = require_file(entry["gray"])
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise RuntimeError(f"缺少本人结果提示模板或文件损坏: {path}")
            self.templates.append((entry, image))

    def inspect(self, image) -> list[dict[str, Any]]:
        import cv2

        region, roi = roi_image(image, self.profile)
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        found = []
        for entry, template in self.templates:
            if self.profile.get("hud_scale"):
                rw, rh = self.profile["reference_size"]
                sx, sy = self.profile["hud_scale"]
                fx = image.shape[1] / rw * 2 * sx
                fy = image.shape[0] / rh * 2 * sy
            else:
                fx = fy = image.shape[1] / entry["sample_size"][0]
            if fx != 1 or fy != 1:
                template = cv2.resize(template, None, fx=fx, fy=fy)
            if template.shape[0] > gray.shape[0] or template.shape[1] > gray.shape[1]:
                continue
            result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
            # 两张助攻模板只包含共用的“助攻”字形，路径名不能证明后续动作。
            action = "unknown" if entry["kind"] == "assist" else entry["kind"]
            # 同时出现多个同类提示时保留多个行位置，避免只支持最高匹配的一行。
            for n in range(4):
                _, score, _, loc = cv2.minMaxLoc(result)
                if n and score < self.profile.get("calibration", {}).get(
                    "gate_threshold", 0.65
                ):
                    break
                found.append(
                    {
                        "kind": entry["kind"],
                        "action": action,
                        "score": float(score),
                        "box": [
                            loc[0] + roi[0],
                            loc[1] + roi[1],
                            loc[0] + roi[0] + template.shape[1],
                            loc[1] + roi[1] + template.shape[0],
                        ],
                    }
                )
                x, y = loc
                th, tw = template.shape
                result[
                    max(0, y - th // 2) : min(result.shape[0], y + th // 2 + 1),
                    max(0, x - tw // 2) : min(result.shape[1], x + tw // 2 + 1),
                ] = -2
        return found


def recognize_frame(
    image,
    profile: dict[str, Any],
    engine,
    expand_context: bool = False,
    prefix_matches: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """toast ROI 整行 OCR + 行首语法；不含原生目标名裁片复核。"""
    import cv2

    if hasattr(engine, "text_det"):
        # RapidOCR 兼容引擎：放大检测上限，长 toast 行不能被默认 960 截断。
        try:
            from rapidocr_onnxruntime.ch_ppocr_det.utils import DetPreProcess

            engine.text_det.get_preprocess = lambda max_wh: DetPreProcess(
                960, "max", engine.text_det.mean, engine.text_det.std
            )
        except Exception:  # noqa: BLE001, S110 — optional preprocessing hint
            pass
    region, roi = roi_image(image, profile)
    if expand_context:
        pad = round(
            profile.get("context_padding_left", 100)
            * image.shape[1]
            / profile["reference_size"][0]
        )
        roi[0] = max(0, roi[0] - pad)
        region = image[roi[1] : roi[3], roi[0] : roi[2]]
    scaled = cv2.resize(region, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    recognized, _ = engine(scaled, use_cls=False)
    detections = []
    for points, text, confidence in recognized or []:
        xs = [p[0] / 2 + roi[0] for p in points]
        ys = [p[1] / 2 + roi[1] for p in points]
        detections.append(
            {
                "text": text,
                "confidence": float(confidence),
                "left": min(xs),
                "right": max(xs),
                "top": min(ys),
                "bottom": max(ys),
            }
        )
    lines = group_lines(detections)
    events = []
    ambiguous = []
    dialogue = []
    if not expand_context and any(
        line["box"][0] <= roi[0] + 4
        and not is_known_dialogue(line["text"])
        and re.search(RESULT_CUE, line["text"], re.IGNORECASE)
        for line in lines
    ):
        return recognize_frame(
            image, profile, engine, expand_context=True, prefix_matches=prefix_matches
        )
    for line in lines:
        if is_known_dialogue(line["text"]):
            dialogue.append(line)
            continue
        parsed = parse_result_line(line["text"])
        if parsed and line["prefix_confidence"] >= 0.65:
            supports = [
                match
                for match in prefix_matches or []
                if match["kind"] == parsed["kind"]
                and parsed["kind"] != "assist"
                and match["score"] >= 0.75
                and abs(match["box"][0] - line["box"][0]) <= 10
                and abs(
                    (
                        match["box"][1]
                        + match["box"][3]
                        - line["box"][1]
                        - line["box"][3]
                    )
                    / 2
                )
                <= max(10, (line["box"][3] - line["box"][1]) * 0.4)
            ]
            if parsed["kind"] == "assist" and parsed.get("action") == "knock":
                # 助攻共用模板不含“擊倒”；只有后面实际匹配到的完整擊倒图形
                # 才能作为姓名左边界，避免把“倒”的残竖笔裁成 I。
                font = max(1, line["box"][3] - line["box"][1])
                supports = [
                    match
                    for match in prefix_matches or []
                    if match["kind"] == "knock"
                    and match["score"] >= 0.80
                    and font * 0.8 <= match["box"][0] - line["box"][0] <= font * 4.5
                    and match["box"][2] < line["box"][2]
                    and abs(
                        (
                            match["box"][1]
                            + match["box"][3]
                            - line["box"][1]
                            - line["box"][3]
                        )
                        / 2
                    )
                    <= max(10, font * 0.4)
                ]
            if supports:
                line["prefix_box"] = max(supports, key=lambda m: m["score"])["box"]
            # 整行低置信度且无独立姓名复核时，名字不能采信。
            if line["confidence"] < 0.80:
                parsed["target_text"] = ""
            events.append(
                {
                    **parsed,
                    "confidence": line["prefix_confidence"],
                    "text_confidence": line["confidence"],
                    "box": line["box"],
                }
            )
        elif re.search(RESULT_CUE, line["text"], re.IGNORECASE):
            ambiguous.append(line)
    for event in events:
        # 紧挨在对白行下方的同色行可能是换行后的字幕，不能当本人提示。
        event["after_dialogue"] = any(
            0
            <= event["box"][1] - d["box"][3]
            <= max(6, (d["box"][3] - d["box"][1]) * 0.8)
            and abs(event["box"][0] - d["box"][0]) <= (d["box"][3] - d["box"][1]) * 3
            for d in dialogue
        )
    return {
        "lines": lines,
        "events": events,
        "ambiguous_lines": ambiguous,
        "known_dialogue": dialogue,
        "context_expanded": expand_context,
    }


class OutcomeReader:
    """逐帧收集战果提示/赛后界面观测，finalize 时做跨帧确认与段级决策。

    Parameters:
        profile: ``combat-outcome-profile.json`` 结构（含 ``prefix_templates``）。
        backend: 'cpu'|'dml'|'cuda' —— 传给 ``src.ocr.create_full_ocr_engine``。
        cancel_event: 可选协作取消；observe/finalize 途中检查。

    Usage:
        reader = OutcomeReader(profile)
        for frame in frames:
            reader.observe(frame.image, row)   # row: dict, 需 row['time']
        segments = reader.finalize_segments(candidates, rows, duration, sample_fps=fps)
    """

    # 每帧观测（内部，不进 row）：上游 cache['observations'] 的等价物。
    _OBS_KEYS = (
        "status",
        "events",
        "possible_events",
        "lines",
        "known_dialogue",
        "gate",
        "sample_size",
        "time",
        "frame",
        "index",
    )

    def __init__(
        self,
        profile: dict[str, Any],
        backend: str = "cpu",
        cancel_event: threading.Event | None = None,
    ) -> None:
        if backend not in VALID_BACKENDS:
            raise ValueError(
                f"backend must be one of {VALID_BACKENDS} (got {backend!r}); "
                "no silent CPU fallback is allowed"
            )
        if not isinstance(profile, dict):
            raise TypeError("profile must be the combat-outcome-profile dict")
        self.profile = profile
        self.backend = backend
        self.cancel_event = cancel_event
        self._gate: PrefixGate | None = None
        self._engine = None
        self._observations: dict[int, dict[str, Any]] = {}
        self._prev_inactive = False
        self._end_scan_remaining = 0
        self._index = 0
        self.outcome_events: list[dict[str, Any]] = []

    # -- OCR engine（惰性、每次运行一个） ------------------------------------

    def _ocr_engine(self):
        if self._engine is None:
            try:
                from .ocr import create_full_ocr_engine
            except ImportError as exc:
                raise RuntimeError(
                    "src.ocr.create_full_ocr_engine is required for outcome OCR "
                    "but is not available yet (HUD agent deliverable)"
                ) from exc
            try:
                self._engine = create_full_ocr_engine(backend=self.backend)
            except Exception as exc:
                raise RuntimeError(
                    f"create_full_ocr_engine(backend={self.backend!r}) failed: {exc}. "
                    "Install the requested ONNX provider or pick an available backend; "
                    "no silent CPU fallback."
                ) from exc
        return self._engine

    def _prefix_gate(self) -> PrefixGate:
        if self._gate is None:
            self._gate = PrefixGate(self.profile)
        return self._gate

    def _check_cancel(self) -> None:
        if self.cancel_event is not None and self.cancel_event.is_set():
            raise Cancelled()

    # -- 赛后界面 OCR ---------------------------------------------------------

    def _scan_end_text(self, image) -> str | None:
        """inactive 转换窗口内的中部 OCR；返回 'win'|'end'|None。"""
        h, w = image.shape[:2]
        region = image[
            round(h * 0.18) : round(h * 0.67), round(w * 0.20) : round(w * 0.80)
        ]
        engine = self._ocr_engine()
        result, _ = engine(region, use_cls=False)
        lines = [(x[1], float(x[2])) for x in result or []]
        text = " ".join(t for t, c in lines if c >= 0.60)
        return result_kind(text, True), text

    # -- 逐帧观察 -------------------------------------------------------------

    def observe(self, image, row: dict[str, Any]) -> None:
        """检查一帧 BGR ndarray 并把观测写入 ``row``。

        ``row['time']``（秒）必需；其余键由本方法写入/覆盖（见模块 docstring）。
        不存储图像；只保存标量与 dict 观测。
        """
        import numpy as np

        self._check_cancel()
        if not isinstance(image, np.ndarray) or image.ndim != 3:
            raise TypeError("image must be a BGR ndarray (H, W, 3)")
        if "time" not in row:
            raise ValueError("row must carry a 'time' value in seconds")
        t = float(row["time"])
        index = self._index
        self._index += 1
        row["index"] = index

        inactive = inactive_view(image)
        row["inactive_view"] = inactive
        friend = friend_view(image, inactive)
        row["friend_spectate"] = friend
        row["round_ended"] = False
        row["outcome_kind"] = None
        row["outcome_status"] = "negative"
        row["outcome_toast"] = []
        row["outcome_gate"] = []

        item: dict[str, Any] = {
            "index": index,
            "time": t,
            "frame": row.get("frame"),
            "events": [],
            "status": "negative",
            "sample_size": [image.shape[1], image.shape[0]],
        }

        # 赛后全灭/夺冠：只在 inactive 转换后的窗口内做中部 OCR（上游 chosen
        # 范围为 [i-2, i+10)；流式处理无法回溯 i-2，等价地覆盖 [i, i+10)）。
        if inactive and not self._prev_inactive:
            self._end_scan_remaining = 10
        self._prev_inactive = inactive
        if inactive and self._end_scan_remaining > 0:
            self._end_scan_remaining -= 1
            kind, text = self._scan_end_text(image)
            if kind:
                row["round_ended"] = True
                row["outcome_kind"] = kind
                row["outcome_text"] = text
                if not self.outcome_events or t - self.outcome_events[-1]["time"] > 12:
                    self.outcome_events.append(
                        {
                            "time": t,
                            "kind": kind,
                            "text": text,
                            "frame": row.get("frame"),
                        }
                    )

        # 战果提示：只读本人视角；inactive 与观战队友视角排除。
        own = not inactive
        if friend:
            own = False
        if not own:
            item["status"] = "excluded_view"
            row["outcome_status"] = "excluded_view"
            self._observations[index] = item
            return

        matches = self._prefix_gate().inspect(image)
        item["gate"] = matches
        row["outcome_gate"] = matches
        best = max((m["score"] for m in matches), default=0.0)
        threshold = self.profile.get("calibration", {}).get("gate_threshold", 0.65)
        generic = bool(
            self.profile.get("layout_version")
            and (
                text_line_gate(image, self.profile)
                or short_text_hint(image, self.profile)
            )
        )
        if best >= threshold or generic:
            result = recognize_frame(
                image, self.profile, self._ocr_engine(), prefix_matches=matches
            )
            item.update(lines=result["lines"], known_dialogue=result["known_dialogue"])
            possible = []
            for event in result["events"]:
                event = {**event, "sample_size": item["sample_size"]}
                support = _same_line_gate(event, matches)
                matching = [
                    m
                    for m in matches
                    if m["kind"] == event["kind"]
                    and abs(
                        (m["box"][1] + m["box"][3]) / 2
                        - (event["box"][1] + event["box"][3]) / 2
                    )
                    <= 10
                    and event["box"][0] - 5 <= m["box"][0] <= event["box"][2] + 5
                ]
                if matching:
                    strongest = max(matching, key=lambda m: m["score"])
                    event["prefix_box"] = strongest["box"]
                event["template_score"] = support
                potential = {
                    **event,
                    "time": t,
                    "frame": item["frame"],
                    "ownership": "self",
                }
                english = bool(
                    re.match(
                        r"^(?:ASSIST|KNOCKED DOWN|ELIMINATED)\b",
                        event["text"],
                        re.IGNORECASE,
                    )
                )
                locale = event.get("locale") or (
                    parse_result_line(event["text"]) or {}
                ).get("locale")
                if (
                    english
                    and event["confidence"] >= 0.90
                    and len(event.get("target_text", "")) >= 2
                    and support < threshold
                ):
                    potential["confirmation_basis"] = "english_toast_grammar"
                    possible.append(potential)
                elif (
                    locale in GRAMMAR_LOCALES
                    and support < threshold
                    and event["confidence"] >= 0.75
                    and grammar_confirmable(event)
                ):
                    potential["confirmation_basis"] = "toast_grammar"
                    possible.append(potential)
                elif event["confidence"] >= 0.85 and support >= threshold:
                    potential["confirmation_basis"] = "prefix_template"
                    item["events"].append(potential)
                elif event["confidence"] >= 0.65 and support >= 0.85:
                    potential["confirmation_basis"] = "two_frame_prefix_candidate"
                    possible.append(potential)
            if possible:
                item["possible_events"] = possible
            if item["events"]:
                item["status"] = "self_event"
            elif (
                possible
                or result["ambiguous_lines"]
                or best >= 0.84
                and not result["known_dialogue"]
            ):
                item["status"] = "ambiguous"
        item["status"] = item.get("status", "negative")
        row["outcome_status"] = item["status"]
        row["outcome_toast"] = list(item["events"]) + list(
            item.get("possible_events", [])
        )
        self._observations[index] = item

    # -- 跨帧确认 + 段级决策 ----------------------------------------------------

    def _confirm_sequences(self) -> None:
        """两帧确认（english/grammar/prefix）；等价上游 observe 内联确认，但延迟到 finalize。"""
        observations = self._observations
        for index in sorted(observations):
            item = observations[index]
            for event in item.get("possible_events", []):
                basis = event.get("confirmation_basis")
                if basis == "english_toast_grammar":
                    confirmed = any(
                        previous["time"] != item["time"]
                        and abs(previous["time"] - item["time"]) <= 1.1
                        and any(
                            e["kind"] == event["kind"]
                            and same_english_target(
                                e.get("target_text", ""), event.get("target_text", "")
                            )
                            for e in previous.get("possible_events", [])
                        )
                        for previous in observations.values()
                    )
                    if confirmed and not any(
                        e["kind"] == event["kind"]
                        and same_english_target(
                            e.get("target_text", ""), event.get("target_text", "")
                        )
                        for e in item["events"]
                    ):
                        item["events"].append(
                            {**event, "confirmation": "two_frame_english_toast"}
                        )
                elif basis == "toast_grammar":
                    confirmed = any(
                        previous["time"] != item["time"]
                        and abs(previous["time"] - item["time"]) <= 1.1
                        and any(
                            other.get("confirmation_basis") == "toast_grammar"
                            and other["kind"] == event["kind"]
                            and (
                                same_english_target(
                                    other.get("target_text", ""),
                                    event.get("target_text", ""),
                                )
                                or same_toast_slot(other, event)
                            )
                            for other in previous.get("possible_events", [])
                        )
                        for previous in observations.values()
                    )
                    if confirmed:
                        item["events"].append(
                            {**event, "confirmation": "two_frame_toast_grammar"}
                        )
                    elif (
                        event["confidence"] >= 0.93
                        and len(event.get("target_text") or "") >= 2
                    ):
                        item["events"].append(
                            {**event, "confirmation": "strong_single_toast_grammar"}
                        )
                elif basis == "two_frame_prefix_candidate":
                    support = event.get("template_score", 0.0)
                    confirmed = any(
                        abs(previous["time"] - item["time"]) <= 1.1
                        and previous["time"] != item["time"]
                        and any(
                            e["kind"] == event["kind"]
                            and (
                                min(support, e.get("template_score", 0)) >= 0.9
                                or same_toast_slot(e, event)
                            )
                            for e in previous.get("possible_events", [])
                        )
                        for previous in observations.values()
                    )
                    if confirmed:
                        item["events"].append(
                            {**event, "confirmation": "two_frame_prefix"}
                        )
            if item["events"]:
                item["status"] = "self_event"

    def _locale_confirmed(self, rows: Sequence[dict[str, Any]]) -> bool:
        if not self.profile.get("calibration", {}).get("require_traditional_ui", True):
            return True
        traditional = r"[擊滅敵槍衝鋒輕獵獸轉換瑪蘿]"
        for row in rows:
            if any(
                re.search(traditional, label.get("text", ""))
                for label in row.get("weapon_labels", [])
            ):
                return True
        for item in self._observations.values():
            if any(
                re.search(traditional, line["text"])
                for line in item.get("known_dialogue", [])
            ):
                return True
            if any(
                re.match(r"^(助攻|擊倒|已消滅)", event["text"])
                for event in item.get("events", [])
            ):
                return True
        return False

    def finalize_segments(
        self,
        segments: Sequence[dict[str, Any]],
        rows: Sequence[dict[str, Any]],
        duration: float,
        *,
        sample_fps: float = 2,
    ) -> list[dict[str, Any]]:
        """段级决策：全灭/夺冠截尾 → 队友视角续尾 → 战果提示 kept/rejected/review。

        每个返回的 segment 增加 ``combat_outcome`` dict：
        ``decision`` ('kept'|'rejected'|'review')、``confidence``、``events``、
        ``reason``、``coverage``；``own_kill`` 反证独立保留段。
        """
        self._check_cancel()
        self._confirm_sequences()
        # 尾部：先全灭/夺冠截尾，再队友视角续尾（与上游 apex_clipper 顺序一致）。
        capped = cap_segment_ends(segments, self.outcome_events)
        segments = preserve_team_continuation(
            capped, rows, self.outcome_events, duration
        )
        fps = float(sample_fps)
        calibration = self.profile.get("calibration", {})
        supported_style = self._locale_confirmed(rows) and all(
            r.get("sample_size") == calibrated_sample_size(self.profile)
            for r in self._observations.values()
            if r.get("status") != "excluded_view" or True
        )
        result = []
        # rows 顺序即 observe 顺序：index 由 observe 写入。
        for position, segment in enumerate(segments):
            self._check_cancel()
            indices = candidate_indices(segment, fps)
            # 上游帧号 = ceil(t*fps+0.5)（1 起始）；observe 存的是顺序 index（0 起始）。
            # 用 time 反推帧号对齐，避免「帧号 vs observe index」差 1 造成的假缺帧。
            by_frame: dict[int, dict[str, Any]] = {}
            for item in self._observations.values():
                try:
                    frame_no = math.ceil(item["time"] * fps + 0.5)
                except (TypeError, ValueError, OverflowError):
                    continue
                by_frame[frame_no] = item
            covered = [by_frame[i] for i in indices if i in by_frame]
            found = [
                e for i in indices for e in by_frame.get(i, {}).get("events", [])
            ]
            uncertain = [r for r in covered if r["status"] == "ambiguous"]
            reliable_absence = bool(
                len(covered) == len(indices)
                and indices
                and fps >= calibration.get("minimum_fps", 2)
                and calibration.get("negative_decision_enabled", False)
                and not uncertain
                and supported_style
            )
            counter = None
            if found:
                status = "kept"
                reason = "已确认本人击倒、助攻或消灭提示"
            elif reliable_absence:
                status = "rejected"
                reason = "完整检查候选后未发现本人击倒、助攻或消灭提示"
            else:
                status = "review"
                reason = "提示模糊、采样覆盖不足或尚未校准，需复核本人结果"
            if not found and "own_kill" in segment.get("combat_evidence", {}).get(
                "accepted_reasons", []
            ):
                # 本人 HUD 击杀数在交战内确认增长，与界面语言无关；文字提示读不到时
                # 用它保留交战，但不当成可计数的提示事件。
                status = "kept"
                reason = "本人击杀计数增长（界面语言的结果提示未识别）"
                counter = "own_kill_counter"
            confidence = 1.0 if status == "kept" and not counter else 0.0
            if status == "review":
                confidence = 0.5
            item = dict(segment)
            item["combat_outcome"] = {
                "decision": status,
                "confidence": confidence,
                "events": found,
                "reason": reason,
                **({"counter_basis": counter} if counter else {}),
                "coverage": {
                    "expected_frames": len(indices),
                    "checked_frames": len(covered),
                    "fps": fps,
                    "ambiguous_frames": [r.get("frame") for r in uncertain],
                    "reliable_absence": reliable_absence,
                    "calibrated_style_confirmed": supported_style,
                },
                "version": RULE_VERSION,
            }
            result.append(item)
        return result
