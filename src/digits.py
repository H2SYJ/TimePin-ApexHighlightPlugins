"""本人计分栏（击杀、助攻、伤害）数字的专用识别。

移植自 apex-highlight-clipper-codex/hud_digit_reader.py（双模型一致协议，阈值不变）：
- 模型会话改用 timepin_vision.model.load_onnx_session（含 SHA256 校验）；
- 复核引擎改为 src/ocr.create_ocr_engine（rec-only RapidOCR），显式 provider；
- 不运行时下载模型：复核模型缺失/校验失败直接报错，由打包/主程序保证就位。
"""

import hashlib
import time
from pathlib import Path

import cv2
import numpy as np

METHOD = "hud-digits-v1"
RESOURCE_DIR = Path(__file__).resolve().parent.parent / "resources"
MODEL_PATH = RESOURCE_DIR / "hud_digits_v1.onnx"
MODEL_SHA256 = "b59b9f9a7e174310fbf95931259b1765f8dba2f6a6f05de1bd699818ecf0f5f6"
HEIGHT, WIDTH = 28, 64
# 复核模型：RapidOCR 转换的 PP-OCRv6 small 识别模型（Apache-2.0），由主程序随包分发。
CHECK_SHA256 = "6f327246b50388f3c176ae304bd95767ea6dc0c9ae92153ef8cbe210b3c14884"
CHECK_PATH = RESOURCE_DIR / "models" / "ppocrv6_rec_small.onnx"
# 在没有参与训练的录像上校准（验证组与测试组共 1.5 万张）：主模型最低字符概率不低于 MODEL_MINIMUM、
# 复核模型置信度不低于 CHECK_MINIMUM 且两者一致时，伤害读错率约万分之七，击杀与助攻没有读错。
# 复核模型读出的内容里一个数字都没有（把数字读成符号，等于没给意见）时，主模型不低于 ALONE_MINIMUM 才单独采用；
# 复核模型读出了不同的数字一律不采用。校准集上单独采用的读数没有读错。
MODEL_MINIMUM = 0.5
CHECK_MINIMUM = 0.8
ALONE_MINIMUM = 0.97
# 采用的读数置信度取两者较低值，调用方按这个下限过滤。
MINIMUM = MODEL_MINIMUM
MAX_DIGITS = 5


def _require_model(path, sha256, label):
    if not Path(path).is_file():
        raise FileNotFoundError(f"{label}不存在: {path}（请由主程序分发该模型后重试）")
    data = Path(path).read_bytes()
    if sha256 is not None and hashlib.sha256(data).hexdigest() != sha256:
        raise RuntimeError(f"{label} SHA256 校验不符: {path}")
    return data


def prepare(crop):
    """裁片（BGR 或灰度，任意缩放）按高度缩放到 28 像素，左对齐放进 28x64 画布。"""
    gray = crop if crop.ndim == 2 else cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = gray.astype(np.float32) / 255.0
    width = max(4, round(gray.shape[1] * HEIGHT / gray.shape[0]))
    gray = cv2.resize(
        gray,
        (width, HEIGHT),
        interpolation=cv2.INTER_AREA if gray.shape[0] >= HEIGHT else cv2.INTER_LINEAR,
    )
    canvas = np.zeros((HEIGHT, WIDTH), np.float32)
    canvas[:, : min(WIDTH, width)] = gray[:, :WIDTH]
    return canvas


def decode(probs):
    """CTC 贪心解码；置信度取各输出字符的最大概率中最小的一个。"""
    best = probs.argmax(-1)
    result = []
    for row, path in zip(probs, best):
        text, previous, confidence = [], 0, 1.0
        for step, label in enumerate(path.tolist()):
            if label != previous and label != 0:
                text.append(str(label - 1))
                confidence = min(confidence, float(row[step, label]))
            previous = label
        result.append(("".join(text), confidence if text else 0.0))
    return result


def agreed(primary, check):
    """主模型与复核模型逐张比对：数字串完全相同且各自达到门槛才采用；复核模型没读出任何数字时见 ALONE_MINIMUM。"""
    result = []
    for (text, confidence), (other, other_confidence) in zip(primary, check):
        other = (other or "").strip()
        usable = bool(text) and len(text) <= MAX_DIGITS and confidence >= MODEL_MINIMUM
        if usable and other == text and float(other_confidence) >= CHECK_MINIMUM:
            result.append((text, min(confidence, float(other_confidence))))
        elif (
            usable
            and confidence >= ALONE_MINIMUM
            and not any(character.isdigit() for character in other)
        ):
            result.append((text, confidence))
        else:
            result.append(("", 0.0))
    return result


def load_primary_session():
    """加载计分数字主模型（严格 CPU provider，与上游标定时一致）。"""
    from timepin_vision.model import load_onnx_session

    data = _require_model(MODEL_PATH, MODEL_SHA256, "计分数字识别模型")
    return load_onnx_session(
        data,
        providers=["CPUExecutionProvider"],
        intra_op_num_threads=2,
        inter_op_num_threads=1,
    )


def load_check_engine():
    """加载 PP-OCRv6 复核识别引擎。两个模型都固定用 CPU：裁片宽度各不相同，
    GPU 动态形状反而慢，也避免不同后端读数不一致。"""
    from .ocr import create_ocr_engine

    _require_model(CHECK_PATH, CHECK_SHA256, "计分数字复核模型 PP-OCRv6")
    return create_ocr_engine(backend="cpu", model_path=CHECK_PATH, rec_batch_num=24)


class HudDigitReader:
    """与上游相同的调用方式：reader(images) -> ([(text, confidence), ...], elapsed)。

    会话/引擎按需延迟创建并缓存，避免握手阶段加载模型。
    """

    def __init__(self, primary_session=None, check_engine=None):
        self._primary = primary_session
        self._check_engine = check_engine

    def _primary_session(self):
        if self._primary is None:
            self._primary = load_primary_session()
        return self._primary

    def _check_recognizer(self):
        if self._check_engine is None:
            self._check_engine = load_check_engine()
        return self._check_engine.text_rec

    def __call__(self, images):
        started = time.perf_counter()
        if not images:
            return [], 0.0
        batch = np.stack([prepare(image) for image in images])[:, None]
        session = self._primary_session()
        input_name = session.get_inputs()[0].name
        primary = decode(session.run(None, {input_name: batch})[0])
        # 只复核主模型读得出的裁片。
        wanted = [
            index
            for index, (text, confidence) in enumerate(primary)
            if text and confidence >= MODEL_MINIMUM
        ]
        check = [("", 0.0)] * len(images)
        if wanted:
            reads, _ = self._check_recognizer()(
                [check_view(images[index]) for index in wanted]
            )
            for index, read in zip(wanted, reads):
                check[index] = (read[0].strip().rstrip(DIVIDER).strip(), float(read[1]))
        return agreed(primary, check), time.perf_counter() - started


# 伤害数字右侧紧贴斜分隔线，复核模型常把它读成末尾的「/」。
DIVIDER = "/|\\ "


def check_view(crop):
    """复核模型的输入（与校准时相同）：放大 3 倍；伤害这类宽裁片先去掉右侧贯穿的斜分隔线。"""
    from .stat_counter_fallback import without_counter_divider

    view = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    if crop.shape[1] >= 2 * crop.shape[0]:
        clean = without_counter_divider(view)
        if clean is not None:
            view = clean
    return view
