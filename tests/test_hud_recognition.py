"""HUD 识别单测：数字 CTC 预处理/解码/双模型一致协议、数值解析、图标签名、
HudReader 帧批读（含击杀图标补读与 0 环证明）。

模型相关用例分两层：协议层用注入的伪会话/伪引擎（不依赖 ONNX）；
真实主模型存在时另跑一条 ONNX 形状校验。
"""
import sys
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.digits import (
    CHECK_MINIMUM,
    MODEL_MINIMUM,
    MODEL_PATH,
    MODEL_SHA256,
    HudDigitReader,
    agreed,
    check_view,
    decode,
    prepare,
)
from src.hud import (
    HudReader,
    ammo_number,
    icon_signature,
    icon_similarity,
    inactive_view,
    numeric,
)
from src.layout import RESOURCE_DIR, make_profile, template_for


def probs_for(labels, confidence=.99, steps=64):
    """按时间步给出 CTC 输出：labels 中 0 为空白，1..10 对应数字 0..9。"""
    probs = np.full((steps, 11), (1 - confidence) / 10, np.float32)
    path = list(labels) + [0] * (steps - len(labels))
    for step, label in enumerate(path):
        probs[step, label] = confidence
    return probs


def _marker():
    marker = cv2.imread(str(RESOURCE_DIR / 'templates/damage-marker.png'), cv2.IMREAD_GRAYSCALE)
    if marker is None:
        raise unittest.SkipTest('缺少 resources/templates/damage-marker.png')
    return marker


def _frame_with_marker(profile, fill=70):
    """把计分图标贴到 half-res 画布的锚点上再放大；底色用亮色避免误判 inactive。"""
    marker = _marker()
    resized = template_for(profile, marker)
    width, height = profile['reference_size']
    canvas = np.full((height // 2, width // 2, 3), fill, dtype=np.uint8)
    x = round(profile['counter_marker_origin'][0] / 2)
    y = round(profile['counter_marker_origin'][1] / 2)
    canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized[:, :, None]
    return cv2.resize(canvas, (width, height), interpolation=cv2.INTER_NEAREST)


def _ring_glyph():
    """合成满足 zero_glyph_proof 的单个闭合环：60x90 黑底、居中椭圆环。"""
    image = np.zeros((90, 60, 3), np.uint8)
    cv2.ellipse(image, (30, 45), (11, 20), 0, 0, 360, (255, 255, 255), 5)
    return image


class DigitModelTests(unittest.TestCase):
    def test_prepare_scales_any_crop_to_the_model_canvas(self):
        for shape in [(28, 64, 3), (84, 192, 3), (41, 95, 3), (20, 13)]:
            canvas = prepare(np.full(shape, 255, np.uint8))
            self.assertEqual(canvas.shape, (28, 64))
            self.assertAlmostEqual(float(canvas[:, 0].mean()), 1., places=3)
        # 窄裁片左对齐，右侧补 0。
        self.assertEqual(float(prepare(np.full((28, 12, 3), 255, np.uint8))[:, 20:].max()), 0.)

    def test_ctc_decode_collapses_repeats_and_keeps_digits_split_by_blank(self):
        probs = np.stack([probs_for([4, 4, 0, 1, 1, 0, 9]), probs_for([2, 0, 2]), probs_for([])])
        self.assertEqual([text for text, _ in decode(probs)], ['308', '11', ''])
        self.assertEqual(decode(probs)[2][1], 0.)

    def test_reading_needs_both_models_to_agree(self):
        primary = [('718', .99), ('718', .99), ('718', MODEL_MINIMUM - .01), ('718', .99), ('', 0.), ('771', .98), ('771', .95)]
        check = [('718', .95), ('78', .99), ('718', .99), ('718', CHECK_MINIMUM - .01), ('5', .99), ('π', .9), ('π', .9)]
        # 复核模型把数字读成符号时只采用主模型的高置信读数；读出不同数字一律不采用。
        self.assertEqual(agreed(primary, check), [('718', .95), ('', 0.), ('', 0.), ('', 0.), ('', 0.), ('771', .98), ('', 0.)])

    def test_reader_only_checks_crops_the_main_model_read(self):
        session = SimpleNamespace(get_inputs=lambda: [SimpleNamespace(name='x')])
        session.run = lambda outputs, feeds: [np.stack([probs_for([8, 2]), probs_for([])])]
        seen = []

        def check(images):
            seen.append(len(images))
            # 复核模型把伤害右侧的分隔斜线读成末尾的「/」时去掉再比对。
            return [('71/', .97)], 0.

        engine = SimpleNamespace(text_rec=check)
        values, _ = HudDigitReader(primary_session=session, check_engine=engine)(
            [np.zeros((28, 30, 3), np.uint8)] * 2)
        self.assertEqual(seen, [1])
        self.assertEqual(values, [('71', .97), ('', 0.)])

    def test_bundled_primary_model_matches_checksum_and_shape(self):
        import hashlib
        data = MODEL_PATH.read_bytes()
        self.assertEqual(hashlib.sha256(data).hexdigest(), MODEL_SHA256)
        session = HudDigitReader()._primary_session()
        name = session.get_inputs()[0].name
        probs = session.run(None, {name: np.zeros((1, 1, 28, 64), np.float32)})[0]
        self.assertEqual(probs.shape[0], 1)
        self.assertEqual(probs.shape[-1], 11)

    def test_missing_check_model_raises_actionable_error(self):
        import src.digits as digits_module
        missing = MODEL_PATH.parent / 'models' / 'definitely-absent.onnx'
        with patch.object(digits_module, 'CHECK_PATH', missing):
            with self.assertRaises(FileNotFoundError):
                digits_module.load_check_engine()

    def test_check_view_strips_wide_divider_but_keeps_narrow_crops(self):
        narrow = np.full((20, 30, 3), 40, np.uint8)
        self.assertEqual(check_view(narrow).shape, (60, 90, 3))
        wide = np.full((20, 60, 3), 40, np.uint8)
        cv2.line(wide, (52, 3), (42, 17), (255, 255, 255), 2)
        view = check_view(wide)
        self.assertEqual(view.shape[0], 60)
        self.assertLess(view.shape[1], 180)  # 右侧斜分隔线被去掉


class NumericParserTests(unittest.TestCase):
    def test_numeric_threshold_limit_and_format(self):
        self.assertEqual(numeric('718', .9, 20000), 718)
        self.assertIsNone(numeric('718', .87, 20000))
        self.assertIsNone(numeric('71a', .9, 20000))
        self.assertIsNone(numeric('20001', .9, 20000))

    def test_ammo_slash_zero_and_circle_aliases(self):
        self.assertEqual(ammo_number('2日', .7), 20)
        self.assertIsNone(ammo_number('2日', .66))
        self.assertEqual(ammo_number('O3', .85), 3)
        self.assertEqual(ammo_number('45', .9), 45)
        # 纯数字也走 0.80 的别名分支；更低分数回落到 0.86 门槛被拒。
        self.assertEqual(ammo_number('45', .81), 45)
        self.assertIsNone(ammo_number('45', .75))
        self.assertIsNone(ammo_number('O3', .75))


class IconSignatureTests(unittest.TestCase):
    def test_identical_crops_have_perfect_similarity(self):
        profile = make_profile(2560, 1080)
        frame = np.full((1080, 2560, 3), 70, np.uint8)
        x1, y1, x2, y2 = profile['weapon_icon']
        # 纯色区域的 Canny 边缘为空：签名全 0 时相似度按上游定义为 0。
        self.assertEqual(icon_similarity(icon_signature(frame, profile),
                                       icon_signature(frame, profile)), 0.)
        cv2.rectangle(frame, (x1 + 5, y1 + 5), (x2 - 5, y2 - 5), (255, 255, 255), 3)
        signature = icon_signature(frame, profile)
        self.assertEqual(icon_similarity(signature, signature), 1.)
        other = np.full((1080, 2560, 3), 70, np.uint8)
        cv2.rectangle(other, (x1 + 10, y1 + 10), (x2 - 10, y2 - 10), (255, 255, 255), 6)
        self.assertLess(icon_similarity(signature, icon_signature(other, profile)), .8)


class HudReaderTests(unittest.TestCase):
    def _reader(self, profile):
        reader = HudReader(profile, backend='cpu')
        reader._marker = _marker()
        return reader

    def test_empty_batch_loads_no_engines(self):
        reader = HudReader(make_profile(2560, 1080))
        self.assertEqual(reader.read_batch([], []), [])
        self.assertIsNone(reader._ammo_engine)
        self.assertIsNone(reader._digit_reader)

    def test_cancel_event_stops_before_reading(self):
        cancel = threading.Event()
        cancel.set()
        reader = HudReader(make_profile(2560, 1080), cancel_event=cancel)
        with self.assertRaises(InterruptedError):
            reader.read_batch([np.zeros((1080, 2560, 3), np.uint8)], [0.])
        self.assertIsNone(reader._ammo_engine)

    def test_mismatched_times_rejected(self):
        reader = HudReader(make_profile(2560, 1080))
        with self.assertRaises(ValueError):
            reader.read_batch([np.zeros((1080, 2560, 3), np.uint8)], [0., .5])

    def test_rows_carry_ahc_fields_with_injected_engines(self):
        profile = make_profile(2560, 1080)
        frame = _frame_with_marker(profile)
        reader = self._reader(profile)
        reader._digit_reader = lambda crops: ([('718', .99)] * len(crops), 0.)
        reader._counter_fallback = SimpleNamespace(kills_crop=lambda *a: None)
        reader._ammo_engine = SimpleNamespace(text_rec=lambda crops: ([('12', .9)] * len(crops), 0.))
        rows = reader.read_batch([frame], [1.5])
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row['t'], 1.5)
        self.assertEqual(row['damage'], 718)
        self.assertEqual(row['kills'], 718 if 718 <= 60 else None)  # kills 上限 60
        self.assertEqual(row['ammo'], 12)
        self.assertIn('weapon_icon', row)
        self.assertFalse(row['inactive_view'])
        self.assertGreaterEqual(row['counter_marker_score'], .78)
        self.assertIsNotNone(row['counter_offset'])

    def test_kills_zero_ring_fallback_marks_source(self):
        profile = make_profile(2560, 1080)
        frame = _frame_with_marker(profile)
        reader = self._reader(profile)
        # 固定框读成 O（同形字），图标补读也读成 D：像素是唯一闭合环时记 0。
        reader._digit_reader = lambda crops: ([('O', .5), ('O', .5)][:len(crops)], 0.)
        ring = _ring_glyph()
        reader._counter_fallback = SimpleNamespace(kills_crop=lambda *a: ring)
        calls = []

        def rec(crops):
            calls.append(len(crops))
            return [('D', .5)] * len(crops), 0.

        reader._ammo_engine = SimpleNamespace(text_rec=rec)
        rows = reader.read_batch([frame], [0.])
        row = rows[0]
        self.assertEqual(row['kills'], 0)
        self.assertEqual(row['kills_source'], 'icon_anchor_zero_ring')
        self.assertTrue(calls)  # ammo 与补读共用 rec 引擎

    def test_missing_marker_resource_errors_immediately(self):
        profile = make_profile(2560, 1080)
        reader = HudReader(profile)
        reader._marker = None
        with patch('src.hud._load_gray', side_effect=FileNotFoundError('缺少')):
            with self.assertRaises(FileNotFoundError):
                reader.read_batch([np.zeros((1080, 2560, 3), np.uint8)], [0.])

    def test_inactive_view_blocks_ammo_semantics(self):
        black = np.zeros((1080, 2560, 3), np.uint8)
        self.assertTrue(inactive_view(black))
        lit = np.full((1080, 2560, 3), 80, np.uint8)
        self.assertFalse(inactive_view(lit))


if __name__ == '__main__':
    unittest.main()
