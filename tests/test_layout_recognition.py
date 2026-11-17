"""布局检测单测：stretch/compact/pillarbox 候选确认、未确认回落、名称行复核。

合成方式与上游 test_hud_layout.py 相同：把 damage-marker 模板贴到
半分辨率画布的 counter_marker_origin 处，再放大回整帧。
"""

import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.layout import (
    RESOURCE_DIR,
    confirm_by_name_row,
    counter_offset,
    detect_images,
    layout_candidates,
    make_profile,
    template_for,
)


def _marker():
    marker = cv2.imread(
        str(RESOURCE_DIR / "templates/damage-marker.png"), cv2.IMREAD_GRAYSCALE
    )
    if marker is None:
        raise unittest.SkipTest("缺少 resources/templates/damage-marker.png")
    return marker


def _marker_image(profile, dy=0):
    marker = _marker()
    resized = template_for(profile, marker)
    width, height = profile["reference_size"]
    image = np.zeros((height // 2, width // 2, 3), dtype=np.uint8)
    x = round(profile["counter_marker_origin"][0] / 2)
    y = round(profile["counter_marker_origin"][1] / 2 + dy / 2)
    image[y : y + resized.shape[0], x : x + resized.shape[1]] = resized[:, :, None]
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_NEAREST)


class HudLayoutTests(unittest.TestCase):
    def test_stretched_counter_moved_below_old_search_is_confirmed(self):
        profile = make_profile(1920, 1080, 0.75)
        image = _marker_image(profile, dy=48)
        detected = detect_images([image], 1920, 1080)
        self.assertEqual(detected["layout_status"], "confirmed")
        self.assertEqual(detected["layout"], "4:3-stretched")
        offset, score = counter_offset(image, detected, _marker())
        self.assertGreaterEqual(score, 0.78)
        self.assertGreaterEqual(offset[1], 46)

    def test_pillarboxed_four_by_three_recording_anchors_hud_inside_black_bars(self):
        inner = make_profile(1920, 1440, 0.75)
        marker = _marker()
        resized = template_for(inner, marker)
        game = np.full((720, 960, 3), 70, dtype=np.uint8)
        x, y = (
            round(inner["counter_marker_origin"][0] / 2),
            round(inner["counter_marker_origin"][1] / 2),
        )
        game[y : y + resized.shape[0], x : x + resized.shape[1]] = resized[:, :, None]
        game = cv2.resize(game, (1920, 1440), interpolation=cv2.INTER_NEAREST)
        frame = np.zeros((1440, 2560, 3), np.uint8)
        frame[:, 320:2240] = game
        detected = detect_images([frame], 2560, 1440)
        self.assertEqual(detected["layout_status"], "confirmed")
        self.assertEqual(detected["content_box"], [320, 0, 1920, 1440])
        self.assertEqual(
            detected["damage"],
            [v + 320 * (i % 2 == 0) for i, v in enumerate(inner["damage"])],
        )
        self.assertNotIn("content_box", make_profile(2560, 1440))

    def test_sixteen_by_ten_hud_scales_by_width_not_height(self):
        # 16:10 的 HUD 排在 16:9 安全区内：3456x2160 实测缩放 1.8，不是按高度算的 2.0。
        expected = next(
            p
            for p in layout_candidates(3456, 2160)
            if [round(v, 3) for v in p["hud_scale"]] == [1.8, 1.8]
        )
        detected = detect_images([_marker_image(expected)], 3456, 2160)
        self.assertEqual(detected["layout_status"], "confirmed")
        self.assertEqual([round(v, 3) for v in detected["hud_scale"]], [1.8, 1.8])

    def test_compact_interface_layout_is_detected_separately(self):
        expected = make_profile(1280, 720, 0.75, ui="compact")
        detected = detect_images([_marker_image(expected)], 1280, 720)
        self.assertEqual(detected["layout_status"], "confirmed")
        self.assertEqual(detected.get("ui_layout"), "compact")
        self.assertEqual(detected["account_roi"], expected["account_roi"])
        # 标准界面的配置保持旧版字段，已有识别缓存可继续复用。
        self.assertNotIn("ui_layout", make_profile(1280, 720, 0.75))

    def test_blank_frames_leave_layout_unconfirmed(self):
        blank = np.zeros((1080, 1920, 3), np.uint8)
        detected = detect_images([blank, blank], 1920, 1080)
        self.assertEqual(detected["layout_status"], "unconfirmed")
        # 未确认时回落到首个候选而不是猜测拉伸几何。
        self.assertEqual(detected["layout_version"], "anchored-hud-v2")
        self.assertIn("layout_score", detected)

    def test_self_name_row_voice_icon_confirms_native_layout_only_on_two_frames(self):
        voice_path = RESOURCE_DIR / "templates/hud/voice.png"
        voice = cv2.imread(str(voice_path), cv2.IMREAD_GRAYSCALE)
        if voice is None:
            self.skipTest("缺少 resources/templates/hud/voice.png（已向主程序申报）")
        profile = make_profile(2560, 1080)
        x1, y1, x2, y2 = profile["account_roi"]
        top = round((y1 + y2) / 2 - voice.shape[0] / 2)
        frame = np.full((1080, 2560, 3), 40, np.uint8)
        frame[top : top + voice.shape[0], x1 + 120 : x1 + 120 + voice.shape[1]] = voice[
            :, :, None
        ]
        blank = np.full((1080, 2560, 3), 40, np.uint8)
        self.assertIsNotNone(confirm_by_name_row([frame, frame], profile))
        self.assertIsNone(confirm_by_name_row([frame, blank], profile))
        shifted = np.full((1080, 2560, 3), 40, np.uint8)
        shifted[
            top + 60 : top + 60 + voice.shape[0], x1 + 120 : x1 + 120 + voice.shape[1]
        ] = voice[:, :, None]
        self.assertIsNone(confirm_by_name_row([shifted, shifted], profile))
        self.assertIsNone(
            confirm_by_name_row([frame, frame], make_profile(2560, 1080, 0.75))
        )

    def test_missing_marker_resource_is_an_error_not_a_silent_layout(self):
        from unittest.mock import patch

        from src import layout

        with patch.object(layout, "_load_gray", side_effect=FileNotFoundError("缺少")):
            with self.assertRaises(FileNotFoundError):
                detect_images([np.zeros((1080, 1920, 3), np.uint8)], 1920, 1080)


if __name__ == "__main__":
    unittest.main()
