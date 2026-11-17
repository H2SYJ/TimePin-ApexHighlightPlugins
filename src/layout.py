"""按 HUD 锚点适配原始分辨率及拉伸比例；与 AHC hud_layout.py 保持一致。

从 apex-highlight-clipper-codex/hud_layout.py 移植（anchored-hud-v2）：
- 资源路径改为本插件的 resources/ 目录，不再引用上游 BASE；
- detect_images 直接接受 BGR ndarray 帧（视频取帧由主程序负责）；
- 去掉了依赖 PyAV 的 detect_video、写回相关的 ensure_activity_is_readable
  与 needs_layout_retry，以及 outcome 专用的 outcome_profile（outcome agent
  若需要可基于 make_profile + box 重建）。
"""

import json
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np

VERSION = "anchored-hud-v2"
# 标准界面的锚点区域（参考坐标 2560x1080）；与上游 make_profile 完全一致，已有缓存不受影响。
STANDARD_LAYOUT = {
    "assists": [2202, 94, 2228, 124],
    "participation": [2260, 94, 2300, 124],
    "rank_roi": [2440, 0, 2560, 108],
    "account_roi": [175, 958, 337, 989],
    "teammate_rois": [[136, 780, 292, 808], [136, 845, 292, 873]],
    "hero_tactical_roi": [555, 967, 653, 1035],
    "hero_ultimate_roi": [1243, 928, 1323, 1028],
}
# 游戏「界面配置模式：精简」：HUD 整体约放大 1.25 倍，计分栏改为「击杀/助攻/参与」一体条，
# 各组位置也重排，不能由标准坐标推出。以下坐标在 1280x720（4:3 拉伸）真实录像上实测，
# 以放大后的参考坐标表达，仍按各自屏幕边锚定，可随分辨率与拉伸换算。
COMPACT_SCALE = 1.25
COMPACT_LAYOUT = {
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
# 拉伸录像常见的游戏渲染比例。HUD 排在 16:9 安全区内，比 16:9 窄的画面按宽度缩小。
STRETCH_RENDERS = ((16, 9), (16, 10), (4, 3))
# 候选集合（比例×界面）的版本。旧任务冻结的布局若来自更早的检测，续接时重新检测一次；
# 标准布局几何不变时结果与缓存一致，不会触发重识别。
DETECTOR_VERSION = "fit-scale-candidates-v1"

# 本插件自带资源目录（独立计算，不等 src/resources.py，避免共享依赖）。
RESOURCE_DIR = Path(__file__).resolve().parent.parent / "resources"


def _load_profile(base):
    return (
        json.loads((RESOURCE_DIR / "profile-ultrawide.json").read_text("utf-8"))
        if base is None
        else base
    )


def _load_gray(relative):
    """资源缺失是可诊断错误，绝不静默返回 None 走旧分支。"""
    path = RESOURCE_DIR / relative
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"缺少 HUD 模板资源: {path}")
    return image


def point(profile, x, y, horizontal="right", vertical="top"):
    # 两侧带黑边时（4:3 游戏画面录进 16:9 文件），HUD 锚定在实际游戏画面的边上。
    left, top, width, height = profile.get("content_box") or [
        0,
        0,
        *profile["reference_size"],
    ]
    sx, sy = profile.get("hud_scale", [height / 1080, height / 1080])
    px = (
        x * sx
        if horizontal == "left"
        else width - (2560 - x) * sx
        if horizontal == "right"
        else width / 2 + (x - 1280) * sx
    )
    py = (
        y * sy
        if vertical == "top"
        else height - (1080 - y) * sy
        if vertical == "bottom"
        else height / 2 + (y - 540) * sy
    )
    return [round(left + px), round(top + py)]


def pillarbox_content(images, width, height):
    """左右对称、所有画面都不亮的纯黑边；返回实际游戏画面 [x, y, w, h]，没有黑边时返回 None。"""
    lit = None
    for image in images:
        if image.shape[:2] != (height, width):
            continue
        bright = image.max(axis=2) > 24
        if bright.mean() < 0.2:
            continue  # 载入黑屏不能证明边界
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


def box(profile, values, horizontal="right", vertical="top"):
    return point(profile, *values[:2], horizontal, vertical) + point(
        profile, *values[2:], horizontal, vertical
    )


def make_profile(
    width,
    height,
    vertical_scale=1.0,
    horizontal_scale=1.0,
    base=None,
    ui="standard",
    layout=None,
    content_box=None,
):
    original = _load_profile(base)
    compact = ui == "compact"
    # 精简界面的坐标以放大后的参考单位表示；标准界面保持原有键和数值不变。
    source = {**original, **STANDARD_LAYOUT, **(COMPACT_LAYOUT if compact else {})}
    k = COMPACT_SCALE if compact else 1.0
    profile = deepcopy(original)
    profile["reference_size"] = [width, height]
    if content_box:
        profile["content_box"] = list(content_box)
    scale_height = content_box[3] if content_box else height
    profile["hud_scale"] = [
        scale_height / 1080 * horizontal_scale * k,
        scale_height / 1080 * vertical_scale * k,
    ]
    profile["layout_version"] = VERSION
    profile["layout"] = layout or (
        "4:3-stretched" if abs(vertical_scale - 0.75) < 0.05 else "native"
    )
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
        profile,
        [search[0] - 40, max(0, search[1] - 64), search[2] + 60, search[3] + 80],
    )
    profile["assists"] = box(profile, source["assists"])
    profile["participation"] = box(profile, source["participation"])
    profile["counter_marker_origin"] = point(profile, *source["counter_marker_origin"])
    for key in ("ammo", "weapon", "weapon_icon"):
        profile[key] = box(profile, source[key], vertical="bottom")
    for key in ("weapon_name_boxes", "weapon_name_metrics"):
        profile[key] = [box(profile, b, vertical="bottom") for b in source[key]]
    for key in ("spectator", "downed_marker_search"):
        profile[key] = box(
            profile, original[key], horizontal="center", vertical="bottom"
        )
    profile["banner"] = box(
        profile, original["banner"], horizontal="center", vertical="center"
    )
    # Native sidecar crops use the same bottom-right quarter as the existing decoder.
    # 有黑边时按实际游戏画面取右下角，黑边以外不存在武器栏。
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
    profile["account_roi"] = box(
        profile, source["account_roi"], horizontal="left", vertical="bottom"
    )
    profile["teammate_rois"] = [
        box(profile, b, horizontal="left", vertical="bottom")
        for b in source["teammate_rois"]
    ]
    profile["hero_tactical_roi"] = box(
        profile, source["hero_tactical_roi"], horizontal="left", vertical="bottom"
    )
    profile["hero_ultimate_roi"] = box(
        profile, source["hero_ultimate_roi"], horizontal="center", vertical="bottom"
    )
    return profile


def layout_candidates(width, height, base=None, content_box=None):
    """列出录像可能的 HUD 几何：游戏渲染比例（原生/拉伸）× 界面模式（标准/精简）。

    content_box 为两侧黑边内的实际游戏画面；比例按它计算，坐标仍是整帧坐标。
    """
    candidates = []
    seen = set()
    aspect = (content_box[2] / content_box[3]) if content_box else width / height

    def add(vertical, horizontal, ui, layout=None):
        profile = make_profile(
            width,
            height,
            vertical,
            horizontal,
            base=base,
            ui=ui,
            layout=layout,
            content_box=content_box,
        )
        key = (ui, round(profile["hud_scale"][0], 3), round(profile["hud_scale"][1], 3))
        if key not in seen:
            seen.add(key)
            candidates.append(profile)

    for ui in ("standard", "compact"):
        # 旧版候选顺序保持在前，平分时仍优先原有结果。
        add(1.0, 1.0, ui)
        add(0.75, 1.0, ui)
        if aspect < 1.55:
            add(0.75, 0.75, ui)
        for render in [None, *STRETCH_RENDERS]:
            ratio = aspect if render is None else render[0] / render[1]
            if ratio > aspect + 0.01:
                continue  # 更宽的画面放进窄录像会留黑边，不是拉伸
            unit = min(1.0, ratio / (16 / 9))
            label = (
                "native"
                if render is None or abs(ratio - aspect) < 0.01
                else f"{render[0]}:{render[1]}-stretched"
            )
            add(unit, unit * aspect / ratio, ui, label)
    return candidates


def template_for(profile, template):
    sx, sy = profile.get("hud_scale", [1.0, 1.0])
    if abs(sx - 1) < 0.001 and abs(sy - 1) < 0.001:
        return template
    return cv2.resize(
        template,
        (max(1, round(template.shape[1] * sx)), max(1, round(template.shape[0] * sy))),
        interpolation=cv2.INTER_CUBIC,
    )


def counter_offset(frame, profile, template):
    """与上游 apex_clipper.counter_offset 一致：用计分图标做锚点定位可平移的数字框。

    返回 ((dx, dy), score)；未命中（score < .78）返回 (None, score)。
    """
    template = template_for(profile, template)
    # Apex 的计分 HUD 会因动画和数字宽度整体平移，用图标作锚点。
    ref = cv2.resize(
        frame, (profile["reference_size"][0] // 2, profile["reference_size"][1] // 2)
    )
    x1, y1, x2, y2 = [round(v / 2) for v in profile["counter_marker_search"]]
    region = cv2.cvtColor(ref[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    _, score, _, location = cv2.minMaxLoc(
        cv2.matchTemplate(region, template, cv2.TM_CCOEFF_NORMED)
    )
    if score < 0.78:
        return None, score
    origin = profile["counter_marker_origin"]
    return (
        (location[0] + x1) * 2 - origin[0],
        (location[1] + y1) * 2 - origin[1],
    ), score


def detect_images(images, width, height, base=None):
    """在最多 8 张 BGR 采样帧上选择 HUD 布局候选；输入帧尺寸不符的自动跳过。

    返回带 layout_status（confirmed/unconfirmed）与 layout_score 的 profile。
    与上游一致：得分不足或与第二名差距过小不静默选拉伸几何，回落到首个候选
    并标记 unconfirmed，由调用方决定是否用 confirm_by_name_row 复核。
    """
    original = _load_profile(base)
    marker = _load_gray(original["counter_marker"])
    candidates = layout_candidates(width, height, base=original)
    # 显卡按比例缩放时 4:3 画面会带左右黑边录进宽屏文件；只有确实检测到对称黑边才追加这些候选。
    content = pillarbox_content(images, width, height)
    if content:
        candidates += layout_candidates(
            width, height, base=original, content_box=content
        )
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
    if score < 0.78 or len(evidence) > 1 and score - evidence[1][0] < 0.025:
        # Do not silently select stretched geometry when there was no in-game HUD.
        profile = candidates[0]
        score = next(entry[0] for entry in evidence if entry[1] is profile)
        profile["layout_status"] = "unconfirmed"
    else:
        profile["layout_status"] = "confirmed"
    profile["layout_score"] = round(score, 4)
    return profile


def confirm_by_name_row(images, profile, minimum=2):
    """右上计分栏被直播叠加层遮住时，用左下本人名称行的固定语音图标确认原生标准布局。

    只接受标准界面、未拉伸的候选；图标须在至少两帧中落在本人名称行的高度上。
    """
    if profile.get("ui_layout") == "compact" or any(
        abs(v - profile["reference_size"][1] / 1080) > 0.001
        for v in profile.get("hud_scale", [1, 1])
    ):
        return None
    voice = _load_gray("templates/hud/voice.png")
    width, height = profile["reference_size"]
    x1, y1, x2, y2 = profile["account_roi"]
    hits = []
    for image in images:
        h, w = image.shape[:2]
        if (w, h) != (width, height):
            continue
        top = max(0, y1 - 8)
        band = cv2.cvtColor(
            image[top : min(h, y2 + 8), x1 : min(w, x2 + 260)], cv2.COLOR_BGR2GRAY
        )
        best = 0.0
        for factor in (0.75, 1.0, 1.25):
            template = cv2.resize(
                voice, None, fx=factor * h / 1080, fy=factor * h / 1080
            )
            if band.shape[0] < template.shape[0] or band.shape[1] < template.shape[1]:
                continue
            _, score, _, (_, y) = cv2.minMaxLoc(
                cv2.matchTemplate(band, template, cv2.TM_CCOEFF_NORMED)
            )
            if abs(top + y + template.shape[0] / 2 - (y1 + y2) / 2) <= 6 * h / 1080:
                best = max(best, float(score))
        hits.append(round(best, 3))
    return (
        {"name_row_scores": hits} if sum(s >= 0.85 for s in hits) >= minimum else None
    )
