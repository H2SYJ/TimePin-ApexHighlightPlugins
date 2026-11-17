# Apex Highlight

TimePin sidecar plugin for Apex Legends highlight detection. The plugin ports the apex-highlight-clipper (AHC) HUD recognition pipeline: ultrawide HUD layout detection, dual-model digit reading, combat-event extraction (shoot / hit / kill / downed), downed-protection intervals, and combat-outcome trimming (own result, squad wipe, champion, teammate spectate tail). Output is a versioned `Mark[]` list the host merges into the timeline.

## Features

- **自动交战识别**：击倒、全灭、队友交战、武器识别、统计信息
- **HUD 信息提取**：段位、英雄、玩家名、战斗统计（可选启用）
- **多模态融合**：视觉 + 音频 + 元数据综合判断
- **武器识别**：HUD 武器名 OCR + 图标目录 + 自学习样本

## Installation

### From TimePin Plugin Marketplace (Recommended)

1. Open TimePin → Plugins → Marketplace
2. Search "Apex Highlight"
3. Click "Install"

### Manual Installation

1. Download the latest `.tppkg` from [Releases](https://github.com/H2SYJ/TimePin-ApexHighlightPlugins/releases)
2. Open TimePin → Plugins → "Import Plugin"
3. Select the `.tppkg` file

## Quick Start

1. Record Apex Legends gameplay using **native recording** (OBS, GeForce Experience, AMD ReLive)
2. Open TimePin, drag the video into the timeline
3. Right-click the video → "Run Apex Highlight"
4. Wait for analysis to complete
5. View detected highlights on the timeline

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `backend` | string (`cpu` \| `dml` \| `cuda`) | `cpu` | ONNX Runtime execution provider. `dml` = DirectML (Windows GPU), `cuda` = NVIDIA CUDA |
| `sample_fps` | number (0, 120] | `2` | Frame extraction rate. Higher = more accurate but slower |
| `gap_seconds` | number ≥ 0 | `8` | Maximum gap to merge adjacent combat segments |
| `min_segment_seconds` | number > 0 | `3` | Minimum combat segment duration |
| `max_segment_seconds` | number > 0 | `90` | Maximum combat segment duration |
| `extract_mode` | string (`keyframes` \| `image2`) | `keyframes` | FFmpeg extraction mode: `keyframes` = I-frames only, `image2` = all frames at sample_fps |
| `enable_weapon_detection` | boolean | `true` | Enable weapon name OCR and icon learning |
| `hud_info_enabled` | boolean | `false` | Enable HUD info extraction (rank, hero, names, stats) — experimental |

## Supported Resolutions

| Aspect Ratio | Status | Notes |
|-------------|--------|-------|
| 16:9 (1920×1080, 2560×1440, 3840×2160) | ✅ Full support | Native ultrawide HUD layout |
| 16:10 (1920×1200, 2560×1600) | ✅ Supported | Pillarbox candidate in layout detection |
| 21:9 (2560×1080, 3440×1440, 5120×2160) | ✅ Full support | Primary target layout (AHC anchored-hud-v2) |
| 4:3 (1440×1080, 1600×1200) | ⚠️ Limited | Stretch/compact candidates; reduced accuracy |

**Note**: The plugin uses AHC's anchored-hud-v2 layout detection which natively supports ultrawide (21:9). Other aspect ratios work through pillarbox/stretch/compact fallback candidates. Unconfirmed layouts fall back with `layout_status: unconfirmed` and may have reduced accuracy.

## Supported Game Modes

| Mode | Support Level | Notes |
|------|--------------|-------|
| Battle Royale (BR) | ✅ Full | Primary target, all events detected |
| Arenas | ⚠️ Partial | HUD layout differs; combat events may work but HUD info unreliable |
| Mixtape / Control / Gun Game | ❌ Not supported | HUD layout significantly different |
| Firing Range | ❌ Not supported | No combat events to detect |

## Performance Reference

| Video Length | Resolution | Sample FPS | Backend | Est. Time | Disk Usage |
|-------------|------------|------------|---------|-----------|------------|
| 1 hour | 1920×1080 | 2 | CPU | ~8-12 min | ~500 MB temp |
| 1 hour | 2560×1440 | 2 | CPU | ~12-18 min | ~800 MB temp |
| 1 hour | 3440×1440 | 2 | DML (GPU) | ~4-6 min | ~800 MB temp |
| 30 min | 1920×1080 | 2 | CPU | ~4-6 min | ~250 MB temp |

- **CPU**: Uses 1-2 cores, < 2 GB RAM, no VRAM
- **DML/CUDA**: Offloads digit/OCR inference to GPU; < 4 GB VRAM
- Temp frames are cleaned up after each batch; only final marks and learned weapons persist

## Troubleshooting / FAQ

### Recognition not working / missing highlights

1. **HUD Customization**: The plugin expects **default HUD layout and scale**. If you changed:
   - HUD scale ≠ 100% → recognition will fail
   - Custom HUD positions → recognition will fail
   - Colorblind modes → may affect color-based markers

2. **Recording Source**: Must be **native recording** (OBS Game Capture, GeForce Experience, AMD ReLive).
   - ❌ Stream VODs (Twitch/YouTube) — re-encoded, bitrate artifacts
   - ❌ Clips edited in DaVinci/CapCut/Premiere — timeline altered
   - ❌ Phone recordings of monitor — perspective distortion

3. **Game Version**: Apex updates can change HUD layout. Check if plugin version matches your game version.

4. **Resolution/Aspect**: Ensure your recording resolution matches one of the supported aspect ratios above.

5. **UI Scale**: In Apex Settings → Video → "UI Scale" must be at default (usually 100% or "Normal").

### "Model file not found" error

- The plugin bundles `hud_digits_v1.onnx` and templates. If you see this error, the package may be corrupted.
- Re-download the `.tppkg` and re-import.

### "ONNX Runtime provider not available" error

- `backend: "dml"` requires Windows 10 1903+ and a DirectX 12 compatible GPU
- `backend: "cuda"` requires NVIDIA GPU + CUDA toolkit + `onnxruntime-gpu` package
- Fallback: Use `backend: "cpu"` (always works, slower)

### Plugin takes too long / high memory

- Reduce `sample_fps` to 1
- Use `extract_mode: "keyframes"` (default)
- Close other applications
- For GPU: ensure `backend` is set to `dml` or `cuda` with proper drivers

### No weapon names detected

- Weapon detection requires `enable_weapon_detection: true` (default)
- Weapon OCR works best at 1440p+; at 1080p accuracy drops
- Learning: the plugin learns weapon icons per output directory. First run on a video builds the local catalog.

### "Layout unconfirmed" in logs

- The plugin couldn't confidently detect the HUD layout
- Check resolution/aspect ratio matches supported list
- Ensure recording is not letterboxed/pillarboxed with black bars
- Try `extract_mode: "image2"` for more frame samples

## Compatibility

### Apex Legends Version

| Game Season | Plugin Version | Notes |
|------------|----------------|-------|
| Season 20+ | 0.1.0+ | Current HUD layout |
| Season 18-19 | 0.1.0+ | Compatible (minor HUD changes) |
| Season < 18 | ❌ | HUD layout significantly different |

**Important**: Apex HUD layout changes with major updates. The plugin version must track game updates. Check the plugin's changelog for compatibility notes.

### Recording Requirements

- ✅ OBS Studio (Game Capture / Window Capture)
- ✅ NVIDIA GeForce Experience (Alt+Z → Record)
- ✅ AMD ReLive
- ✅ Windows Game Bar (Win+G)
- ❌ Stream VODs (Twitch, YouTube, Bilibili)
- ❌ Edited/clipped videos
- ❌ Screen recordings with variable framerate (VFR) — use CFR

### Privacy & Network

- **完全本地运行** — 无任何网络请求、无遥测、无上传
- 模型文件随插件包分发，首次运行不下载模型
- Python 依赖由 TimePin 的内嵌 venv 机制在首次运行时自动安装（需联网），或使用离线包安装

## For Developers

### Building the Package

```bash
# Standard package
python build_package.py

# With offline dependencies (for air-gapped environments)
python build_package.py --include-offline-deps

# Custom output directory
python build_package.py --output-dir ./dist
```

### Development Setup

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements-test.txt
.venv\Scripts\python -m pytest tests/ -v
```

### Project Structure

```text
plugin.json              TimePin manifest
main.py                  JSON-RPC stdio entry point
src/                     Plugin source code
resources/               Models, templates, profiles (sha256 verified)
requirements.txt         Python dependencies
build_package.py         Package builder
README.md                This file
```

## License & Attribution

- Plugin code: MIT License (see LICENSE)
- HUD digit model (`hud_digits_v1.onnx`): ported from apex-highlight-clipper (AHC)
- Templates/assets: derived from Apex Legends game UI (EA/Respawn Entertainment)
- PP-OCRv6 model: Apache 2.0 (PaddlePaddle)

**Redistribution note**: Some assets derive from Apex Legends. Confirm redistribution rights before public release. The source AHC snapshot has no root LICENSE.

## Changelog

### 0.1.0 (Current)
- Initial release: combat detection, HUD info, weapon recognition
- Protocol v1.0.0, Python sidecar
- Bundled models and templates

## Support

- Issues: [GitHub Issues](https://github.com/H2SYJ/TimePin-ApexHighlightPlugins/issues)
- TimePin Discord: #plugins channel
- Documentation: [TimePin Plugin Development Guide](https://github.com/H2SYJ/TimePin/tree/main/docs/plugin-dev)