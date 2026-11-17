# Resource provenance

All files under this directory were copied byte-for-byte from the user-specified local `apex-highlight-clipper-codex-main` snapshot (`F:/apex-highlight-clipper-codex-main`). TPIN-7 brought the protocol skeleton assets; TPIN-10 added the HUD-info assets used by `src/hud_layout.py`, `src/rank_reader.py`, `src/hero_reader.py`, `src/name_reader.py`, and `src/stats.py`.

## Core (TPIN-7)

| Destination | Source in AHC | SHA-256 |
| --- | --- | --- |
| `hud_digits_v1.onnx` | `assets/counters/hud_digits_v1.onnx` | `b59b9f9a7e174310fbf95931259b1765f8dba2f6a6f05de1bd699818ecf0f5f6` |
| `profile-ultrawide.json` | `profile-ultrawide.json` | `5440b2a2f1953017268451dd279878910f209ace34d0f926b02d4bc661751b33` |
| `templates/damage-marker.png` | `templates/damage-marker.png` | `9fbe4bc9558f59e13d23f29ac0104e73bed1dc31a0fa1869a28934ae53c1b5b7` |
| `templates/knockdown-shield.png` | `templates/knockdown-shield.png` | `b178474a7ff3ceaf710085c7914574255d10843e3fbbe5f789e49810f21ea484` |

## HUD info (TPIN-10)

Per-file `sources.json` manifests (AHC originals, sha256-verified) sit beside the assets in `ranks/`, `legends/`, `legends/hud/`, `legends/portraits/`, and `templates/statistics/` — the hashes below cover the copies that have no per-file manifest.

| Destination | Source in AHC | SHA-256 |
| --- | --- | --- |
| `ranks/{bronze,silver,gold,platinum,diamond,master,predator,rookie,unranked}.png` | `assets/ranks/*.png` | manifest in `ranks/sources.json` |
| `modes/mode-clock.png` | `assets/modes/mode-clock.png` | `b93debe24b29875aa1c7a62b6393d6230126b32bffc181fc63178a31ed346717` |
| `modes/mode-heart.png` | `assets/modes/mode-heart.png` | `e51b7e31abd092a4ea7bae5087fa72f3a2bacf78579f7c3ab540d657a3b245df` |
| `legends/*-{tactical,ultimate}.png` (56) | `assets/legends/*-{tactical,ultimate}.png` | manifest in `legends/sources.json` |
| `legends/hud/*.png` (10) | `assets/legends/hud/*.png` | manifest in `legends/hud/sources.json` |
| `legends/portraits/*.png` (36) | `assets/legends/portraits/*.png` | manifest in `legends/portraits/sources.json` |
| `templates/voice.png` | `templates/voice.png` | `ba215b557205762623097c42fa58f27b3706de2587b15f18e0a68687e851f358` |
| `templates/statistics/*.png` (7) | `templates/statistics/*.png` | manifest in `templates/statistics/sources.json` |

## Notes

- `profile-ultrawide.json` paths resolve relative to this `resources/` directory. HUD-info ROIs (`rank_roi`, `account_roi`, `teammate_rois`, `hero_*_roi`) are **not** in that file — `src/hud_layout.STANDARD_LAYOUT` / `COMPACT_LAYOUT` adds them, matching the AHC anchored-hud-v2 scheme.
- `hud_digits_v1.onnx` is the CNN+CTC counter decoder (AHC `hud_digit_reader`). The optional consensus check model `ppocrv6_rec_small.onnx` is **not** bundled — primary decode with a high-confidence cutoff is the only path.
- AHC `name_rec_japanese_v4.onnx` and `numeric_rec_en_v4.onnx` (latin-glyph review) are **not** bundled; `src/name_reader.py` uses the multilingual OCR engine from the `timepin-vision` SDK on caption views instead.
- The source snapshot has no root LICENSE. Its README describes an unofficial community tool without granting redistribution rights; some templates derive from Apex Legends game UI (EA/Respawn rights may apply). Copying follows the task's explicit instruction, **not** a claim of ownership — confirm redistribution rights before a public release.
