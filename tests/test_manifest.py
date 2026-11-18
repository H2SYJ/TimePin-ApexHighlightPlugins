"""Lightweight manifest localization and runtime-contract regression tests."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.config import load_manifest, parse_params

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_PARAMETERS = {
    "backend": {"type": "string", "enum": ["cpu", "dml", "cuda"], "default": "cpu"},
    "sample_fps": {"type": "number", "exclusiveMinimum": 0, "maximum": 120, "default": 2},
    "gap_seconds": {"type": "number", "minimum": 0, "default": 8},
    "min_segment_seconds": {"type": "number", "exclusiveMinimum": 0, "default": 3},
    "max_segment_seconds": {"type": "number", "exclusiveMinimum": 0, "default": 90},
    "extract_mode": {"type": "string", "enum": ["keyframes", "image2"], "default": "keyframes"},
    "enable_weapon_detection": {"type": "boolean", "default": True},
    "hud_info_enabled": {"type": "boolean", "default": False},
}


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.manifest = load_manifest(ROOT / "plugin.json")

    def load_modified(self, field, value):
        manifest = {**self.manifest, field: value}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plugin.json"
            path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
            return load_manifest(path)

    def test_all_display_text_has_both_translations(self):
        fields = {"description": self.manifest["description"]}
        for name, spec in self.manifest["parametersSchema"].items():
            for field in ("title", "description"):
                fields[f"{name}.{field}"] = spec[field]
        self.assertEqual(len(fields), 17)  # Plugin description + eight title/description pairs.
        for name, text in fields.items():
            with self.subTest(field=name):
                self.assertIsInstance(text, dict)
                self.assertEqual(set(text), {"en-US", "zh-CN"})
                for locale, value in text.items():
                    self.assertIsInstance(value, str)
                    self.assertTrue(value.strip(), f"{name}.{locale} is blank")
                self.assertNotEqual(text["en-US"], text["zh-CN"])
                self.assertTrue(any("一" <= char <= "鿿" for char in text["zh-CN"]))

    def test_existing_english_descriptions_are_preserved(self):
        self.assertEqual(
            self.manifest["description"]["en-US"],
            "Apex Legends HUD recognition plugin: detects shoot/hit/kill/downed combat "
            "events with optional HUD info enrichment (rank, hero, player names, combat "
            "stats); returns classified highlight Mark[].",
        )
        self.assertEqual(
            self.manifest["parametersSchema"]["enable_weapon_detection"]["description"]["en-US"],
            "Read active HUD weapon names and learn icon samples inside outputDir.",
        )

    def test_runtime_contract_is_unchanged(self):
        schema = self.manifest["parametersSchema"]
        runtime_schema = {
            name: {key: value for key, value in spec.items() if key not in {"title", "description"}}
            for name, spec in schema.items()
        }
        self.assertEqual(runtime_schema, EXPECTED_PARAMETERS)
        for field, expected in {
            "protocolVersion": "1.0.0",
            "name": "apex-highlight",
            "version": "0.1.1",
            "entry": "main.py",
            "runtime": "python",
            "capabilities": ["videoframes", "filewrite"],
        }.items():
            with self.subTest(field=field):
                self.assertEqual(self.manifest[field], expected)
        self.assertEqual(
            parse_params({}, schema),
            {name: spec["default"] for name, spec in EXPECTED_PARAMETERS.items()},
        )
        for backend in ("cpu", "dml", "cuda"):
            for mode in ("keyframes", "image2"):
                values = {
                    "backend": backend,
                    "sample_fps": 4,
                    "gap_seconds": 0,
                    "min_segment_seconds": 1,
                    "max_segment_seconds": 120,
                    "extract_mode": mode,
                    "enable_weapon_detection": False,
                    "hud_info_enabled": True,
                }
                with self.subTest(backend=backend, mode=mode):
                    self.assertEqual(parse_params(values, schema), values)

    def test_loader_accepts_legacy_and_localized_descriptions(self):
        for text in (
            "Legacy description",
            {"en-US": "English", "zh-CN": "中文"},
            {"fr-FR": "Description"},
        ):
            with self.subTest(description=text):
                self.assertEqual(self.load_modified("description", text)["description"], text)

    def test_loader_rejects_empty_or_malformed_descriptions(self):
        for text in (
            None, "", "  ", 42, [], {},
            {"en-US": ""}, {"zh-CN": "  "}, {"en-US": 42},
            {"en-US": {"text": "nested"}}, {"": "English"},
            {"en-US": "English", "zh-CN": None},
        ):
            with self.subTest(description=text), self.assertRaisesRegex(
                ValueError, "plugin.json requires a nonempty description"
            ):
                self.load_modified("description", text)

    def test_identity_fields_still_require_plain_strings(self):
        for field in ("protocolVersion", "name", "version", "author", "entry"):
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.load_modified(field, {"en-US": self.manifest[field]})


if __name__ == "__main__":
    unittest.main()
