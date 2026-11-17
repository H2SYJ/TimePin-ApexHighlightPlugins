"""Manifest loading and strict parameter validation without heavy dependencies."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Mapping

from .errors import RpcError


PROTOCOL_VERSION = "1.0.0"
PLUGIN_NAME = "apex-highlight"


def load_manifest(path: Path) -> dict[str, object]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("plugin.json must contain an object")
    for field in (
        "protocolVersion",
        "name",
        "version",
        "description",
        "author",
        "entry",
    ):
        if not isinstance(manifest.get(field), str) or not manifest[field].strip():
            raise ValueError(f"plugin.json requires a nonempty {field}")
    if (
        manifest["protocolVersion"] != PROTOCOL_VERSION
        or manifest["name"] != PLUGIN_NAME
    ):
        raise ValueError("Unsupported plugin identity or protocolVersion")
    if manifest.get("runtime") != "python" or manifest["entry"] != "main.py":
        raise ValueError("Expected Python main.py entry point")
    if manifest.get("capabilities") != ["videoframes", "filewrite"]:
        raise ValueError("Expected TimePin videoframes/filewrite capabilities")
    schema = manifest.get("parametersSchema")
    if not isinstance(schema, dict) or set(schema) != {
        "backend",
        "sample_fps",
        "gap_seconds",
        "min_segment_seconds",
        "max_segment_seconds",
        "extract_mode",
    }:
        raise ValueError("Expected the six Apex parameter schemas")
    parse_params({}, schema)
    return manifest


def positive_number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise RpcError(-32602, f"{field} must be a finite positive number")
    try:
        number = float(value)
    except OverflowError as exc:
        raise RpcError(-32602, f"{field} is too large") from exc
    if not math.isfinite(number) or number <= 0:
        raise RpcError(-32602, f"{field} must be a finite positive number")
    return number


def parse_params(
    raw: object,
    schema: Mapping[str, object],
) -> dict[str, str | float]:
    if not isinstance(raw, dict):
        raise RpcError(-32602, "params must be an object")
    if set(raw) - set(schema):
        raise RpcError(-32602, "Unknown plugin parameter")
    parsed: dict[str, str | float] = {}
    for name, spec in schema.items():
        if not isinstance(spec, dict) or "default" not in spec:
            raise ValueError(f"Invalid parameter schema: {name}")
        value = raw.get(name, spec["default"])
        if spec.get("type") == "string":
            if not isinstance(value, str) or value not in spec.get("enum", []):
                raise RpcError(-32602, f"Invalid {name}")
            parsed[name] = value
            continue
        if name == "gap_seconds" and value == 0 and not isinstance(value, bool):
            number = 0.0
        else:
            number = positive_number(value, name)
        if "maximum" in spec and number > spec["maximum"]:
            raise RpcError(-32602, f"{name} exceeds {spec['maximum']}")
        parsed[name] = number
    if parsed["max_segment_seconds"] < parsed["min_segment_seconds"]:
        raise RpcError(-32602, "max_segment_seconds must be >= min_segment_seconds")
    return parsed
