"""Read-only access to bundled AHC recognition resources.

Bundled model/template files are checked into ``resources/`` and are the only
filesystem inputs these detectors may use. Frame image files remain the host's
responsibility; callers supply decoded BGR arrays or ``Frame.path`` objects.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

RESOURCE_DIR = Path(__file__).resolve().parents[1] / "resources"
_PROVENANCE: dict[str, str] | None = None


def require_file(relative: str | Path, *, verify_hash: bool = True) -> Path:
    """Return a regular bundled resource path, rejecting unsafe relatives."""
    if not isinstance(relative, (str, Path)):
        raise TypeError("relative must be a path string")
    path = Path(relative)
    if path.is_absolute() or not str(relative).strip():
        raise ValueError("relative must be a nonempty bundled-resource path")
    target = (RESOURCE_DIR / path).resolve()
    try:
        target.relative_to(RESOURCE_DIR.resolve())
    except ValueError as exc:
        raise ValueError("relative must stay inside resources/") from exc
    if not target.is_file():
        raise FileNotFoundError(f"Missing bundled recognition resource: {target}")
    if verify_hash:
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        expected = _checksums().get(path.as_posix())
        if expected is not None and digest != expected:
            raise RuntimeError(
                f"Bundled recognition resource failed SHA-256 verification: {target}"
            )
    return target


def load_json(relative: str | Path) -> dict[str, Any]:
    path = require_file(relative)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Cannot read bundled JSON resource {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise RuntimeError(f"Bundled JSON resource must contain an object: {path}")
    return data


def load_profile(relative: str | Path = "profile-ultrawide.json") -> dict[str, Any]:
    """Load a HUD profile without mutating the checked-in object."""
    profile = load_json(relative)
    return dict(profile)


def checksums() -> Mapping[str, str]:
    return dict(_checksums())


def _checksums() -> dict[str, str]:
    global _PROVENANCE
    if _PROVENANCE is not None:
        return _PROVENANCE
    inventory = RESOURCE_DIR / "provenance.json"
    checksums: dict[str, str] = {}
    try:
        data = json.loads(inventory.read_text(encoding="utf-8"))
        files = data.get("files") if isinstance(data, dict) else None
        if isinstance(files, list):
            for entry in files:
                if not isinstance(entry, dict):
                    continue
                path = entry.get("path")
                digest = entry.get("sha256")
                if (
                    isinstance(path, str)
                    and isinstance(digest, str)
                    and len(digest) == 64
                ):
                    checksums[path] = digest
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        # Inventory absence is allowed for newly bundled files; explicit resource
        # requirements still fail above when the actual file is missing.
        checksums = {}
    _PROVENANCE = checksums
    return checksums


def reset_cache() -> None:
    """Clear the provenance cache (used by tests that rewrite fixtures)."""
    global _PROVENANCE
    _PROVENANCE = None
