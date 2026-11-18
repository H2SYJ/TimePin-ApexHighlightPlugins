#!/usr/bin/env python3
"""Build script to create TimePin plugin package (.tppkg / .zip).

Usage:
    python build_package.py [--output-dir DIR] [--include-offline-deps] [--format {tppkg,zip}]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED


REPO_ROOT = Path(__file__).resolve().parent

# Files/dirs to include in the plugin package
INCLUDE_PATTERNS = [
    "plugin.json",
    "main.py",
    "src/**/*",
    "resources/**/*",
    "requirements.txt",
    "README.md",
]

# Files/dirs to exclude from the plugin package
EXCLUDE_PATTERNS = [
    "tests/**/*",
    "__pycache__/**/*",
    "*.py[cod]",
    ".git/**/*",
    ".github/**/*",
    ".venv/**/*",
    ".ruff_cache/**/*",
    ".coverage",
    "htmlcov/**/*",
    "build_package.py",
    "pytest.ini",
    "requirements-test.txt",
    "*.tppkg",
    "*.zip",
    "*.mp4",
    "*.mkv",
    "*.avi",
    "*.mov",
    "tests/fixtures/**/*",
]


def should_include(path: Path, include_patterns: list[str], exclude_patterns: list[str]) -> bool:
    """Check if a path should be included in the package."""
    rel_path = path.relative_to(REPO_ROOT)
    rel_str = str(rel_path).replace("\\", "/")

    # Check exclude patterns first
    for pattern in exclude_patterns:
        if _match_pattern(rel_str, pattern):
            return False

    # Check include patterns
    for pattern in include_patterns:
        if _match_pattern(rel_str, pattern):
            return True

    return False


def _match_pattern(path: str, pattern: str) -> bool:
    """Simple glob pattern matching."""
    import fnmatch
    return fnmatch.fnmatch(path, pattern) or fnmatch.fnmatch(path, pattern.replace("**/*", "**"))


def collect_files() -> list[Path]:
    """Collect all files to include in the package."""
    files = []
    for pattern in INCLUDE_PATTERNS:
        for path in REPO_ROOT.glob(pattern):
            if path.is_file() and should_include(path, INCLUDE_PATTERNS, EXCLUDE_PATTERNS):
                files.append(path)
    return sorted(set(files))


def build_offline_deps(output_dir: Path) -> Path:
    """Build offline dependencies (wheel files) into a directory.

    Returns the path to the offline deps directory.
    """
    offline_dir = output_dir / "offline_deps"
    offline_dir.mkdir(parents=True, exist_ok=True)

    print("Building offline dependencies...")

    # Download wheels for dependencies in requirements.txt
    # Use pip download with --no-deps to avoid pulling transitive deps we don't control
    # Actually, we want all transitive deps for offline install, so use normal download
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "download",
            "-r", str(REPO_ROOT / "requirements.txt"),
            "-d", str(offline_dir),
            "--no-build-isolation",
        ], check=True, capture_output=True, text=True)
        print(f"Offline dependencies saved to {offline_dir}")
    except subprocess.CalledProcessError as e:
        print(f"Warning: Failed to download some dependencies: {e.stderr}")
        # Continue anyway - the online install will work

    return offline_dir


def create_package(
    output_path: Path,
    format: str = "tppkg",
    include_offline_deps: bool = False,
) -> Path:
    """Create the plugin package."""

    files = collect_files()
    if not files:
        raise RuntimeError("No files collected for packaging")

    print(f"Packaging {len(files)} files...")

    # Create a temporary directory for staging
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        package_root = tmpdir_path / "apex-highlight"
        package_root.mkdir()

        # Copy files preserving structure
        for src_file in files:
            rel_path = src_file.relative_to(REPO_ROOT)
            dst_file = package_root / rel_path
            dst_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_file, dst_file)

        # Add offline dependencies if requested
        if include_offline_deps:
            offline_dir = build_offline_deps(package_root)
            # Move offline deps into package
            for wheel in offline_dir.glob("*.whl"):
                dst = package_root / "offline_deps" / wheel.name
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(wheel), str(dst))
            shutil.rmtree(offline_dir, ignore_errors=True)

        # Create package.json metadata inside the package
        plugin_json = json.loads((REPO_ROOT / "plugin.json").read_text(encoding="utf-8"))
        package_meta = {
            "name": plugin_json["name"],
            "version": plugin_json["version"],
            "protocolVersion": plugin_json["protocolVersion"],
            "runtime": plugin_json["runtime"],
            "entry": plugin_json["entry"],
            "python": plugin_json["python"],
            "hasOfflineDeps": include_offline_deps,
            "files": [str(f.relative_to(package_root)).replace("\\", "/") for f in package_root.rglob("*") if f.is_file()],
        }
        (package_root / "package.json").write_text(json.dumps(package_meta, indent=2), encoding="utf-8")

        # Create the archive
        if format == "tppkg":
            # .tppkg is just a .zip with a different extension
            # Use string concatenation instead of with_suffix to handle versions like 0.1.1
            archive_path = Path(str(output_path) + ".tppkg")
        else:
            archive_path = Path(str(output_path) + ".zip")

        # Remove existing
        if archive_path.exists():
            archive_path.unlink()

        with ZipFile(archive_path, "w", ZIP_DEFLATED, compresslevel=6) as zf:
            for file_path in package_root.rglob("*"):
                if file_path.is_file():
                    arcname = file_path.relative_to(tmpdir_path)
                    zf.write(file_path, arcname)

        # Verify the package
        size_mb = archive_path.stat().st_size / (1024 * 1024)
        print(f"Created {archive_path.name} ({size_mb:.2f} MB)")

        # Show contents
        with ZipFile(archive_path, "r") as zf:
            print("Package contents:")
            for info in zf.infolist():
                print(f"  {info.file_size:>10}  {info.filename}")

        return archive_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Build TimePin Apex Highlight plugin package")
    parser.add_argument("--output-dir", "-o", type=Path, default=REPO_ROOT / "dist",
                        help="Output directory for the package")
    parser.add_argument("--include-offline-deps", action="store_true",
                        help="Include offline dependency wheels in the package")
    parser.add_argument("--format", choices=["tppkg", "zip"], default="tppkg",
                        help="Package format (default: tppkg)")
    parser.add_argument("--version", help="Override version (defaults to plugin.json)")

    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Determine version
    plugin_json = json.loads((REPO_ROOT / "plugin.json").read_text(encoding="utf-8"))
    version = args.version or plugin_json["version"]

    output_name = f"apex-highlight-{version}"
    output_path = args.output_dir / output_name

    try:
        package_path = create_package(
            output_path=output_path,
            format=args.format,
            include_offline_deps=args.include_offline_deps,
        )
        print(f"\n[OK] Package created: {package_path}")
        return 0
    except Exception as e:
        print(f"\n[FAIL] Build failed: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())