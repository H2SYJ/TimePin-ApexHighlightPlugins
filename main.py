#!/usr/bin/env python3
"""TimePin Apex Highlight JSON-RPC sidecar entry point."""

from __future__ import annotations

import sys
from pathlib import Path

from src.app import Application
from src.protocol import JsonRpcServer


def main() -> int:
    if not (3, 10) <= sys.version_info[:2] < (3, 13):
        print("Apex Highlight requires Python >=3.10,<3.13", file=sys.stderr)
        return 2
    # Windows console encodings must never alter JSON or Chinese paths.
    sys.stdin.reconfigure(encoding="utf-8", errors="strict")
    sys.stdout.reconfigure(encoding="utf-8", errors="strict", newline="\n")
    sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")
    # Native extension imports (cv2 → numpy → onnxruntime) can deadlock on
    # Windows when first executed on a worker thread while the reader thread
    # is blocked in a pipe readline(). Warm them here, once, on the main
    # thread before serve() spawns the run worker.
    try:
        import timepin_vision  # noqa: F401  (warms cv2/numpy/onnxruntime)
    except ImportError:
        # Recognition modules report actionable errors when they run;
        # don't make startup fail just because the SDK is missing.
        pass
    server = JsonRpcServer(sys.stdin, sys.stdout)
    app = Application(server, Path(__file__).resolve().parent / "plugin.json")
    server.serve(app.dispatch, prepare_run=app.prepare_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
