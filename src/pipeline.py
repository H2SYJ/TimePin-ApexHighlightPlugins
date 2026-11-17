"""Cancellable metadata traversal; recognition is deliberately not implemented."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import threading

from .errors import Cancelled
from .events import Mark
from .frames import Frame


PROGRESS_INTERVAL = 32


def analyze(
    frames: Sequence[Frame],
    params: Mapping[str, object],
    *,
    cancel_event: threading.Event,
    on_progress: Callable[[int, int], None] | None = None,
) -> list[Mark]:
    """Visit metadata only, reporting counts every 32 frames and at completion.

    Parameters are reserved for future detectors. No image/SDK/model is loaded,
    no media is written, and no synthetic marks are emitted. Empty inputs report
    (0, 0); a final exact multiple of 32 is reported only once.
    """
    if cancel_event.is_set():
        raise Cancelled()
    total = len(frames)
    processed = 0
    for _frame in frames:
        if cancel_event.is_set():
            raise Cancelled()
        processed += 1
        if on_progress is not None and processed % PROGRESS_INTERVAL == 0:
            on_progress(processed, total)

    if cancel_event.is_set():
        raise Cancelled()
    if on_progress is not None and (
        processed == 0 or processed % PROGRESS_INTERVAL != 0
    ):
        on_progress(processed, total)
    # Also observe cancellation requested from the final progress callback.
    if cancel_event.is_set():
        raise Cancelled()
    return []
