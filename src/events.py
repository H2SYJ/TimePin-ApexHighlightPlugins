"""Host-compatible marks and a pure conversion boundary for future detectors."""

from __future__ import annotations

import math
from typing import TypedDict
from uuid import uuid4


class Mark(TypedDict):
    id: str
    startTime: float
    endTime: float | None
    label: str
    color: str | None
    confidence: float | None
    sourcePlugin: str | None
    thumbnailPath: str | None
    metadata: dict[str, str]


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    try:
        number = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    return number


def event_to_mark(
    start_time: float,
    label: str,
    *,
    end_time: float | None = None,
    confidence: float | None = None,
    metadata: dict[str, str] | None = None,
) -> Mark:
    """Convert a real detected event, raising ValueError for invalid values.

    This helper does not detect events and is intentionally not called by the
    skeleton pipeline. Metadata is copied so later caller mutations are isolated.
    """
    start = _finite_number(start_time, "start_time")
    if start < 0:
        raise ValueError("start_time must be nonnegative")
    if not isinstance(label, str):
        raise ValueError("label must be a string")
    end = None if end_time is None else _finite_number(end_time, "end_time")
    if end is not None and end < start:
        raise ValueError("end_time must be greater than or equal to start_time")
    score = None if confidence is None else _finite_number(confidence, "confidence")
    if score is not None and not 0 <= score <= 1:
        raise ValueError("confidence must be between 0 and 1")
    if metadata is not None and (
        not isinstance(metadata, dict)
        or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in metadata.items()
        )
    ):
        raise ValueError("metadata must be a dictionary of strings to strings")

    return {
        "id": str(uuid4()),
        "startTime": start,
        "endTime": end,
        "label": label,
        "color": None,
        "confidence": score,
        "sourcePlugin": "apex-highlight",
        "thumbnailPath": None,
        "metadata": {} if metadata is None else dict(metadata),
    }
