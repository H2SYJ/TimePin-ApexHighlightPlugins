"""Pure downed lifecycle grouping ported from AHC's apex_clipper.py.

The two-vote confirmation, rescue/re-downed grouping and terminal-tail rules are
retained verbatim in behavior. Detection is supplied by the host's BGR frame
recognizers; this module never opens media, templates, models or the SDK.
"""

from __future__ import annotations

from .events import _ahc_rows


def downed_intervals(
    rows: list[dict],
    duration: float,
    gap: float = 35,
    post: float = 15,
    terminal_tail: float = 2,
) -> list[dict]:
    """Cover downed/rescue/re-downed until terminal view or recovery tail.

    A terminal view requires *consecutive* inactive rows within 1.1 seconds,
    after the last confirmed downed signal and no later than ``post`` seconds.
    Both t/inactive and AHC time/inactive_view row spellings are accepted.
    """
    rows = _ahc_rows(rows)
    confirmed = []
    for i, row in enumerate(rows):
        if not row.get("downed") or row.get("inactive_view"):
            continue
        neighbours = rows[max(0, i - 2) : i + 3]
        votes = sum(
            r.get("downed", False)
            and not r.get("inactive_view")
            and abs(r["time"] - row["time"]) <= 1.1
            for r in neighbours
        )
        if votes >= 2:
            confirmed.append(row["time"])
    if not confirmed:
        return []
    groups = []
    first = last = confirmed[0]
    for t in confirmed[1:]:
        if t - last <= gap:
            last = t
        else:
            groups.append((first, last))
            first = last = t
    groups.append((first, last))
    intervals = []
    for first, last in groups:
        terminal = None
        for i, row in enumerate(rows[:-1]):
            if not last < row["time"] <= last + post:
                continue
            following = rows[i + 1]
            if (
                row.get("inactive_view")
                and following.get("inactive_view")
                and following["time"] - row["time"] <= 1.1
            ):
                terminal = row["time"]
                break
        end = min(
            duration, terminal + terminal_tail if terminal is not None else last + post
        )
        intervals.append(
            {
                "start": first,
                "end": end,
                "last_downed": last,
                "terminal_time": terminal,
                "reason": "downed_until_terminal"
                if terminal is not None
                else "downed_recovery_tail",
            }
        )
    return intervals
