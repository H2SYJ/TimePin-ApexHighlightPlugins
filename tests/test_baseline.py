"""Regression tests for the baseline comparator, guards, and report exit code."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest
from baseline import FFmpegFrameService, compare_marks, main, sha256, validate_source


def mark(start, end):
    return {"startTime": start, "endTime": end}


def segment(start, end):
    return {"start": start, "end": end}


def test_strict_tolerance_and_sorted_alignment():
    report = compare_marks([mark(8.1, 10.1), mark(0.49, 2.49)],
                           [segment(0, 2), segment(8, 10)])
    assert report["passed"] is True
    assert report["countsMatch"] is True
    assert report["boundaries"][0]["startError"] == pytest.approx(0.49)


@pytest.mark.parametrize("actual", [mark(0.5, 2), mark(0, 2.5), mark(0.51, 2.51)])
def test_half_second_is_not_accepted(actual):
    assert compare_marks([actual], [segment(0, 2)])["passed"] is False


@pytest.mark.parametrize("actual,expected,missing_key", [
    ([mark(0, 1), mark(2, 3)], [segment(0, 1)], "unmatchedMarks"),
    ([mark(0, 1)], [segment(0, 1), segment(2, 3)], "unmatchedSegments"),
    ([], [segment(0, 1)], "unmatchedSegments"),
    ([mark(0, 1)], [], "unmatchedMarks"),
])
def test_count_mismatches_cannot_pass(actual, expected, missing_key):
    report = compare_marks(actual, expected)
    assert report["passed"] is False
    assert report["countsMatch"] is False
    assert report[missing_key]


@pytest.mark.parametrize("start,end", [(float("nan"), 2), (0, float("inf")), (-1, 2), (2, 1)])
def test_invalid_boundaries_fail(start, end):
    with pytest.raises(ValueError, match="boundaries"):
        compare_marks([mark(start, end)], [])
    with pytest.raises(ValueError, match="boundaries"):
        compare_marks([], [segment(start, end)])


@pytest.mark.parametrize("tolerance", [0, -1, float("nan"), float("inf")])
def test_invalid_tolerance_fails(tolerance):
    with pytest.raises(ValueError, match="Tolerance"):
        compare_marks([], [], tolerance)


def test_point_mark_is_compared_as_zero_duration_not_dropped():
    assert compare_marks([mark(1, None)], [segment(1, 1)])["passed"] is True
    assert compare_marks([mark(1, None)], [segment(1, 2)])["passed"] is False


def test_review_and_protected_marks_are_not_silently_filtered():
    review = {**mark(1, 3), "metadata": {"classification": "review"}}
    downed = {**mark(4, 6), "metadata": {"type": "downed"}}
    result = compare_marks([review, downed], [segment(1, 3)])
    assert result["markCount"] == 2
    assert result["passed"] is False


@pytest.fixture
def oracle(tmp_path):
    video = tmp_path / "录像.mp4"
    video.write_bytes(b"source is never decoded in comparator tests")
    stat = video.stat()
    analysis = {
        "source": {"path": str(video), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns},
        "parameters": {"gap": 35.0, "pre": 10, "post": 15},
        "segments": [segment(1, 3)],
    }
    path = tmp_path / "analysis.json"
    path.write_text(json.dumps(analysis), encoding="utf-8")
    return video, analysis, path


def test_oracle_fingerprint_matches_unicode_video(oracle):
    video, analysis, _ = oracle
    validate_source(analysis, video)
    assert len(sha256(video)) == 64


@pytest.mark.parametrize("key,value", [("path", "other.mp4"), ("size", -1), ("mtime_ns", -1)])
def test_wrong_video_or_stale_oracle_is_rejected(oracle, key, value):
    video, analysis, _ = oracle
    analysis["source"][key] = value
    with pytest.raises(ValueError, match="fingerprint"):
        validate_source(analysis, video)


def test_oracle_requires_segments(oracle):
    video, analysis, _ = oracle
    analysis["segments"] = None
    with pytest.raises(ValueError, match="segments"):
        validate_source(analysis, video)


def test_changed_video_invalidates_cached_analysis(oracle):
    video, analysis, _ = oracle
    video.write_bytes(b"changed video")
    with pytest.raises(ValueError, match="fingerprint"):
        validate_source(analysis, video)


@pytest.mark.parametrize("marks,logs,segments,expected", [
    ([mark(1.1, 3.1)], [], [segment(1, 3)], 0),
    ([mark(1.5, 3)], [], [segment(1, 3)], 1),
    ([], [], [segment(1, 3)], 1),
    ([], [], [], 1),  # empty/empty is not positive accuracy evidence
    ([mark(1, 3)], ["HUD read degraded: missing model"], [segment(1, 3)], 1),
    ([mark(1, 3)], ["layout_status: unconfirmed"], [segment(1, 3)], 1),
])
def test_report_and_exit_code_cannot_claim_false_parity(oracle, tmp_path, marks, logs,
                                                      segments, expected):
    video, analysis, path = oracle
    analysis["segments"] = segments
    path.write_text(json.dumps(analysis), encoding="utf-8")
    output = tmp_path / "baseline"
    fake_run = {"result": {"marks": marks, "logs": logs}}
    with patch("baseline.run_plugin", return_value=fake_run) as run:
        code = main([str(video), "--ahc-analysis", str(path), "--out", str(output)])
    assert code == expected
    run.assert_called_once()
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert report["passed"] == (expected == 0)
    assert json.loads((output / "marks.json").read_text(encoding="utf-8")) == marks


def test_existing_output_is_never_overwritten(oracle, tmp_path):
    video, _, path = oracle
    with patch("baseline.run_plugin") as run, pytest.raises(FileExistsError):
        main([str(video), "--ahc-analysis", str(path), "--out", str(tmp_path)])
    run.assert_not_called()
    assert path.is_file() and video.is_file()


@pytest.mark.parametrize("change", [{"pre": 6}, {"gap": 10}, {"fps": 4}])
def test_mismatched_ahc_parameters_fail_before_plugin_run(oracle, tmp_path, change):
    video, analysis, path = oracle
    analysis["parameters"].update(change)
    path.write_text(json.dumps(analysis), encoding="utf-8")
    with patch("baseline.run_plugin") as run, pytest.raises(SystemExit):
        main([str(video), "--ahc-analysis", str(path), "--out", str(tmp_path / "out")])
    run.assert_not_called()


@pytest.mark.parametrize("other_video,escaped_output", [(True, False), (False, True)])
def test_test_host_enforces_authorized_paths(tmp_path, other_video, escaped_output):
    video = tmp_path / "source.mp4"
    video.write_bytes(b"unused")
    output = tmp_path / "authorized"
    output.mkdir()
    service = FFmpegFrameService("unused-ffmpeg", video, output)
    request = {"videoPath": str(tmp_path / "other.mp4" if other_video else video),
               "options": {"scratch_dir": str(tmp_path if escaped_output else output)}}
    with pytest.raises(PermissionError):
        service.extract(request)
    assert service.calls == []


def test_nonzero_window_pts_are_absolute(tmp_path):
    video = tmp_path / "source.mp4"
    video.write_bytes(b"unused")
    service = FFmpegFrameService("unused-ffmpeg", video, tmp_path)
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    for index in range(1, 3):
        (frames_dir / f"frame_{index:06d}.png").write_bytes(b"not decoded here")
    stderr = "[Parsed_showinfo] n: 0 pts: 0 pts_time:0\n[Parsed_showinfo] n: 1 pts: 1 pts_time:0.5"
    with patch.object(service, "_run", return_value=stderr):
        frames = service._image2(frames_dir, 8, 9, 2)
    assert [frame["ptsSec"] for frame in frames] == [8, 8.5]


def test_missing_showinfo_pts_are_not_invented(tmp_path):
    video = tmp_path / "source.mp4"
    service = FFmpegFrameService("unused-ffmpeg", video, tmp_path)
    (tmp_path / "frame_000001.png").write_bytes(b"metadata only")
    with patch.object(service, "_run", return_value=""), pytest.raises(RuntimeError, match="timestamps"):
        service._image2(tmp_path, 0, 1, 2)
