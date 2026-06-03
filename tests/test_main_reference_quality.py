from __future__ import annotations

from app.main import _has_complete_reference_window, _is_clean_reference_jump


def _report_with_times(times: list[float]) -> dict[str, object]:
    return {
        "chart_data": {"time_s": times},
        "notes": {"analysis_blocked": False, "t0_review_required": False},
    }


def test_complete_reference_window_passes_with_dense_0_30_data():
    times = [i * 0.1 for i in range(0, 301)]
    report = _report_with_times(times)
    assert _has_complete_reference_window(report, start_s=0.0, end_s=25.0) is True


def test_complete_reference_window_fails_for_large_gap_before_25s():
    times = [0.0, 0.1, 0.2, 42.0, 42.1, 42.2, 42.3]
    report = _report_with_times(times)
    assert _has_complete_reference_window(report, start_s=0.0, end_s=25.0) is False


def test_complete_reference_window_fails_when_window_ends_too_early():
    times = [i * 0.1 for i in range(0, 221)]  # 0.0 .. 22.0
    report = _report_with_times(times)
    assert _has_complete_reference_window(report, start_s=0.0, end_s=25.0) is False


def test_clean_reference_jump_requires_complete_window():
    jump_row = {"quality_flags": "[]"}
    incomplete_report = _report_with_times([0.0, 0.1, 0.2, 42.0, 42.1, 42.2])
    complete_report = _report_with_times([i * 0.1 for i in range(0, 301)])

    assert _is_clean_reference_jump(jump_row, incomplete_report) is False
    assert _is_clean_reference_jump(jump_row, complete_report) is True

