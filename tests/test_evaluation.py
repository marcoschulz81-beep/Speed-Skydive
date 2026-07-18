from __future__ import annotations

import json

from app.analysis.evaluation import effective_eval_window_end_s, is_reference_eligible
from app.main import _build_rule_score_issue_lines, _safe_json_dumps


def test_effective_eval_window_uses_earliest_shared_boundary() -> None:
    end_s = effective_eval_window_end_s(
        metrics={"performance_window_end_s": 24.7},
        notes={"decel_start_s": 27.0, "curve_window_end_s": 31.0},
        chart_data={"time_s": [0.0, 20.0, 40.0]},
    )

    assert end_s == 24.7


def test_reference_eligibility_requires_valid_clean_rule_score() -> None:
    metrics = {"rule_based_3s_score": 430.0, "rule_score_status": "valid"}
    notes = {"analysis_blocked": False, "t0_review_required": False}

    assert is_reference_eligible(metrics=metrics, notes=notes, quality_flags=[]) is True
    assert is_reference_eligible(metrics=metrics, notes=notes, quality_flags=["TIME_GAPS"]) is False
    assert is_reference_eligible(
        metrics={**metrics, "rule_score_status": "estimated"},
        notes=notes,
        quality_flags=[],
    ) is False


def test_inline_json_serialization_blocks_script_end_tag_and_round_trips() -> None:
    payload = {"file_name": "</script><script>alert('x')</script>", "separator": "\u2028"}

    serialized = _safe_json_dumps(payload)

    assert "</script>" not in serialized
    assert "<script>" not in serialized
    assert "\\u003c/script\\u003e" in serialized
    assert json.loads(serialized) == payload


def test_invalid_rule_score_always_produces_visible_exclusion_reason() -> None:
    lines = _build_rule_score_issue_lines(
        {
            "rule_score_status": "invalid",
            "rule_score_reasons": ["NO_RULE_WINDOW", "PERFORMANCE_WINDOW_INCOMPLETE"],
        }
    )

    assert any("keinen gültigen Regel-Score" in line for line in lines)
    assert any("Breakoff-Höhe" in line for line in lines)
