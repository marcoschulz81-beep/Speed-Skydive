from __future__ import annotations

from app.analysis.review import (
    _action_phase_rank,
    _build_angle_status_from_phases,
    _build_priority_action_items,
    _build_priority_actions,
    _forward_eval_end_s,
    _goal_metrics_for_action,
    _technical_phase_status_map,
    build_jump_review,
)


def test_build_jump_review_contains_all_sections():
    report = {
        "jump": {"file_name": "a.csv"},
        "metrics": {
            "best_3s_start_s": 12.0,
            "best_3s_end_s": 15.0,
            "best_3s_vVert_kmh": 430.0,
            "best_3s_vHor_kmh": 34.0,
        },
        "notes": {
            "curve_window_start_s": 0.0,
            "curve_window_end_s": 38.0,
        },
        "scorecard": {
            "exit": "sauber",
            "phase_10_20": "optimal",
            "hot_zone": "stabil",
            "kipp_risiko": "mittel",
        },
        "quality_flags": [],
        "fixpoints": [
            {"t_rel_s": 20.0, "vHor_kmh": 34.0},
            {"t_rel_s": 28.0, "vHor_kmh": 30.0},
        ],
    }
    best_compare = {
        "summary": [
            {"label": "3s Max (Training)", "delta": 4.2},
        ],
        "fixpoint_rows": [
            {"t_rel_s": 20.0, "delta_vHor_kmh": 7.5},
        ],
    }

    review = build_jump_review(report, best_compare=best_compare)
    assert set(review.keys()) == {
        "happened",
        "good",
        "not_good",
        "improve",
        "coaching_goals",
        "primary_diagnosis",
        "technical_assessment",
    }
    assert review["primary_diagnosis"]["available"] is False
    assert len(review["happened"]) >= 1
    assert len(review["good"]) >= 1
    assert len(review["not_good"]) >= 1
    assert len(review["improve"]) >= 1
    assert len(review["coaching_goals"]) >= 1
    assert {"id", "priority", "phase", "text", "display_text", "target_metrics"}.issubset(
        review["coaching_goals"][0]
    )
    assert isinstance(review["coaching_goals"][0]["target_metrics"], list)


def test_angle_status_prefers_normalized_technical_phases_over_legacy_scorecard():
    phase_statuses = _technical_phase_status_map(
        {
            "phases": [
                {"name": "Hauptbeschleunigung", "angle_status": "in_band"},
                {"name": "Hot-Zone Aufbau", "angle_status": "too_flat"},
            ]
        }
    )

    assert phase_statuses == {
        "Hauptbeschleunigung": "im Zielbereich",
        "Hot-Zone Aufbau": "zu flach",
    }
    assert (
        _build_angle_status_from_phases(
            scorecard={"phase_10_20": "optimal"},
            phase_statuses=phase_statuses,
        )
        == "zu flach"
    )


def test_build_jump_review_prioritizes_start_and_build_when_low():
    report = {
        "jump": {"file_name": "b.csv"},
        "metrics": {
            "best_3s_start_s": 16.0,
            "best_3s_end_s": 19.0,
            "best_3s_vVert_kmh": 398.0,
            "best_3s_vHor_kmh": 29.0,
        },
        "notes": {
            "curve_window_start_s": 0.0,
            "curve_window_end_s": 36.0,
        },
        "scorecard": {
            "exit": "sauber",
            "phase_10_20": "optimal",
            "hot_zone": "stabil",
            "kipp_risiko": "niedrig",
        },
        "quality_flags": [],
        "fixpoints": [
            {"t_rel_s": 10.0, "vVert_kmh": 205.0, "vHor_kmh": 72.0, "angle_deg": 73.0},
            {"t_rel_s": 20.0, "vVert_kmh": 275.0, "vHor_kmh": 40.0, "angle_deg": 82.0},
            {"t_rel_s": 24.0, "vVert_kmh": 300.0, "vHor_kmh": 35.0, "angle_deg": 83.0},
            {"t_rel_s": 28.0, "vVert_kmh": 320.0, "vHor_kmh": 31.0, "angle_deg": 84.0},
        ],
        "chart_data": {
            "time_s": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "accVert_mps2": [0.9, 1.1, 1.4, 1.2, 1.0, 0.8],
        },
    }

    review = build_jump_review(report, best_compare=None)

    assert any("Priorität" in item for item in review["improve"])
    assert any("Startphase" in item or "Zwischen +10s und +20s mehr Druck aufbauen" in item for item in review["improve"])


def test_build_jump_review_limits_low_build_target_to_next_step():
    report = {
        "jump": {"file_name": "incremental-target.csv"},
        "metrics": {
            "best_3s_start_s": 21.0,
            "best_3s_end_s": 24.0,
            "best_3s_vVert_kmh": 360.0,
            "best_3s_vHor_kmh": 32.0,
        },
        "notes": {
            "curve_window_start_s": 0.0,
            "curve_window_end_s": 30.0,
        },
        "scorecard": {
            "exit": "sauber",
            "phase_10_20": "optimal",
            "hot_zone": "stabil",
            "kipp_risiko": "niedrig",
        },
        "quality_flags": [],
        "fixpoints": [
            {"t_rel_s": 10.0, "vVert_kmh": 200.0, "vHor_kmh": 70.0, "angle_deg": 73.0},
            {"t_rel_s": 15.0, "vVert_kmh": 240.0, "vHor_kmh": 55.0, "angle_deg": 78.0},
            {"t_rel_s": 20.0, "vVert_kmh": 274.0, "vHor_kmh": 40.0, "angle_deg": 82.0},
            {"t_rel_s": 24.0, "vVert_kmh": 350.0, "vHor_kmh": 35.0, "angle_deg": 84.0},
        ],
        "chart_data": {
            "time_s": [float(i) for i in range(0, 31)],
            "vVert_kmh": [200.0 + 5.0 * i for i in range(0, 31)],
            "vHor_kmh": [70.0 - 1.2 * i for i in range(0, 31)],
            "angle_deg": [72.0 + 0.4 * i for i in range(0, 31)],
            "accVert_mps2": [1.0 for _ in range(0, 31)],
        },
    }

    review = build_jump_review(
        report,
        best_compare=None,
        jumper_stability_reference={
            "available": True,
            "thresholds": {
                "gain_10_20_target_low": 170.0,
                "gain_10_20_target_high": 190.0,
            },
        },
    )

    build_actions = [
        item
        for item in review["improve"]
        if "Zwischen +10s und +20s mehr Druck aufbauen" in item
    ]
    assert build_actions
    assert "+99 km/h" in build_actions[0]
    assert "+180 km/h" not in build_actions[0]


def test_build_jump_review_suppresses_global_top5_tips_below_elite_level():
    time_s = [float(i) for i in range(0, 31)]
    report = {
        "jump": {"jump_id": "current", "file_name": "current.csv"},
        "metrics": {
            "best_3s_start_s": 21.0,
            "best_3s_end_s": 24.0,
            "best_3s_vVert_kmh": 401.0,
            "best_3s_vHor_kmh": 26.0,
        },
        "notes": {
            "curve_window_start_s": 0.0,
            "curve_window_end_s": 30.0,
        },
        "scorecard": {
            "exit": "sauber",
            "phase_10_20": "optimal",
            "hot_zone": "stabil",
            "kipp_risiko": "mittel",
        },
        "quality_flags": [],
        "fixpoints": [
            {"t_rel_s": 10.0, "vVert_kmh": 255.0, "vHor_kmh": 62.0, "angle_deg": 76.0},
            {"t_rel_s": 15.0, "vVert_kmh": 300.0, "vHor_kmh": 50.0, "angle_deg": 80.0},
            {"t_rel_s": 20.0, "vVert_kmh": 372.0, "vHor_kmh": 36.0, "angle_deg": 84.0},
            {"t_rel_s": 24.0, "vVert_kmh": 400.0, "vHor_kmh": 28.0, "angle_deg": 85.0},
        ],
        "chart_data": {
            "time_s": time_s,
            "vVert_kmh": [220.0 + i * 6.0 for i in range(0, 31)],
            "vHor_kmh": [78.0 - i * 1.4 for i in range(0, 31)],
            "angle_deg": [72.0 + i * 0.45 for i in range(0, 31)],
            "accVert_mps2": [2.0 for _ in time_s],
        },
    }
    top_reference_compare = {
        "reference": {"best_3s_vVert_kmh": 520.0, "file_name": "elite.csv"},
        "comparison": {"jump_id": "current"},
        "summary": [{"label": "3s Max (Training)", "delta": -119.0}],
        "fixpoint_rows": [
            {"t_rel_s": 10.0, "delta_vVert_kmh": -95.0},
            {"t_rel_s": 15.0, "delta_vVert_kmh": -150.0},
        ],
        "charts": {
            "left": {"time_s": time_s, "vVert_kmh": [420.0 for _ in time_s]},
            "right": {"time_s": time_s, "vVert_kmh": [360.0 for _ in time_s]},
        },
    }
    stability_reference = {
        "available": True,
        "performance_profile": {
            "available": True,
            "performance_band": "aufbau",
            "summary": "Aufbauprofil 401.0 km/h.",
        },
    }

    review = build_jump_review(
        report,
        top_reference_compares=[top_reference_compare],
        jumper_stability_reference=stability_reference,
    )
    combined = " ".join(review["happened"] + review["not_good"] + review["improve"])

    assert "Top-5" not in combined


def test_build_jump_review_detects_unsteady_curve_corrections():
    # Oscillating angle/vHor pattern to simulate repeated corrections.
    time_s = [float(i) for i in range(0, 31)]
    angle = [72 + (4 if i % 2 == 0 else -4) + i * 0.45 for i in range(0, 31)]
    vhor = [52 + (6 if i % 2 == 0 else -6) - i * 0.8 for i in range(0, 31)]
    vvert = [220 + i * 6 for i in range(0, 31)]

    report = {
        "jump": {"file_name": "c.csv"},
        "metrics": {
            "best_3s_start_s": 20.0,
            "best_3s_end_s": 23.0,
            "best_3s_vVert_kmh": 410.0,
            "best_3s_vHor_kmh": 27.0,
        },
        "notes": {
            "curve_window_start_s": 0.0,
            "curve_window_end_s": 30.0,
            "canopy_open_s": 32.0,
        },
        "scorecard": {
            "exit": "sauber",
            "phase_10_20": "optimal",
            "hot_zone": "stabil",
            "kipp_risiko": "mittel",
        },
        "quality_flags": [],
        "fixpoints": [
            {"t_rel_s": 10.0, "vVert_kmh": 280.0, "vHor_kmh": 42.0, "angle_deg": 80.0},
            {"t_rel_s": 20.0, "vVert_kmh": 390.0, "vHor_kmh": 30.0, "angle_deg": 86.0},
            {"t_rel_s": 24.0, "vVert_kmh": 400.0, "vHor_kmh": 24.0, "angle_deg": 87.5},
            {"t_rel_s": 28.0, "vVert_kmh": 405.0, "vHor_kmh": 22.0, "angle_deg": 88.0},
        ],
        "chart_data": {
            "time_s": time_s,
            "vHor_kmh": vhor,
            "angle_deg": angle,
            "vVert_kmh": vvert,
            "accVert_mps2": [2.5 for _ in time_s],
        },
    }

    review = build_jump_review(report, best_compare=None)

    assert any("Kurvenverlauf" in item for item in review["not_good"])


def test_priority_actions_follow_flight_sequence():
    actions = {
        "peak_stability": {
            "score": 10,
            "text": "In der Peak-Phase Druck ruhiger halten und kleine, frühe Korrekturen machen.",
        },
        "build_speed_low": {
            "score": 9,
            "text": "Zwischen +10s und +20s mehr Druck aufbauen. Ziel: in diesem Abschnitt mindestens +90 km/h Zuwachs.",
        },
        "exit_carryover_low": {
            "score": 7,
            "text": "Nach dem Exit den Druck länger tragen: ab +2s stabil weiter beschleunigen, statt früh nachzulassen.",
        },
        "kipp_risk_high": {
            "score": 8,
            "text": "Bei Instabilität Körperspannung früher stabilisieren (Schulter und Hüfte).",
        },
    }

    ordered = _build_priority_actions(actions, max_items=5)
    plain = [item.split(": ", 1)[1] for item in ordered]

    exit_idx = plain.index(actions["exit_carryover_low"]["text"])
    build_idx = plain.index(actions["build_speed_low"]["text"])
    peak_idx = plain.index(actions["peak_stability"]["text"])
    stability_idx = plain.index(actions["kipp_risk_high"]["text"])

    assert exit_idx < build_idx < peak_idx < stability_idx


def test_priority_action_items_add_structured_goal_metrics():
    actions = {
        "phase_10_15_too_steep_not_hold": {
            "score": 9,
            "text": "Im Aufbau nicht zu schnell maximal steil werden und die Linie beruhigen.",
        }
    }

    items = _build_priority_action_items(actions, max_items=5)

    assert len(items) == 1
    assert items[0]["id"] == "phase_10_15_too_steep_not_hold"
    assert items[0]["phase"] == "Hauptbeschleunigung"
    assert items[0]["display_text"].startswith("Priorit")
    metrics = {item["metric"]: item for item in items[0]["target_metrics"]}
    assert metrics["angle_10"]["direction"] == "decrease"
    assert metrics["angle_15"]["direction"] == "decrease"
    assert metrics["angle_turns_20_25"]["direction"] == "decrease"


def test_priority_action_items_consolidate_duplicate_hot_zone_goals():
    actions = {
        "efficiency_20_25_low": {
            "score": 8,
            "text": "Hot-Zone: früher klein korrigieren und die Linie ruhiger halten.",
        },
        "vhor_price_20_25": {
            "score": 7,
            "text": "Ab +20s sauberer arbeiten, damit vHor langsamer sinkt.",
        },
        "peak_stability": {
            "score": 8,
            "text": "In der Peak-Phase Druck ruhiger halten und kleine, frühe Korrekturen machen.",
        },
        "segment_20_25_vhor_drop": {
            "score": 8,
            "text": "In der Hot-Zone die Linie ruhiger halten, damit vHor nicht einbricht.",
        },
        "kipp_risk_high": {
            "score": 8,
            "text": "Bei Instabilität Körperspannung früher stabilisieren.",
        },
    }

    items = _build_priority_action_items(actions, max_items=5)
    ids = [item["id"] for item in items]
    hot_items = [
        item
        for item in items
        if any(target.get("metric") == "vhor_min_20_25" for target in item["target_metrics"])
    ]

    assert len(hot_items) == 1
    assert hot_items[0]["id"] == "efficiency_20_25_low"
    assert "vhor_price_20_25" not in ids
    assert "peak_stability" not in ids
    assert "segment_20_25_vhor_drop" not in ids
    assert "kipp_risk_high" in ids


def test_goal_metric_direction_matches_flatter_angle_goal():
    metrics = _goal_metrics_for_action(
        key="angle_20_personal_steep",
        text="Im Aufbau bis +20s etwas flacher bleiben und den Winkel stabil halten.",
        phase_rank=1,
    )
    angle20 = [item for item in metrics if item["metric"] == "angle_20"]

    assert angle20
    assert all(item["direction"] == "decrease" for item in angle20)


def test_goal_metric_does_not_raise_angle_when_build_goal_has_guardrail():
    metrics = _goal_metrics_for_action(
        key="phase_10_15_below_corridor",
        text=(
            "Im Segment +10 bis +15s früher Druck aufbauen, aber den Winkel nicht erzwingen "
            "und die Linie ruhiger halten. Erst Stabilität sichern: nicht sofort steiler werden."
        ),
        phase_rank=1,
    )

    directions = {(item["metric"], item["direction"]) for item in metrics}
    assert ("gain_10_20", "increase") in directions
    assert ("angle_20", "increase") not in directions


def test_goal_metric_safe_limit_prefers_stabilizing_early_angle():
    metrics = _goal_metrics_for_action(
        key="exit_angle_10_safe_limit",
        text=(
            "Bis +10s zuerst stabil im Bereich 72.3 bis 76.5 Grad bleiben. "
            "Erst wenn die Linie bis +15s ruhig bleibt, den Winkel in kleinen Schritten weiter aufbauen."
        ),
        phase_rank=0,
    )

    directions = {(item["metric"], item["direction"]) for item in metrics}
    assert ("angle_10", "decrease") in directions
    assert ("angle_20", "increase") not in directions


def test_action_phase_rank_prefers_hot_zone_over_generic_build_word():
    rank = _action_phase_rank(
        key="efficiency_20_25_low",
        text="Hot-Zone (+17.5s bis +22.5s): in der Aufbau-Mitte mehr Ruhe halten.",
    )

    assert rank == 2


def test_action_phase_rank_keeps_early_corridor_goals_in_build_phase():
    rank = _action_phase_rank(
        key="phase_10_15_below_corridor",
        text="Im Segment +10 bis +15s frueher Druck aufbauen und die Linie ruhiger halten.",
    )

    assert rank == 1


def test_build_jump_review_detects_negative_forward_drift():
    time_s = [float(i) for i in range(0, 31)]
    vvert = [220.0 + i * 5.5 for i in range(0, 31)]
    vhor = [95.0 - i * 2.0 for i in range(0, 31)]
    angle = [70.0 + i * 0.6 for i in range(0, 31)]
    acc = [2.5 for _ in time_s]
    # Vorwärts-Strecke: erst vorwärts, dann klare Rückdrift.
    forward_m = [i * 5.0 if i <= 20 else 100.0 - (i - 20) * 3.0 for i in range(0, 31)]
    running_max = []
    cur = float("-inf")
    for value in forward_m:
        cur = max(cur, float(value))
        running_max.append(cur)
    backtrack_m = [running_max[i] - forward_m[i] for i in range(len(forward_m))]

    report = {
        "jump": {"file_name": "drift.csv"},
        "metrics": {
            "best_3s_start_s": 20.0,
            "best_3s_end_s": 23.0,
            "best_3s_vVert_kmh": 410.0,
            "best_3s_vHor_kmh": 24.0,
        },
        "notes": {
            "curve_window_start_s": 0.0,
            "curve_window_end_s": 30.0,
            "canopy_open_s": 35.0,
        },
        "scorecard": {
            "exit": "sauber",
            "phase_10_20": "optimal",
            "hot_zone": "kritisch",
            "kipp_risiko": "hoch",
        },
        "quality_flags": [],
        "fixpoints": [
            {"t_rel_s": 10.0, "vVert_kmh": 280.0, "vHor_kmh": 74.0, "angle_deg": 76.0},
            {"t_rel_s": 20.0, "vVert_kmh": 390.0, "vHor_kmh": 38.0, "angle_deg": 84.5},
            {"t_rel_s": 24.0, "vVert_kmh": 402.0, "vHor_kmh": 24.0, "angle_deg": 86.5},
            {"t_rel_s": 28.0, "vVert_kmh": 396.0, "vHor_kmh": 20.0, "angle_deg": 87.0},
        ],
        "chart_data": {
            "time_s": time_s,
            "vVert_kmh": vvert,
            "vHor_kmh": vhor,
            "angle_deg": angle,
            "hAGL_m": [3500.0 - i * 60.0 for i in range(0, 31)],
            "accVert_mps2": acc,
            "forward_m": forward_m,
            "backtrack_m": backtrack_m,
        },
    }

    review = build_jump_review(report, best_compare=None)
    assert any("Rückdrift" in item for item in review["happened"])


def test_forward_eval_end_prefers_decel_start_over_generic_window():
    assert _forward_eval_end_s(decel_start_s=30.5, fallback_end_s=25.0) == 30.5
    assert _forward_eval_end_s(decel_start_s=6.5, fallback_end_s=25.0) == 25.0


def test_build_jump_review_personalizes_angle_tip_from_stability_reference():
    report = {
        "jump": {"file_name": "personalized.csv"},
        "metrics": {
            "best_3s_start_s": 19.0,
            "best_3s_end_s": 22.0,
            "best_3s_vVert_kmh": 401.0,
            "best_3s_vHor_kmh": 26.0,
        },
        "notes": {
            "curve_window_start_s": 0.0,
            "curve_window_end_s": 31.0,
        },
        "scorecard": {
            "exit": "sauber",
            "phase_10_20": "zu flach",
            "hot_zone": "stabil",
            "kipp_risiko": "mittel",
        },
        "quality_flags": [],
        "fixpoints": [
            {"t_rel_s": 10.0, "vVert_kmh": 248.0, "vHor_kmh": 82.0, "angle_deg": 70.5},
            {"t_rel_s": 20.0, "vVert_kmh": 360.0, "vHor_kmh": 43.0, "angle_deg": 79.4},
            {"t_rel_s": 24.0, "vVert_kmh": 384.0, "vHor_kmh": 31.0, "angle_deg": 84.0},
        ],
    }
    stability_reference = {
        "available": True,
        "thresholds": {
            "angle_20_target_low": 82.3,
            "angle_20_target_high": 84.1,
            "vhor_min_20_25_floor": 30.0,
        },
    }

    review = build_jump_review(
        report,
        best_compare=None,
        jumper_stability_reference=stability_reference,
    )

    joined = " ".join(review["improve"])
    assert "80.3" in joined
    assert "84.1" in joined


def test_build_jump_review_adds_phase_corridor_hint_when_10_15_is_below_personal_band():
    report = {
        "jump": {"file_name": "phase-gap.csv"},
        "metrics": {
            "best_3s_start_s": 20.0,
            "best_3s_end_s": 23.0,
            "best_3s_vVert_kmh": 408.0,
            "best_3s_vHor_kmh": 30.0,
        },
        "notes": {
            "curve_window_start_s": 0.0,
            "curve_window_end_s": 32.0,
        },
        "scorecard": {
            "exit": "sauber",
            "phase_10_20": "optimal",
            "hot_zone": "stabil",
            "kipp_risiko": "niedrig",
        },
        "quality_flags": [],
        "fixpoints": [
            {"t_rel_s": 10.0, "vVert_kmh": 250.0, "vHor_kmh": 90.0, "angle_deg": 68.0},
            {"t_rel_s": 15.0, "vVert_kmh": 330.0, "vHor_kmh": 74.0, "angle_deg": 76.0},
            {"t_rel_s": 20.0, "vVert_kmh": 390.0, "vHor_kmh": 50.0, "angle_deg": 83.5},
            {"t_rel_s": 24.0, "vVert_kmh": 405.0, "vHor_kmh": 36.0, "angle_deg": 85.0},
        ],
    }
    stability_reference = {
        "available": True,
        "thresholds": {
            "phase_10_15_gain_low": 95.0,
            "phase_10_15_gain_high": 110.0,
            "phase_15_20_gain_low": 70.0,
            "phase_15_20_gain_high": 85.0,
            "angle_20_target_low": 82.5,
            "angle_20_target_high": 84.8,
        },
        "capability_profile": {
            "mode": "safe",
            "text": "Letzte 8 Sprünge: 3/8 stabil (38%). Erst Stabilität sichern, dann Tempo pushen.",
            "stable_ratio_pct": 38.0,
        },
    }

    review = build_jump_review(
        report,
        best_compare=None,
        jumper_stability_reference=stability_reference,
    )

    assert any("Phase +10 bis +15s" in line for line in review["not_good"])
    assert any("Personalisierter Modus" in line for line in review["happened"])


def test_build_jump_review_detects_too_fast_steep_not_hold_pattern():
    time_s = [float(i) for i in range(0, 27)]
    angle = []
    for i in range(0, 27):
        t = float(i)
        if t <= 10.0:
            value = 62.0 + 1.2 * t
        elif t <= 15.0:
            value = 74.0 + 2.4 * (t - 10.0)
        elif t <= 18.0:
            value = 86.0 + 0.6 * (t - 15.0)
        else:
            value = 87.8 - 1.2 * (t - 18.0)
            value += 0.9 if i % 2 == 0 else -0.9
        angle.append(value)
    vvert = [220.0 + 9.0 * t if t <= 18.0 else 382.0 - 7.0 * (t - 18.0) for t in time_s]
    vhor = [96.0 - 2.8 * t for t in time_s]

    report = {
        "jump": {"file_name": "anna-steep.csv"},
        "metrics": {
            "best_3s_start_s": 20.0,
            "best_3s_end_s": 23.0,
            "best_3s_vVert_kmh": 402.0,
            "best_3s_vHor_kmh": 24.0,
        },
        "notes": {
            "curve_window_start_s": 0.0,
            "curve_window_end_s": 26.0,
            "decel_start_s": 30.0,
        },
        "scorecard": {
            "exit": "sauber",
            "phase_10_20": "optimal",
            "hot_zone": "kritisch",
            "kipp_risiko": "hoch",
        },
        "quality_flags": [],
        "fixpoints": [
            {"t_rel_s": 10.0, "vVert_kmh": 300.0, "vHor_kmh": 67.0, "angle_deg": 74.0},
            {"t_rel_s": 15.0, "vVert_kmh": 350.0, "vHor_kmh": 52.0, "angle_deg": 86.0},
            {"t_rel_s": 20.0, "vVert_kmh": 389.0, "vHor_kmh": 35.0, "angle_deg": 84.0},
            {"t_rel_s": 24.0, "vVert_kmh": 368.0, "vHor_kmh": 30.0, "angle_deg": 80.0},
        ],
        "chart_data": {
            "time_s": time_s,
            "vVert_kmh": vvert,
            "vHor_kmh": vhor,
            "angle_deg": angle,
            "accVert_mps2": [2.4 for _ in time_s],
            "hAGL_m": [3600.0 - 55.0 * t for t in time_s],
        },
    }
    stability_reference = {
        "available": True,
        "thresholds": {
            "phase_10_15_angle_high": 82.5,
            "angle_20_target_low": 82.0,
            "angle_20_target_high": 84.5,
            "angle_turns_20_25_max": 2.0,
        },
    }

    review = build_jump_review(
        report,
        best_compare=None,
        jumper_stability_reference=stability_reference,
    )

    assert any("zu schnell steil" in line for line in review["not_good"])
    assert any("nicht zu schnell maximal steil" in line for line in review["improve"])


def test_build_jump_review_detects_late_hard_steepening_vhor_collapse():
    time_s = [float(i) for i in range(0, 27)]
    angle = []
    vhor = []
    vvert = []
    for i in range(0, 27):
        t = float(i)
        if t <= 20.0:
            angle.append(66.0 + 0.62 * t)
            vhor.append(96.0 - 1.45 * t)
            vvert.append(220.0 + 5.2 * t)
        elif t <= 23.0:
            angle.append(78.4 + 2.65 * (t - 20.0))
            vhor.append(67.0 - 15.0 * (t - 20.0))
            vvert.append(324.0 + 1.0 * (t - 20.0))
        else:
            angle.append(86.4 - 0.3 * (t - 23.0))
            vhor.append(22.0 + 7.0 * (t - 23.0))
            vvert.append(327.0 - 2.0 * (t - 23.0))

    report = {
        "jump": {"file_name": "late-hard.csv"},
        "metrics": {
            "best_3s_start_s": 22.0,
            "best_3s_end_s": 25.0,
            "best_3s_vVert_kmh": 326.0,
            "best_3s_vHor_kmh": 24.0,
        },
        "notes": {
            "curve_window_start_s": 0.0,
            "curve_window_end_s": 26.0,
            "decel_start_s": 30.0,
        },
        "scorecard": {
            "exit": "sauber",
            "phase_10_20": "zu flach",
            "hot_zone": "kritisch",
            "kipp_risiko": "hoch",
        },
        "quality_flags": [],
        "fixpoints": [
            {"t_rel_s": 10.0, "vVert_kmh": 272.0, "vHor_kmh": 81.5, "angle_deg": 72.2},
            {"t_rel_s": 15.0, "vVert_kmh": 298.0, "vHor_kmh": 74.2, "angle_deg": 75.3},
            {"t_rel_s": 20.0, "vVert_kmh": 324.0, "vHor_kmh": 67.0, "angle_deg": 78.4},
            {"t_rel_s": 24.0, "vVert_kmh": 325.0, "vHor_kmh": 29.0, "angle_deg": 86.1},
        ],
        "chart_data": {
            "time_s": time_s,
            "vVert_kmh": vvert,
            "vHor_kmh": vhor,
            "angle_deg": angle,
            "accVert_mps2": [2.0 for _ in time_s],
            "hAGL_m": [3600.0 - 55.0 * t for t in time_s],
        },
    }
    stability_reference = {
        "available": True,
        "thresholds": {
            "angle_20_target_low": 78.0,
            "angle_20_target_high": 85.0,
            "angle_20_risk_above": 85.0,
            "phase_10_15_angle_high": 85.0,
            "vhor_min_20_25_floor": 35.0,
        },
    }

    review = build_jump_review(
        report,
        best_compare=None,
        jumper_stability_reference=stability_reference,
    )

    diagnosis = review["primary_diagnosis"]
    assert diagnosis["available"] is True
    assert diagnosis["pattern"] == "late_hard_steepening_vhor_collapse"
    assert diagnosis["evidence"]["angle_20"] < diagnosis["evidence"]["angle_peak"]
    assert any("Steilflug kommt zu hart" in line for line in review["not_good"])
    assert any("Übergang in den Steilflug" in line for line in review["improve"])


def test_build_jump_review_prioritizes_early_horizontal_reserve_collapse():
    time_s = [i * 0.5 for i in range(0, 61)]
    vvert = []
    vhor = []
    angle = []
    forward_m = []
    for t in time_s:
        vvert.append(230.0 + min(t, 25.0) * 8.2)
        if t <= 18.0:
            vhor.append(92.0 - t * 3.0)
        elif t <= 21.0:
            vhor.append(38.0 - (t - 18.0) * 5.2)
        elif t <= 25.0:
            vhor.append(22.4 + (t - 21.0) * 5.5)
        else:
            vhor.append(44.4 + min(3.0, (t - 25.0) * 0.8))

        if t <= 21.0:
            angle.append(70.0 + t * 0.82)
        elif t <= 25.0:
            angle.append(87.2 - (t - 21.0) * 0.75)
        else:
            angle.append(84.2)

        if t <= 21.0:
            forward_m.append(t * 28.0)
        else:
            forward_m.append(588.0 - (t - 21.0) * 12.0)

    report = {
        "jump": {"file_name": "early-reserve.csv", "jump_id": "early-reserve"},
        "metrics": {
            "best_3s_start_s": 24.5,
            "best_3s_end_s": 27.5,
            "best_3s_vVert_kmh": 432.0,
            "best_3s_vHor_kmh": 45.0,
        },
        "notes": {"curve_window_start_s": 0.0, "curve_window_end_s": 30.0, "decel_start_s": 31.0},
        "scorecard": {"exit": "sauber", "phase_10_20": "optimal", "hot_zone": "stabil", "kipp_risiko": "niedrig"},
        "quality_flags": [],
        "fixpoints": [
            {"t_rel_s": 10.0, "vVert_kmh": 312.0, "vHor_kmh": 62.0, "angle_deg": 78.2},
            {"t_rel_s": 15.0, "vVert_kmh": 353.0, "vHor_kmh": 47.0, "angle_deg": 82.3},
            {"t_rel_s": 20.0, "vVert_kmh": 394.0, "vHor_kmh": 27.6, "angle_deg": 86.4},
            {"t_rel_s": 24.0, "vVert_kmh": 426.8, "vHor_kmh": 38.9, "angle_deg": 85.0},
            {"t_rel_s": 28.0, "vVert_kmh": 435.0, "vHor_kmh": 46.8, "angle_deg": 84.2},
        ],
        "chart_data": {
            "time_s": time_s,
            "vVert_kmh": vvert,
            "vHor_kmh": vhor,
            "angle_deg": angle,
            "accVert_mps2": [2.0 for _ in time_s],
            "hAGL_m": [3600.0 - 55.0 * t for t in time_s],
            "forward_m": forward_m,
        },
    }
    stability_reference = {
        "available": True,
        "thresholds": {
            "angle_20_target_low": 82.0,
            "angle_20_target_high": 86.0,
            "angle_20_risk_above": 86.0,
            "vhor_min_20_25_floor": 35.0,
        },
    }

    review = build_jump_review(
        report,
        best_compare=None,
        jumper_stability_reference=stability_reference,
    )

    diagnosis = review["primary_diagnosis"]
    assert diagnosis["available"] is True
    assert diagnosis["pattern"] == "early_horizontal_reserve_collapse"
    assert diagnosis["evidence"]["vhor_min_time_s"] < diagnosis["evidence"]["best_3s_start_s"]
    assert any("horizontale Reserve" in line for line in review["not_good"])
    assert review["coaching_goals"][0]["id"] == "early_horizontal_reserve_collapse"
    assert any("horizontale Reserve" in line for line in review["technical_assessment"]["issues"])


def test_build_jump_review_flags_late_personal_timing_without_changing_phases():
    time_s = [float(i) for i in range(0, 32)]
    angle = [68.0 + 0.47 * t for t in time_s]
    report = {
        "jump": {"file_name": "late-personal-timing.csv", "jump_id": "late-personal-timing"},
        "metrics": {
            "best_3s_start_s": 25.0,
            "best_3s_end_s": 28.0,
            "best_3s_vVert_kmh": 425.0,
            "best_3s_vHor_kmh": 38.0,
        },
        "notes": {"curve_window_start_s": 0.0, "curve_window_end_s": 31.0, "decel_start_s": 31.0},
        "scorecard": {"exit": "sauber", "phase_10_20": "optimal", "hot_zone": "stabil", "kipp_risiko": "niedrig"},
        "quality_flags": [],
        "fixpoints": [
            {"t_rel_s": 10.0, "vVert_kmh": 285.0, "vHor_kmh": 74.0, "angle_deg": 72.5},
            {"t_rel_s": 15.0, "vVert_kmh": 330.0, "vHor_kmh": 62.0, "angle_deg": 74.8},
            {"t_rel_s": 20.0, "vVert_kmh": 376.0, "vHor_kmh": 50.0, "angle_deg": 77.0},
            {"t_rel_s": 24.0, "vVert_kmh": 410.0, "vHor_kmh": 41.0, "angle_deg": 78.8},
            {"t_rel_s": 28.0, "vVert_kmh": 426.0, "vHor_kmh": 38.0, "angle_deg": 80.6},
        ],
        "chart_data": {
            "time_s": time_s,
            "vVert_kmh": [230.0 + 6.3 * t for t in time_s],
            "vHor_kmh": [95.0 - 1.9 * t for t in time_s],
            "angle_deg": angle,
            "accVert_mps2": [2.1 for _ in time_s],
            "hAGL_m": [3600.0 - 55.0 * t for t in time_s],
            "forward_m": [18.0 * t for t in time_s],
        },
    }
    stability_reference = {
        "available": True,
        "thresholds": {},
        "timing_reference": {
            "available": True,
            "maturity": "active",
            "confidence": "medium",
            "basis_count": 5,
            "anchors": {
                "angle_82_time_s": {"low": 12.5, "high": 14.0, "median": 13.2},
                "best_3s_start_s": {"low": 20.0, "high": 22.0, "median": 21.0},
            },
        },
    }

    review = build_jump_review(
        report,
        best_compare=None,
        jumper_stability_reference=stability_reference,
    )

    assert any("Persoenliches Timing: 82 Grad" in line for line in review["not_good"])
    assert any("beste 3s-Fenster startet spaeter" in line for line in review["not_good"])
    assert [phase["name"] for phase in review["technical_assessment"]["phases"]][:5] == [
        "Exit / Stabilisierung",
        "Dive-Aufbau",
        "Hauptbeschleunigung",
        "Hot-Zone Aufbau",
        "Max-Speed Fenster",
    ]


def test_build_jump_review_does_not_mix_fs2_quality_into_flight_feedback():
    time_s = [float(i) for i in range(0, 31)]
    report = {
        "jump": {"file_name": "fs2.csv", "jump_id": "j-fs2"},
        "metrics": {
            "best_3s_start_s": 20.0,
            "best_3s_end_s": 23.0,
            "best_3s_vVert_kmh": 405.0,
            "best_3s_vHor_kmh": 27.0,
        },
        "notes": {
            "curve_window_start_s": 0.0,
            "curve_window_end_s": 30.0,
            "decel_start_s": 31.0,
            "fs2_track_summary": {
                "available": True,
                "window_start_s": 20.0,
                "window_end_s": 25.0,
                "sAcc_p95": 2.4,
                "hAcc_p95": 15.1,
                "numSV_p10": 9.0,
                "quality_label": "kritisch",
            },
        },
        "scorecard": {
            "exit": "sauber",
            "phase_10_20": "optimal",
            "hot_zone": "kritisch",
            "kipp_risiko": "mittel",
        },
        "quality_flags": [],
        "fixpoints": [
            {"t_rel_s": 10.0, "vVert_kmh": 285.0, "vHor_kmh": 74.0, "angle_deg": 76.0},
            {"t_rel_s": 15.0, "vVert_kmh": 340.0, "vHor_kmh": 63.0, "angle_deg": 81.0},
            {"t_rel_s": 20.0, "vVert_kmh": 395.0, "vHor_kmh": 40.0, "angle_deg": 84.0},
            {"t_rel_s": 24.0, "vVert_kmh": 402.0, "vHor_kmh": 30.0, "angle_deg": 85.0},
            {"t_rel_s": 28.0, "vVert_kmh": 360.0, "vHor_kmh": 60.0, "angle_deg": 78.0},
        ],
        "chart_data": {
            "time_s": time_s,
            "vVert_kmh": [240 + i * 5.3 for i in range(0, 31)],
            "vHor_kmh": [100 - i * 2.2 for i in range(0, 31)],
            "angle_deg": [70 + i * 0.55 for i in range(0, 31)],
            "hAGL_m": [3600 - i * 55 for i in range(0, 31)],
            "accVert_mps2": [2.1 for _ in time_s],
            "velN_mps": [18.0 for _ in time_s],
            "velE_mps": [1.0 for _ in time_s],
            "forward_m": [i * 5.0 for i in range(0, 31)],
            "backtrack_m": [0.0 for _ in range(0, 31)],
        },
    }

    review = build_jump_review(report, best_compare=None)

    combined = review["happened"] + review["not_good"] + review["improve"]
    assert not any("FS2-Werte" in line for line in combined)


def test_build_jump_review_adds_best_3s_window_quality_issue():
    time_s = [i * 0.5 for i in range(0, 58)]
    vvert = []
    angle = []
    vhor = []
    for i, t in enumerate(time_s):
        if 20.0 <= t <= 23.0:
            vvert.append(430.0 + (18.0 if i % 2 == 0 else -18.0))
            angle.append(85.0 + (2.4 if i % 2 == 0 else -2.4))
            vhor.append(23.0)
        else:
            vvert.append(240.0 + min(t, 20.0) * 8.0)
            angle.append(70.0 + min(t, 20.0) * 0.7)
            vhor.append(86.0 - min(t, 24.0) * 2.0)

    report = {
        "jump": {"file_name": "unstable-window.csv"},
        "metrics": {
            "best_3s_start_s": 20.0,
            "best_3s_end_s": 23.0,
            "best_3s_vVert_kmh": 430.0,
            "best_3s_vHor_kmh": 23.0,
        },
        "notes": {"curve_window_start_s": 0.0, "curve_window_end_s": 28.0, "decel_start_s": 29.0},
        "scorecard": {"exit": "sauber", "phase_10_20": "optimal", "hot_zone": "stabil", "kipp_risiko": "niedrig"},
        "quality_flags": [],
        "fixpoints": [
            {"t_rel_s": 10.0, "vVert_kmh": 320.0, "vHor_kmh": 66.0, "angle_deg": 77.0},
            {"t_rel_s": 20.0, "vVert_kmh": 430.0, "vHor_kmh": 23.0, "angle_deg": 87.4},
            {"t_rel_s": 24.0, "vVert_kmh": 395.0, "vHor_kmh": 38.0, "angle_deg": 82.0},
        ],
        "chart_data": {
            "time_s": time_s,
            "vVert_kmh": vvert,
            "vHor_kmh": vhor,
            "angle_deg": angle,
            "accVert_mps2": [0.7 if 20 <= t <= 23 else 2.1 for t in time_s],
            "hAGL_m": [3600.0 - 55.0 * t for t in time_s],
        },
    }

    review = build_jump_review(report, best_compare=None)

    assert review["technical_assessment"]["available"] is True
    assert review["technical_assessment"]["best_window_quality"]["label"] == "kritisch"
    assert any("3s-Fenster" in line for line in review["not_good"])
    assert any("beste 3s-Zone" in item["text"] for item in review["technical_assessment"]["actions"])


def test_best_3s_window_quality_issue_only_names_actual_causes():
    time_s = [i * 0.5 for i in range(0, 58)]
    vvert = [240.0 + min(t, 23.0) * 7.0 for t in time_s]
    angle = [72.0 + min(t, 23.0) * 0.45 for t in time_s]
    vhor = [86.0 - min(t, 23.0) * 2.8 for t in time_s]
    for i, t in enumerate(time_s):
        if 20.0 <= t <= 23.0:
            vvert[i] = 402.0 + (0.4 if i % 2 == 0 else -0.4)
            angle[i] = 84.0
            vhor[i] = 21.5
        elif 23.0 < t <= 25.0:
            vvert[i] = 370.0

    report = {
        "jump": {"file_name": "low-vvert-std-window.csv"},
        "metrics": {
            "best_3s_start_s": 20.0,
            "best_3s_end_s": 23.0,
            "best_3s_vVert_kmh": 402.0,
            "best_3s_vHor_kmh": 21.5,
        },
        "notes": {"curve_window_start_s": 0.0, "curve_window_end_s": 28.0, "decel_start_s": 29.0},
        "scorecard": {"exit": "sauber", "phase_10_20": "optimal", "hot_zone": "kritisch", "kipp_risiko": "mittel"},
        "quality_flags": [],
        "fixpoints": [
            {"t_rel_s": 10.0, "vVert_kmh": 310.0, "vHor_kmh": 58.0, "angle_deg": 76.5},
            {"t_rel_s": 20.0, "vVert_kmh": 402.0, "vHor_kmh": 21.5, "angle_deg": 84.0},
            {"t_rel_s": 24.0, "vVert_kmh": 370.0, "vHor_kmh": 38.0, "angle_deg": 82.0},
        ],
        "chart_data": {
            "time_s": time_s,
            "vVert_kmh": vvert,
            "vHor_kmh": vhor,
            "angle_deg": angle,
            "accVert_mps2": [0.6 for _ in time_s],
            "hAGL_m": [3600.0 - 55.0 * t for t in time_s],
        },
    }

    review = build_jump_review(report, best_compare=None)
    issue = next(line for line in review["not_good"] if "3s-Fenster" in line)

    assert "vHor-Min" in issue
    assert "Speed-Drop danach" in issue
    assert "vVert-Streuung" not in issue


def test_best_3s_window_quality_ignores_exit_drop_without_clean_after_window():
    time_s = [i * 0.5 for i in range(0, 54)]
    vvert = []
    angle = []
    vhor = []
    for t in time_s:
        if 20.0 <= t <= 23.0:
            vvert.append(430.0)
            angle.append(84.0)
            vhor.append(35.0)
        elif 23.0 < t <= 25.0:
            vvert.append(330.0)
            angle.append(78.0)
            vhor.append(65.0)
        else:
            vvert.append(250.0 + min(t, 20.0) * 9.0)
            angle.append(72.0 + min(t, 20.0) * 0.55)
            vhor.append(90.0 - min(t, 20.0) * 2.0)

    report = {
        "jump": {"file_name": "exit-drop-window.csv"},
        "metrics": {
            "best_3s_start_s": 20.0,
            "best_3s_end_s": 23.0,
            "best_3s_vVert_kmh": 430.0,
            "best_3s_vHor_kmh": 35.0,
        },
        "notes": {"curve_window_start_s": 0.0, "curve_window_end_s": 28.0, "decel_start_s": 23.2},
        "scorecard": {"exit": "sauber", "phase_10_20": "optimal", "hot_zone": "stabil", "kipp_risiko": "niedrig"},
        "quality_flags": [],
        "fixpoints": [
            {"t_rel_s": 10.0, "vVert_kmh": 320.0, "vHor_kmh": 70.0, "angle_deg": 77.0},
            {"t_rel_s": 20.0, "vVert_kmh": 430.0, "vHor_kmh": 35.0, "angle_deg": 84.0},
            {"t_rel_s": 24.0, "vVert_kmh": 330.0, "vHor_kmh": 65.0, "angle_deg": 78.0},
        ],
        "chart_data": {
            "time_s": time_s,
            "vVert_kmh": vvert,
            "vHor_kmh": vhor,
            "angle_deg": angle,
            "accVert_mps2": [0.4 for _ in time_s],
            "hAGL_m": [3600.0 - 55.0 * t for t in time_s],
        },
    }

    review = build_jump_review(report, best_compare=None)
    best_window = review["technical_assessment"]["best_window_quality"]

    assert best_window["label"] == "stabil"
    assert best_window["drop_after_evaluable"] is False
    assert best_window["vvert_drop_after_kmh"] == 0.0
    assert not any("Speed-Drop danach" in line for line in review["not_good"])
    assert not any("kurzer Peak" in line for line in review["not_good"])


def test_best_3s_window_quality_uses_rule_window_before_late_raw_top_speed():
    time_s = [i * 0.5 for i in range(0, 64)]
    vvert = []
    angle = []
    vhor = []
    for t in time_s:
        if 20.0 <= t <= 23.0:
            vvert.append(410.0)
            angle.append(84.0)
            vhor.append(34.0)
        elif 26.0 <= t <= 29.0:
            vvert.append(470.0)
            angle.append(87.0)
            vhor.append(20.0)
        elif 29.0 < t <= 31.0:
            vvert.append(350.0)
            angle.append(78.0)
            vhor.append(60.0)
        else:
            vvert.append(250.0 + min(t, 20.0) * 8.0)
            angle.append(72.0 + min(t, 20.0) * 0.55)
            vhor.append(90.0 - min(t, 20.0) * 2.0)

    report = {
        "jump": {"file_name": "late-raw-peak.csv"},
        "metrics": {
            "best_3s_start_s": 26.0,
            "best_3s_end_s": 29.0,
            "best_3s_vVert_kmh": 470.0,
            "best_3s_vHor_kmh": 20.0,
            "rule_based_3s_score": 410.0,
            "performance_window_start_s": 0.0,
            "performance_window_end_s": 23.0,
        },
        "notes": {"curve_window_start_s": 0.0, "curve_window_end_s": 31.0, "decel_start_s": 31.0},
        "scorecard": {"exit": "sauber", "phase_10_20": "optimal", "hot_zone": "stabil", "kipp_risiko": "niedrig"},
        "quality_flags": [],
        "fixpoints": [
            {"t_rel_s": 10.0, "vVert_kmh": 320.0, "vHor_kmh": 70.0, "angle_deg": 77.0},
            {"t_rel_s": 20.0, "vVert_kmh": 410.0, "vHor_kmh": 34.0, "angle_deg": 84.0},
            {"t_rel_s": 28.0, "vVert_kmh": 470.0, "vHor_kmh": 20.0, "angle_deg": 87.0},
        ],
        "chart_data": {
            "time_s": time_s,
            "vVert_kmh": vvert,
            "vHor_kmh": vhor,
            "angle_deg": angle,
            "accVert_mps2": [0.5 for _ in time_s],
            "hAGL_m": [3600.0 - 55.0 * t for t in time_s],
        },
    }

    review = build_jump_review(report, best_compare=None)
    best_window = review["technical_assessment"]["best_window_quality"]

    assert best_window["source"] == "rule_window"
    assert best_window["end_s"] <= 23.1
    assert best_window["label"] == "stabil"
    assert not any("kurzer Peak" in line for line in review["not_good"])


def test_best_3s_window_quality_keeps_real_drop_before_decel():
    time_s = [i * 0.5 for i in range(0, 58)]
    vvert = []
    angle = []
    vhor = []
    for t in time_s:
        if 20.0 <= t <= 23.0:
            vvert.append(430.0)
            angle.append(84.0)
            vhor.append(35.0)
        elif 23.0 < t <= 25.0:
            vvert.append(390.0)
            angle.append(83.0)
            vhor.append(36.0)
        else:
            vvert.append(250.0 + min(t, 20.0) * 9.0)
            angle.append(72.0 + min(t, 20.0) * 0.55)
            vhor.append(90.0 - min(t, 20.0) * 2.0)

    report = {
        "jump": {"file_name": "real-pre-decel-drop.csv"},
        "metrics": {
            "best_3s_start_s": 20.0,
            "best_3s_end_s": 23.0,
            "best_3s_vVert_kmh": 430.0,
            "best_3s_vHor_kmh": 35.0,
        },
        "notes": {"curve_window_start_s": 0.0, "curve_window_end_s": 28.0, "decel_start_s": 27.0},
        "scorecard": {"exit": "sauber", "phase_10_20": "optimal", "hot_zone": "stabil", "kipp_risiko": "niedrig"},
        "quality_flags": [],
        "fixpoints": [
            {"t_rel_s": 10.0, "vVert_kmh": 320.0, "vHor_kmh": 70.0, "angle_deg": 77.0},
            {"t_rel_s": 20.0, "vVert_kmh": 430.0, "vHor_kmh": 35.0, "angle_deg": 84.0},
            {"t_rel_s": 24.0, "vVert_kmh": 390.0, "vHor_kmh": 36.0, "angle_deg": 83.0},
        ],
        "chart_data": {
            "time_s": time_s,
            "vVert_kmh": vvert,
            "vHor_kmh": vhor,
            "angle_deg": angle,
            "accVert_mps2": [0.4 for _ in time_s],
            "hAGL_m": [3600.0 - 55.0 * t for t in time_s],
        },
    }

    review = build_jump_review(report, best_compare=None)
    best_window = review["technical_assessment"]["best_window_quality"]

    assert best_window["drop_after_evaluable"] is True
    assert best_window["vvert_drop_after_kmh"] >= 39.0
    assert any("Speed-Drop danach" in line for line in review["not_good"])


def test_build_jump_review_detects_steeper_phase_without_acceleration_gain():
    time_s = [float(i) for i in range(0, 27)]
    angle = []
    vvert = []
    vhor = []
    acc = []
    for i in range(0, 27):
        t = float(i)
        if t <= 8.0:
            angle.append(60.0 + 1.5 * t)
            vvert.append(230.0 + 8.0 * t)
            acc.append(2.2)
        elif t <= 15.0:
            angle.append(72.0 + 2.0 * (t - 8.0))
            vvert.append(294.0 + 3.0 * (t - 8.0))
            acc.append(0.5)
        else:
            angle.append(86.0 - 0.2 * (t - 15.0))
            vvert.append(315.0 + 2.0 * (t - 15.0))
            acc.append(0.4)
        vhor.append(90.0 - 1.8 * min(t, 24.0))

    report = {
        "jump": {"file_name": "steep-no-gain.csv"},
        "metrics": {
            "best_3s_start_s": 21.0,
            "best_3s_end_s": 24.0,
            "best_3s_vVert_kmh": 330.0,
            "best_3s_vHor_kmh": 45.0,
        },
        "notes": {"curve_window_start_s": 0.0, "curve_window_end_s": 26.0, "decel_start_s": 27.0},
        "scorecard": {"exit": "sauber", "phase_10_20": "optimal", "hot_zone": "stabil", "kipp_risiko": "niedrig"},
        "quality_flags": [],
        "fixpoints": [
            {"t_rel_s": 10.0, "vVert_kmh": 300.0, "vHor_kmh": 72.0, "angle_deg": 76.0},
            {"t_rel_s": 15.0, "vVert_kmh": 315.0, "vHor_kmh": 63.0, "angle_deg": 86.0},
            {"t_rel_s": 20.0, "vVert_kmh": 325.0, "vHor_kmh": 54.0, "angle_deg": 85.0},
        ],
        "chart_data": {
            "time_s": time_s,
            "vVert_kmh": vvert,
            "vHor_kmh": vhor,
            "angle_deg": angle,
            "accVert_mps2": acc,
            "hAGL_m": [3600.0 - 50.0 * t for t in time_s],
        },
    }

    review = build_jump_review(report, best_compare=None)
    phase_issues = [
        phase
        for phase in review["technical_assessment"]["phases"]
        if phase.get("steep_without_gain")
    ]

    assert phase_issues
    assert any("profitiert kaum" in line for line in review["not_good"])
    assert any("nicht einfach weiter aufdruecken" in line for line in review["improve"])
