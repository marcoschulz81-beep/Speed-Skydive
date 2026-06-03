from __future__ import annotations

from app.analysis.review import _build_priority_actions, _forward_eval_end_s, build_jump_review


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
    assert set(review.keys()) == {"happened", "good", "not_good", "improve"}
    assert len(review["happened"]) >= 1
    assert len(review["good"]) >= 1
    assert len(review["not_good"]) >= 1
    assert len(review["improve"]) >= 1


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

    assert any("Prioritaet" in item for item in review["improve"])
    assert any("Startphase" in item or "Zwischen +10s und +20s mehr Druck aufbauen" in item for item in review["improve"])


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
            "text": "In der Peak-Phase Druck ruhiger halten und kleine, fruehe Korrekturen machen.",
        },
        "build_speed_low": {
            "score": 9,
            "text": "Zwischen +10s und +20s mehr Druck aufbauen. Ziel: in diesem Abschnitt mindestens +90 km/h Zuwachs.",
        },
        "exit_carryover_low": {
            "score": 7,
            "text": "Nach dem Exit den Druck laenger tragen: ab +2s stabil weiter beschleunigen, statt frueh nachzulassen.",
        },
        "kipp_risk_high": {
            "score": 8,
            "text": "Bei Instabilitaet Koerperspannung frueher stabilisieren (Schulter und Huefte).",
        },
    }

    ordered = _build_priority_actions(actions, max_items=5)
    plain = [item.split(": ", 1)[1] for item in ordered]

    exit_idx = plain.index(actions["exit_carryover_low"]["text"])
    build_idx = plain.index(actions["build_speed_low"]["text"])
    peak_idx = plain.index(actions["peak_stability"]["text"])
    stability_idx = plain.index(actions["kipp_risk_high"]["text"])

    assert exit_idx < build_idx < peak_idx < stability_idx


def test_build_jump_review_detects_negative_forward_drift():
    time_s = [float(i) for i in range(0, 31)]
    vvert = [220.0 + i * 5.5 for i in range(0, 31)]
    vhor = [95.0 - i * 2.0 for i in range(0, 31)]
    angle = [70.0 + i * 0.6 for i in range(0, 31)]
    acc = [2.5 for _ in time_s]
    # Vorwaerts-Strecke: erst vorwaerts, dann klare Rueckdrift.
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
    assert any("Rueckdrift" in item for item in review["happened"])


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
            "text": "Letzte 8 Spruenge: 3/8 stabil (38%). Erst Stabilitaet sichern, dann Tempo pushen.",
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
    assert not any("Messqualitaet (FS2)" in line for line in combined)
