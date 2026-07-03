from __future__ import annotations

import re
from statistics import mean, pstdev
from typing import Any

import numpy as np

from app.text_utils import normalize_german_text

from app.analysis.lateral import analyze_lateral_dynamics


def build_jump_review(
    report: dict[str, Any],
    *,
    best_compare: dict[str, Any] | None = None,
    external_compare: dict[str, Any] | None = None,
    top_reference_compares: list[dict[str, Any]] | None = None,
    jumper_stability_reference: dict[str, Any] | None = None,
    tip_effect_profile: dict[str, Any] | None = None,
) -> dict[str, list[str]]:
    jump = report["jump"]
    metrics = report["metrics"]
    notes = report.get("notes", {})
    scorecard = report.get("scorecard", {})
    quality_flags = set(report.get("quality_flags", []))
    fixpoints = report.get("fixpoints", [])
    chart_data = report.get("chart_data", {})
    exit_profile = notes.get("exit_profile", {}) if isinstance(notes.get("exit_profile"), dict) else {}
    exit_unsteady = bool(exit_profile.get("unsteady"))
    early_acc_p95 = _num(exit_profile.get("early_acc_p95_abs_0_5"))
    analysis_blocked = bool(notes.get("analysis_blocked"))
    analysis_block_reason = str(notes.get("analysis_block_reason") or "").strip()
    personal_profile = _extract_personal_tip_profile(jumper_stability_reference)
    capability_mode = str(personal_profile.get("capability_mode") or "")
    capability_text = str(personal_profile.get("capability_text") or "").strip()
    performance_text = str(personal_profile.get("performance_text") or "").strip()

    if analysis_blocked:
        reason = analysis_block_reason or "Sprungdaten nicht korrekt. Sprung endet zu früh."
        improve = [
            "Dieser Datensatz wird für die Technikbewertung nicht genutzt.",
            "Bitte Absprung neu erkennen lassen oder die Originaldatei neu einlesen.",
            "Wenn der Track wirklich so kurz ist, den Sprung nicht für Speed-Vergleiche verwenden.",
        ]
        if "TIME_GAPS" in quality_flags:
            improve.append("Im Track gibt es Zeitlücken; möglichst lückenfreie Aufzeichnung verwenden.")
        if "SPEED_SPIKE" in quality_flags:
            improve.append("SPEED_SPIKE erkannt; Datensatz für Ranking/Bestwert ausschließen.")
        return {
            "happened": [
                f"Bestes 3-Sekunden-Fenster: +{metrics['best_3s_start_s']}s bis +{metrics['best_3s_end_s']}s mit {metrics['best_3s_vVert_kmh']} km/h.",
                f"Ausgewerteter Hauptbereich: +{notes.get('curve_window_start_s')}s bis +{notes.get('curve_window_end_s')}s.",
            ],
            "good": ["Keine verlässliche Technikbewertung möglich."],
            "not_good": [reason],
            "improve": improve[:5],
        }

    fp10 = _fixpoint_at(fixpoints, 10.0)
    fp15 = _fixpoint_at(fixpoints, 15.0)
    fp20 = _fixpoint_at(fixpoints, 20.0)
    fp24 = _fixpoint_at(fixpoints, 24.0)
    fp28 = _fixpoint_at(fixpoints, 28.0)

    start_vvert = _num(fp10.get("vVert_kmh")) if fp10 else None
    vvert_15_raw = _num(fp15.get("vVert_kmh")) if fp15 else None
    vvert_20_raw = _num(fp20.get("vVert_kmh")) if fp20 else None
    angle_20_raw = _num(fp20.get("angle_deg")) if fp20 else None
    vvert_20 = vvert_20_raw
    angle_20 = angle_20_raw
    gain_10_20 = None if start_vvert is None or vvert_20 is None else (vvert_20 - start_vvert)
    gain_10_15 = None if start_vvert is None or vvert_15_raw is None else (vvert_15_raw - start_vvert)
    gain_15_20 = None if vvert_15_raw is None or vvert_20 is None else (vvert_20 - vvert_15_raw)
    vhor_24 = _num(fp24.get("vHor_kmh")) if fp24 else None
    vhor_28 = _num(fp28.get("vHor_kmh")) if fp28 else None
    early_acc_mean, early_acc_peak = _early_acc_stats(chart_data, start_s=0.0, end_s=6.0)

    happened: list[str] = []
    good: list[str] = []
    not_good: list[str] = []
    actions_by_key: dict[str, dict[str, Any]] = {}

    happened.append(
        "Bestes 3-Sekunden-Fenster: +"
        f"{metrics['best_3s_start_s']}s bis +{metrics['best_3s_end_s']}s mit "
        f"{metrics['best_3s_vVert_kmh']} km/h."
    )

    curve_start = notes.get("curve_window_start_s")
    curve_end = notes.get("curve_window_end_s")
    curve_start_num = _num(curve_start)
    curve_end_num = _num(curve_end)
    best_3s_end_num = _num(metrics.get("best_3s_end_s"))
    canopy_open_num = _num(notes.get("canopy_open_s"))
    hot_zone_end_num = _num(metrics.get("hot_zone_end_s"))
    pw_end_num = _num(metrics.get("performance_window_end_s"))
    decel_start_num = _num(notes.get("decel_start_s"))
    eval_end_s = _effective_eval_window_end_s(notes=notes, chart_data=chart_data)
    window_supports_20 = True
    if eval_end_s is not None and eval_end_s < 19.5:
        window_supports_20 = False
    if not window_supports_20:
        vvert_20 = None
        angle_20 = None
        gain_10_20 = None
        gain_15_20 = None

    coaching_end_s = _coaching_focus_end(
        curve_end_s=curve_end_num,
        hot_zone_end_s=hot_zone_end_num,
    )
    if eval_end_s is not None:
        if coaching_end_s is None:
            coaching_end_s = eval_end_s
        else:
            coaching_end_s = min(float(coaching_end_s), float(eval_end_s))
    forward_focus_end_s = _forward_eval_end_s(
        decel_start_s=decel_start_num,
        fallback_end_s=curve_end_num,
    )

    if curve_start is not None and curve_end is not None:
        happened.append(f"Ausgewerteter Hauptbereich: +{curve_start}s bis +{curve_end}s.")
    if eval_end_s is not None:
        happened.append(f"Individuelles Bewertungsfenster: +0.0s bis +{eval_end_s:.1f}s.")
    if coaching_end_s is not None:
        happened.append(
            f"Coaching-Detail bis +{coaching_end_s:.1f}s (normierte Segmente: Exit, Aufbau, Hauptaufbau, Hot-Zone)."
        )
    if personal_profile.get("is_personalized") and capability_text:
        happened.append(f"Personalisierter Modus: {capability_text}")
    if personal_profile.get("is_personalized") and performance_text:
        happened.append(f"Leistungsprofil: {performance_text}")
    phase_boosts = _extract_phase_boosts(tip_effect_profile)
    effect_line = _extract_effect_line(tip_effect_profile)
    if effect_line:
        happened.append(effect_line)
    if not window_supports_20 and fp20 is not None:
        happened.append(
            "Hinweis: +20s liegt außerhalb des ausgewerteten Hauptbereichs; Aussagen bis +20s sind hier nur eingeschränkt belastbar."
        )
    if start_vvert is not None and vvert_20 is not None and gain_10_20 is not None:
        happened.append(
            f"Speed-Aufbau: bei +10s {start_vvert:.1f} km/h, bei +20s {vvert_20:.1f} km/h "
            f"(Zuwachs +{gain_10_20:.1f})."
        )
    if early_acc_mean is not None and early_acc_peak is not None:
        happened.append(
            f"Anfangsbeschleunigung (0-6s): im Mittel {early_acc_mean:.2f} m/s2, Spitze {early_acc_peak:.2f} m/s2."
        )
    _phase_lines, phase_flags = _build_phase_detail_lines(
        chart_data=chart_data,
        focus_end_s=coaching_end_s,
        eval_end_s=eval_end_s,
    )
    if _phase_lines:
        happened.extend(_phase_lines[:4])
    segment_eff = _segment_efficiency_analysis(
        chart_data=chart_data,
        focus_end_s=coaching_end_s,
        eval_end_s=eval_end_s,
    )
    exit_carry = _exit_acc_carryover_analysis(chart_data=chart_data)
    corridor_tail = _corridor_tail_analysis(
        chart_data=chart_data,
        corridor_end_s=pw_end_num,
    )
    peak_hold = _peak_hold_quality(
        chart_data=chart_data,
        focus_end_s=coaching_end_s,
    )
    forward_track = _forward_track_behavior(
        chart_data=chart_data,
        focus_end_s=forward_focus_end_s,
    )
    lateral_behavior = analyze_lateral_dynamics(
        chart_data,
        eval_end_s=eval_end_s,
    )
    if exit_carry.get("available"):
        happened.append(
            "Exit-Mitnahme: acc 0-2s "
            f"{exit_carry['acc_mean_0_2']:.2f} m/s2, acc 2-6s {exit_carry['acc_mean_2_6']:.2f} m/s2 "
            f"(Quote {exit_carry['carry_ratio']:.2f}), vVert 0-10s +{exit_carry['vvert_gain_0_10_mps']:.1f} m/s."
        )
    if peak_hold.get("available"):
        happened.append(
            "Peak-Haltezeit im Coaching-Fenster: "
            f">390 km/h {peak_hold['dur_above_390_s']:.1f}s, >400 km/h {peak_hold['dur_above_400_s']:.1f}s."
        )
    if corridor_tail.get("available"):
        agl_text = ""
        if corridor_tail.get("hagl_end_m") is not None:
            agl_text = f" (~{corridor_tail['hagl_end_m']:.0f} m AGL)"
        happened.append(
            "Bewertungskorridor endet bei +"
            f"{corridor_tail['corridor_end_s']:.1f}s{agl_text}. "
            f"Letzte {corridor_tail['tail_window_s']:.1f}s: vVert "
            f"{corridor_tail['tail_vvert_start_kmh']:.1f}->{corridor_tail['tail_vvert_end_kmh']:.1f} km/h "
            f"({corridor_tail['tail_vvert_gain_kmh']:+.1f}), Winkel-Korrekturen: {corridor_tail['tail_angle_turns']}."
        )
    if forward_track.get("available"):
        drift_text = ""
        if forward_track.get("drift_start_s") is not None:
            drift_text = f", Rückdrift ab +{forward_track['drift_start_s']:.1f}s"
        happened.append(
            "Fluglinie bis Abbremsbeginn (Vorwärts-Strecke): "
            f"Maximum {forward_track['max_forward_m']:.1f} m, "
            f"Rückdrift {forward_track['max_backtrack_m']:.1f} m "
            f"({forward_track['backtrack_ratio_pct']:.0f}%){drift_text}."
        )
    if lateral_behavior.get("available"):
        hot_start = _num(lateral_behavior.get("hot_start_s"))
        hot_end = _num(lateral_behavior.get("hot_end_s"))
        hot_abs = _num(lateral_behavior.get("hot_vlat_abs_mean_kmh"))
        hot_changes = _num(lateral_behavior.get("hot_sign_changes"))
        hot_direction = str(lateral_behavior.get("hot_direction") or "neutral")
        heading_rms = _num(lateral_behavior.get("hot_heading_rate_rms_dps"))
        if hot_start is not None and hot_end is not None and hot_abs is not None and hot_changes is not None:
            dir_text = "ohne klare Seite"
            if hot_direction == "rechts":
                dir_text = "mit leichter Tendenz nach rechts"
            elif hot_direction == "links":
                dir_text = "mit leichter Tendenz nach links"
            happened.append(
                f"Seitbewegung Hot-Zone (+{hot_start:.1f}s bis +{hot_end:.1f}s): "
                f"|Seitbewegung| im Mittel {hot_abs:.1f} km/h, "
                f"Richtungswechsel {int(round(hot_changes))}, {dir_text}."
            )
            if heading_rms is not None:
                happened.append(
                    f"Seitlinien-Ruhe: Richtungsdrehen in der Hot-Zone {heading_rms:.1f} Grad/s (RMS)."
                )
        lateral_event = lateral_behavior.get("speed_cost_event", {})
        if isinstance(lateral_event, dict) and lateral_event.get("available") and lateral_event.get("likely_speed_cost"):
            t_peak = _num(lateral_event.get("t_peak_s"))
            direction = str(lateral_event.get("direction") or "zur Seite")
            drop_a = _num(lateral_event.get("avert_drop_mps2"))
            drop_v = _num(lateral_event.get("vvert_drop_kmh"))
            event_text = "Eine seitliche Korrektur in der Hot-Zone kostet wahrscheinlich Speed."
            if t_peak is not None and drop_a is not None and drop_v is not None:
                event_text = (
                    f"Um +{t_peak:.1f}s gab es eine deutliche seitliche Korrektur nach {direction}. "
                    f"Danach fiel die vertikale Beschleunigung um {drop_a:.2f} m/s2 "
                    f"und vVert um {drop_v:.1f} km/h."
                )
            not_good.append(event_text)
            asym_dir = str(personal_profile.get("asymmetry_direction") or "")
            asym_hint = ""
            if asym_dir and asym_dir == direction:
                asym_hint = " Das passt zu deinem wiederkehrenden Seitenmuster."
            _add_action(
                actions_by_key,
                key="lateral_speed_cost_event",
                score=9,
                text=(
                    "In der Hot-Zone seitliche Korrekturen früher und kleiner setzen, "
                    "damit die vertikale Beschleunigung nach der Korrektur nicht einbricht."
                    + asym_hint
                ),
            )
        else:
            pattern = str(lateral_behavior.get("hot_pattern") or "")
            if pattern == "ruhig":
                good.append("Seitbewegung ist in der Hot-Zone ruhig und kostet aktuell keinen erkennbaren Speed.")
            elif pattern == "gerichtete_drift":
                good.append(
                    "Es gibt eine seitliche Tendenz, aber ohne klaren Hinweis auf direkten Speedverlust im Peak."
                )
            elif pattern == "schlangenlinie":
                not_good.append(
                    "In der Hot-Zone wechselst du seitlich mehrfach die Richtung. Das spricht für Nachkorrekturen."
                )
                _add_action(
                    actions_by_key,
                    key="lateral_snake_pattern",
                    score=8,
                    text=(
                        "Seitlinie in der Hot-Zone beruhigen: Schulter und Hüfte parallel halten "
                        "und Korrekturen früher, aber kleiner setzen."
                    ),
                )

        high_speed = lateral_behavior.get("high_speed", {})
        if isinstance(high_speed, dict) and high_speed.get("available") and high_speed.get("unstable"):
            hs_threshold = _num(high_speed.get("threshold_kmh"))
            hs_vlat = _num(high_speed.get("vlat_abs_mean_kmh"))
            hs_heading = _num(high_speed.get("heading_rate_rms_dps"))
            hs_theta = _num(high_speed.get("theta_std_deg"))
            detail = []
            if hs_vlat is not None:
                detail.append(f"Seitbewegung {hs_vlat:.1f} km/h")
            if hs_heading is not None:
                detail.append(f"Richtungsdrehen {hs_heading:.1f} Grad/s")
            if hs_theta is not None:
                detail.append(f"Winkelschwankung {hs_theta:.1f} Grad")
            detail_text = ", ".join(detail)
            if hs_threshold is not None:
                not_good.append(
                    f"Sobald du schnell wirst (ab etwa {hs_threshold:.0f} km/h), wird die Linie unruhig ({detail_text})."
                )
            else:
                not_good.append("Unter hoher Geschwindigkeit wird die Linie unruhig.")
            _add_action(
                actions_by_key,
                key="high_speed_stability",
                score=8,
                text=(
                    "Die letzte schnelle Phase mit weniger Lenkimpulsen fliegen: "
                    "kleine frühe Korrekturen statt später großer Gegenkorrektur."
                ),
            )
        if personal_profile.get("asymmetry_available"):
            asym_text = str(personal_profile.get("asymmetry_text") or "").strip()
            if asym_text:
                happened.append(asym_text)

    # FS2 quality belongs to data-quality hints in the UI (separate block),
    # not to flight-technique positives/problems.

    early_loss_mps = _num(phase_flags.get("early_vvert_drop_mps"))
    early_start_mps = _num(phase_flags.get("early_vvert_start_mps"))
    early_min_mps = _num(phase_flags.get("early_vvert_min_mps"))
    if early_loss_mps is not None and early_start_mps is not None and early_min_mps is not None and early_loss_mps >= 2.0:
        not_good.append(
            f"In den ersten 2 Sekunden fällt vVert früh von {early_start_mps:.1f} auf {early_min_mps:.1f} m/s."
        )
        early_drop_text = "Den Absprung in den ersten 2 Sekunden stabiler halten und große Anfangskorrekturen vermeiden."
        if capability_mode == "safe":
            early_drop_text = (
                f"{early_drop_text} Erst in kleinen Schritten arbeiten und +10s Richtung "
                f"{_safe_phase_0_10_target(personal_profile)} bringen."
            )
        elif capability_mode == "push":
            early_drop_text = (
                f"{early_drop_text} Wenn stabil, danach zuegig in den oberen Bereich von "
                f"{_safe_phase_0_10_target(personal_profile)} gehen."
            )
        _add_action(
            actions_by_key,
            key="early_vvert_drop",
            score=8,
            text=early_drop_text,
        )
    phase_20_25_min_vhor = _num(phase_flags.get("phase_20_25_min_vhor"))
    phase_20_25_max_angle = _num(phase_flags.get("phase_20_25_max_angle"))
    personal_vhor_floor = _num(personal_profile.get("vhor_min_20_25_floor"))
    if personal_vhor_floor is None:
        personal_vhor_floor = 25.0
    personal_vhor_floor = max(22.0, min(35.0, float(personal_vhor_floor)))
    if phase_20_25_min_vhor is not None and phase_20_25_min_vhor < personal_vhor_floor:
        angle_text = ""
        if phase_20_25_max_angle is not None:
            angle_text = f" bei Winkel bis {phase_20_25_max_angle:.1f} deg"
        if personal_profile.get("is_personalized"):
            not_good.append(
                f"In der Hot-Zone fällt vHor bis {phase_20_25_min_vhor:.1f} km/h{angle_text}. "
                f"Bei dir wird es meist unter etwa {personal_vhor_floor:.1f} km/h unruhig."
            )
        else:
            not_good.append(
                f"In der Hot-Zone fällt vHor bis {phase_20_25_min_vhor:.1f} km/h{angle_text}."
            )
        _add_action(
            actions_by_key,
            key="segment_20_25_vhor_drop",
            score=8,
            text=(
                "In der Hot-Zone die Linie ruhiger halten, "
                f"damit vHor möglichst nicht unter {personal_vhor_floor:.1f} km/h rutscht."
            ),
        )
    if personal_profile.get("is_personalized") and window_supports_20:
        phase_10_15_low = _num(personal_profile.get("phase_10_15_gain_low"))
        phase_10_15_high = _num(personal_profile.get("phase_10_15_gain_high"))
        phase_15_20_low = _num(personal_profile.get("phase_15_20_gain_low"))
        phase_15_20_high = _num(personal_profile.get("phase_15_20_gain_high"))
        if (
            gain_10_15 is not None
            and phase_10_15_low is not None
            and phase_10_15_high is not None
            and phase_10_15_high > phase_10_15_low
            and gain_10_15 < (phase_10_15_low - 2.0)
        ):
            not_good.append(
                "Phase +10 bis +15s: "
                f"Zuwachs +{gain_10_15:.1f} km/h (dein stabiler Korridor: +{phase_10_15_low:.1f} bis +{phase_10_15_high:.1f})."
            )
            mode_hint = ""
            if capability_mode == "safe":
                mode_hint = (
                    " Erst Stabilität sichern: nicht sofort steiler werden, "
                    "sondern die aktuelle Linie ruhig halten."
                )
            elif capability_mode == "push":
                mode_hint = " Wenn stabil, danach die obere Haelfte des Korridors anpeilen."
            _add_action(
                actions_by_key,
                key="phase_10_15_below_corridor",
                score=8,
                text=(
                    "Im Segment +10 bis +15s früher Druck aufbauen und die Linie ruhiger halten, "
                    f"damit der Zuwachs in Richtung +{phase_10_15_low:.0f} km/h geht.{mode_hint}"
                ),
            )
        if (
            gain_15_20 is not None
            and phase_15_20_low is not None
            and phase_15_20_high is not None
            and phase_15_20_high > phase_15_20_low
            and gain_15_20 < (phase_15_20_low - 2.0)
        ):
            not_good.append(
                "Phase +15 bis +20s: "
                f"Zuwachs +{gain_15_20:.1f} km/h (dein stabiler Korridor: +{phase_15_20_low:.1f} bis +{phase_15_20_high:.1f})."
            )
            mode_hint = ""
            if capability_mode == "safe":
                mode_hint = " In dieser Phase lieber klein korrigieren und stabil bleiben."
            elif capability_mode == "push":
                mode_hint = " Wenn die Linie ruhig bleibt, obere Korridorhaelfte nutzen."
            _add_action(
                actions_by_key,
                key="phase_15_20_below_corridor",
                score=8,
                text=(
                    "Im Segment +15 bis +20s Druck gleichmäßig weiterziehen und kleine, frühe Korrekturen setzen, "
                    f"damit der Zuwachs wieder Richtung +{phase_15_20_low:.0f} km/h geht.{mode_hint}"
                ),
            )
    angle_chain = _angle_progression_stability_analysis(
        chart_data=chart_data,
        focus_end_s=coaching_end_s,
        eval_end_s=eval_end_s,
        profile=personal_profile,
    )
    if angle_chain.get("available"):
        angle_10 = _num(angle_chain.get("angle_10"))
        angle_15 = _num(angle_chain.get("angle_15"))
        angle_20 = _num(angle_chain.get("angle_20"))
        angle_peak = _num(angle_chain.get("angle_peak"))
        angle_end = _num(angle_chain.get("angle_end"))
        peak_t = _num(angle_chain.get("t_peak_s"))
        rollback_deg = _num(angle_chain.get("rollback_deg"))
        turns_after_peak = _num(angle_chain.get("turns_after_peak"))
        vvert_drop_after_peak = _num(angle_chain.get("vvert_drop_after_peak_kmh"))
        band_low = _num(personal_profile.get("phase_10_15_angle_low"))
        band_high = _num(personal_profile.get("phase_10_15_angle_high"))
        if band_low is None:
            band_low = _num(personal_profile.get("angle_20_target_low"))
        if band_high is None:
            band_high = _num(personal_profile.get("angle_20_target_high"))
        if band_low is not None and band_high is not None and band_high > band_low:
            angle_band_text = f"{band_low:.1f} bis {band_high:.1f} Grad"
        else:
            angle_band_text = "deinem stabilen Winkelkorridor"

        if angle_chain.get("too_fast_steep_not_hold"):
            if None not in {angle_10, angle_15, angle_20, angle_peak, angle_end, peak_t, rollback_deg}:
                not_good.append(
                    "Du gehst im Aufbau zu schnell steil "
                    f"(+10s {angle_10:.1f} -> +15s {angle_15:.1f} -> +20s {angle_20:.1f} Grad), "
                    f"kannst den Winkel aber nicht halten: ab +{peak_t:.1f}s fällt er wieder auf {angle_end:.1f} Grad "
                    f"(Rücklauf {rollback_deg:.1f} Grad)."
                )
            else:
                not_good.append(
                    "Du gehst im Aufbau zu schnell steil und kannst den Winkel in der schnellen Phase nicht stabil halten."
                )
            if turns_after_peak is not None and vvert_drop_after_peak is not None:
                not_good.append(
                    f"Im Rücklauf folgen {int(round(turns_after_peak))} Nachkorrekturen und vVert fällt dabei um {vvert_drop_after_peak:.1f} km/h."
                )
            _add_action(
                actions_by_key,
                key="phase_10_15_too_steep_not_hold",
                score=9,
                text=(
                    "Im Aufbau nicht zu schnell maximal steil werden: "
                    f"erst stabil im Bereich {angle_band_text} bleiben und dann schrittweise steigern. "
                    "Wenn der Winkel wieder rückläufig wird, 1 bis 2 Grad rausnehmen und die Linie beruhigen."
                ),
            )
        elif angle_chain.get("rollback_with_instability"):
            if None not in {angle_peak, angle_end, rollback_deg, peak_t}:
                not_good.append(
                    f"Der Tauchwinkel wird in der schnellen Phase rückläufig "
                    f"(max {angle_peak:.1f} Grad bei +{peak_t:.1f}s -> {angle_end:.1f} Grad, Rücklauf {rollback_deg:.1f})."
                )
            else:
                not_good.append("Der Tauchwinkel wird in der schnellen Phase wieder rückläufig.")
            if turns_after_peak is not None and turns_after_peak >= 1.5:
                not_good.append(
                    f"Das passiert zusammen mit {int(round(turns_after_peak))} Nachkorrekturen und kostet Stabilität."
                )
            _add_action(
                actions_by_key,
                key="segment_20_25_angle_rollback",
                score=8,
                text=(
                    "Ab dem Winkel-Peak die Linie ruhiger halten: kleine, frühe Korrekturen statt später Gegenbewegung, "
                    "damit der Winkel nicht rückläufig wird."
                ),
            )
    if exit_carry.get("available"):
        carry_ratio = _num(exit_carry.get("carry_ratio"))
        acc_0_2 = _num(exit_carry.get("acc_mean_0_2"))
        acc_2_6 = _num(exit_carry.get("acc_mean_2_6"))
        gain_0_10 = _num(exit_carry.get("vvert_gain_0_10_mps"))
        if (
            carry_ratio is not None
            and acc_0_2 is not None
            and acc_2_6 is not None
            and gain_0_10 is not None
            and carry_ratio < 0.72
            and gain_0_10 < 62.0
        ):
            not_good.append(
                "Exit-Druck wird zu wenig mitgenommen (früher Beschleunigungsabfall bei schwachem 0-10s Aufbau)."
            )
            carry_text = "Nach dem Exit den Druck länger tragen: ab +2s stabil weiter beschleunigen, statt früh nachzulassen."
            if capability_mode == "safe":
                carry_text = (
                    f"{carry_text} Zuerst konstante Linie aufbauen und +10s mindestens "
                    f"{_safe_phase_0_10_target(personal_profile)} erreichen."
                )
            elif capability_mode == "push":
                carry_text = (
                    f"{carry_text} Wenn die Linie ruhig bleibt, +10s in den oberen Bereich von "
                    f"{_safe_phase_0_10_target(personal_profile)} ziehen."
                )
            _add_action(
                actions_by_key,
                key="exit_carryover_low",
                score=8,
                text=carry_text,
            )
        elif (
            carry_ratio is not None
            and acc_0_2 is not None
            and acc_2_6 is not None
            and gain_0_10 is not None
            and carry_ratio >= 0.78
            and gain_0_10 >= 65.0
        ):
            good.append("Exit-Dynamik wird gut mitgenommen (stabiler Übergang von 0-2s auf 2-6s).")
    if corridor_tail.get("available"):
        if corridor_tail.get("early_release_before_corridor_end"):
            not_good.append(
                "Die schnelle Linie wird vor Korridorende verlassen, dadurch bleibt bis zum Korridorende Speed liegen."
            )
            _add_action(
                actions_by_key,
                key="corridor_not_used",
                score=9,
                text="Bis Korridorende länger in der schnellen Linie bleiben und den Ausstieg erst danach setzen.",
            )
        tail_gain = _num(corridor_tail.get("tail_vvert_gain_kmh"))
        tail_turns = _num(corridor_tail.get("tail_angle_turns"))
        if tail_gain is not None and tail_turns is not None and tail_gain < 10.0 and tail_turns >= 3:
            not_good.append(
                "In den letzten 5-8 Sekunden steigt der Speed kaum, waehrend der Winkel mehrfach korrigiert wird."
            )
            _add_action(
                actions_by_key,
                key="tail_no_gain_with_corrections",
                score=8,
                text="Im Schlussteil kleinere, frühere Korrekturen fliegen, damit der Speed bis Korridorende weiter steigt.",
            )
    if segment_eff.get("available"):
        hot_eff = _num(segment_eff.get("hot_eff_kmh_per_100m"))
        mid_eff = _num(segment_eff.get("mid_eff_kmh_per_100m"))
        hot_gain = _num(segment_eff.get("hot_vvert_gain_kmh"))
        hot_drop = _num(segment_eff.get("hot_alt_drop_m"))
        hot_start = _num(segment_eff.get("hot_start_s"))
        hot_end = _num(segment_eff.get("hot_end_s"))
        hot_window_text = "Hot-Zone"
        if hot_start is not None and hot_end is not None:
            hot_window_text = f"Hot-Zone (+{hot_start:.1f}s bis +{hot_end:.1f}s)"
        if segment_eff.get("inefficient_20_25"):
            not_good.append(
                "Die Hot-Zone ist ineffizient: viel Höhenverlust bei zu wenig zusätzlichem Speed."
            )
            efficiency_text = "In der Hot-Zone früher kleine Korrekturen setzen und den Winkel nicht weiter aufdrücken."
            if hot_drop is not None and hot_gain is not None and hot_eff is not None and mid_eff is not None:
                efficiency_text = (
                    f"{hot_window_text}: bei {hot_drop:.0f} m Höhenverlust kommen nur +{hot_gain:.1f} km/h dazu "
                    f"({hot_eff:.1f} km/h je 100 m; in der Aufbau-Mitte {mid_eff:.1f}). "
                    "Fokus: früher klein korrigieren und die Linie ruhiger halten, statt spät nachzudrücken."
                )
            _add_action(
                actions_by_key,
                key="efficiency_20_25_low",
                score=8,
                text=efficiency_text,
            )
        if segment_eff.get("high_vhor_price_20_25"):
            not_good.append(
                "In der Hot-Zone wird zu viel vHor verloren für den erreichten vVert-Zuwachs."
            )
            _add_action(
                actions_by_key,
                key="vhor_price_20_25",
                score=7,
                text="Ab +20s sauberer und ruhiger arbeiten, damit vHor langsamer sinkt und der Speed-Aufbau effizient bleibt.",
            )
    if peak_hold.get("available"):
        if peak_hold.get("short_hold_above_400"):
            not_good.append("Die sehr hohe Speed-Zone über 400 km/h wird zu kurz gehalten.")
            _add_action(
                actions_by_key,
                key="hold_400_short",
                score=7,
                text="Über 400 km/h länger stabil bleiben, statt früh mit größeren Korrekturen auszusteigen.",
            )
        if peak_hold.get("short_hold_above_390"):
            _add_action(
                actions_by_key,
                key="hold_390_short",
                score=6,
                text="Die Phase über 390 km/h verlängern, indem du im Peak-Bereich kleinere und frühere Korrekturen setzt.",
            )
    if forward_track.get("available"):
        if forward_track.get("label") == "negativ":
            drift_at = forward_track.get("drift_start_s")
            start_text = "" if drift_at is None else f" ab +{drift_at:.1f}s"
            not_good.append(
                "Die Fluglinie driftet im späten Verlauf klar nach hinten"
                f"{start_text}. Das spricht für Kippen/zu harte Nachkorrekturen."
            )
            _add_action(
                actions_by_key,
                key="forward_track_drift_hard",
                score=9,
                text="Ab etwa +20s Linie ruhiger halten und kleinere, frühere Korrekturen setzen, damit die Fluglinie vorwärts bleibt.",
            )
        elif forward_track.get("label") == "leicht_negativ":
            not_good.append(
                "Im Schlussteil geht die Fluglinie teilweise wieder zurück. Die Linie bleibt nicht durchgehend vorwärts."
            )
            _add_action(
                actions_by_key,
                key="forward_track_drift_soft",
                score=7,
                text="Im Schlussteil Druck gleichmäßiger halten, damit keine Rückdrift in der Fluglinie entsteht.",
            )
        else:
            good.append("Die Fluglinie bleibt im relevanten Bereich vorwärtsgerichtet (kein relevanter Rückdrift).")

    smooth_start = max(6.0, 0.0 if curve_start_num is None else curve_start_num)
    smooth_end_candidates: list[float] = []
    if curve_end_num is not None:
        smooth_end_candidates.append(curve_end_num)
    if best_3s_end_num is not None:
        smooth_end_candidates.append(best_3s_end_num + 8.0)
    if canopy_open_num is not None:
        smooth_end_candidates.append(canopy_open_num - 2.0)
    smooth_end = min(smooth_end_candidates) if smooth_end_candidates else curve_end_num
    if smooth_end is not None and smooth_end <= smooth_start + 1.5:
        smooth_end = curve_end_num
    if smooth_end is not None and coaching_end_s is not None:
        smooth_end = min(smooth_end, coaching_end_s)

    curve_smoothness = _curve_smoothness_analysis(
        chart_data=chart_data,
        start_s=smooth_start,
        end_s=smooth_end,
        angle_10=None if fp10 is None else _num(fp10.get("angle_deg")),
        angle_20=angle_20_raw,
    )
    if curve_smoothness["label"] == "sauber":
        happened.append(
            "Kurvenruhe im Grundverlauf: sauber (wenige Richtungswechsel in Winkel und waagerechter Geschwindigkeit)."
        )
    elif curve_smoothness["label"] == "leicht_unruhig":
        happened.append(
            "Kurvenruhe: leicht unruhig (einige zusätzliche Korrekturen im Verlauf)."
        )
    elif curve_smoothness["label"] == "unruhig":
        happened.append(
            "Kurvenruhe: unruhig (viele Richtungswechsel, Hinweis auf wiederholte Nachkorrekturen)."
        )
    if curve_smoothness["label"] == "sauber" and scorecard.get("hot_zone") == "kritisch":
        happened.append("Der Grundverlauf ist ruhig, aber in der Peak-Phase gibt es einen klaren Stabilitätseinbruch.")

    # Personal +10s guardrail: for "safe" mode, avoid entering too steep too early.
    angle10_now = _num(fp10.get("angle_deg")) if fp10 else None
    a10_low, a10_high, a10_text = _safe_phase_0_10_angle_target(personal_profile)
    early_vvert_drop = _num(phase_flags.get("early_vvert_drop_mps"))
    instability_hits = 0
    if scorecard.get("hot_zone") == "kritisch":
        instability_hits += 1
    if scorecard.get("kipp_risiko") in {"mittel", "hoch"}:
        instability_hits += 1
    if curve_smoothness["label"] in {"leicht_unruhig", "unruhig"}:
        instability_hits += 1
    if bool(angle_chain.get("rollback_with_instability")) or bool(angle_chain.get("too_fast_steep_not_hold")):
        instability_hits += 1
    lateral_event_probe = lateral_behavior.get("speed_cost_event", {}) if isinstance(lateral_behavior, dict) else {}
    if isinstance(lateral_event_probe, dict) and lateral_event_probe.get("available") and lateral_event_probe.get("likely_speed_cost"):
        instability_hits += 1
    lateral_high_probe = lateral_behavior.get("high_speed", {}) if isinstance(lateral_behavior, dict) else {}
    if isinstance(lateral_high_probe, dict) and bool(lateral_high_probe.get("unstable")):
        instability_hits += 1
    if exit_unsteady:
        instability_hits += 1
    if early_vvert_drop is not None and early_vvert_drop >= 2.8:
        instability_hits += 1

    if (
        capability_mode == "safe"
        and angle10_now is not None
        and a10_high is not None
        and a10_text is not None
        and angle10_now >= (a10_high + 0.8)
        and instability_hits >= 2
    ):
        detail = ""
        if early_vvert_drop is not None:
            detail = f" In den ersten 2s fällt vVert dabei um {early_vvert_drop:.1f} m/s."
        not_good.append(
            "Bei +10s ist der Tauchwinkel für dein aktuelles stabiles Niveau zu steil "
            f"(Ist: {angle10_now:.1f} Grad, Zielbereich: {a10_text}). "
            "Danach wird die Linie unruhig und es folgen Nachkorrekturen."
            f"{detail}"
        )
        _add_action(
            actions_by_key,
            key="exit_angle_10_safe_limit",
            score=10,
            text=(
                f"Bis +10s zuerst stabil im Bereich {a10_text} bleiben. "
                "Erst wenn die Linie bis +15s ruhig bleibt, schrittweise weiter steiler werden."
            ),
        )

    if scorecard.get("exit") == "sauber":
        good.append("Der Start in den Sprung wirkt sauber und kontrolliert.")
    elif scorecard.get("exit") == "aggressiv":
        not_good.append("Der Start war sehr hart und unruhig (hohe Anfangsdynamik).")
        _add_action(
            actions_by_key,
            key="exit_too_hard",
            score=7,
            text="Die ersten Sekunden etwas weicher aufbauen und den Druck gleichmäßiger verteilen.",
        )
    if scorecard.get("exit_dynamik") == "dynamisch_stabil":
        good.append("Der Start ist dynamisch, aber dabei stabil und kontrolliert.")
    if start_vvert is not None and 230 <= start_vvert <= 300:
        good.append(f"Die Anfangsgeschwindigkeit wird gut mitgenommen ({start_vvert:.1f} km/h bei +10s).")
    if window_supports_20 and gain_10_20 is not None and 90 <= gain_10_20 <= 200:
        good.append(f"Zwischen +10s und +20s steigt der Speed sauber an (+{gain_10_20:.1f} km/h).")
    if window_supports_20 and scorecard.get("phase_10_20") == "optimal":
        good.append("Zwischen +10s und +20s passt der Winkel gut zum Speed-Aufbau.")
        target_angle_low = _num(personal_profile.get("angle_20_target_low"))
        target_angle_high = _num(personal_profile.get("angle_20_target_high"))
        if (
            personal_profile.get("is_personalized")
            and angle_20_raw is not None
            and target_angle_low is not None
            and target_angle_high is not None
            and target_angle_high > target_angle_low
        ):
            if angle_20_raw < (target_angle_low - 0.6):
                not_good.append(
                    "Im persönlichen Vergleich ist der Winkel bei +20s noch zu flach "
                    f"(Ist: {angle_20_raw:.1f} deg, Korridor: {target_angle_low:.1f} bis {target_angle_high:.1f})."
                )
                _add_action(
                    actions_by_key,
                    key="angle_20_personal_flat",
                    score=7,
                    text=(
                        "Im Aufbau bis +20s den Winkel etwas früher anheben und in deinen stabilen Korridor bringen: "
                        f"{target_angle_low:.1f} bis {target_angle_high:.1f} Grad."
                    ),
                )
            elif angle_20_raw > (target_angle_high + 0.6):
                not_good.append(
                    "Im persönlichen Vergleich ist der Winkel bei +20s schon zu steil "
                    f"(Ist: {angle_20_raw:.1f} deg, Korridor: {target_angle_low:.1f} bis {target_angle_high:.1f})."
                )
                _add_action(
                    actions_by_key,
                    key="angle_20_personal_steep",
                    score=7,
                    text=(
                        "Im Aufbau bis +20s etwas flacher bleiben und den Winkel stabil im eigenen Korridor halten: "
                        f"{target_angle_low:.1f} bis {target_angle_high:.1f} Grad."
                    ),
                )
    if scorecard.get("hot_zone") in {"stabil", "sehr gut"}:
        good.append("In der schnellen Phase bleibt der Verlauf über weite Strecken ruhig.")
    if metrics.get("best_3s_vHor_kmh") is not None and float(metrics["best_3s_vHor_kmh"]) >= 30:
        good.append("Im besten Fenster ist noch genug waagerechte Geschwindigkeit vorhanden.")
    if curve_smoothness["label"] == "sauber":
        good.append("Der Kurvenverlauf ist ruhig und gleichmäßig.")

    if window_supports_20 and scorecard.get("phase_10_20") == "zu flach":
        target_angle_low = _num(personal_profile.get("angle_20_target_low"))
        target_angle_high = _num(personal_profile.get("angle_20_target_high"))
        if personal_profile.get("is_personalized"):
            if (
                target_angle_low is not None
                and target_angle_high is not None
                and angle_20_raw is not None
                and angle_20_raw < (target_angle_low - 0.6)
            ):
                not_good.append(
                    "Zwischen +10s und +20s ist der Winkel für dein stabiles Niveau oft zu flach "
                    f"(Ist +20s: {angle_20_raw:.1f} deg, Korridor: {target_angle_low:.1f} bis {target_angle_high:.1f})."
                )
            elif target_angle_low is None or target_angle_high is None or angle_20_raw is None:
                not_good.append("Zwischen +10s und +20s ist der Winkel für dein stabiles Niveau oft zu flach.")
        else:
            not_good.append("Zwischen +10s und +20s ist der Winkel oft zu flach.")
        should_add_angle_steeper = True
        if (
            personal_profile.get("is_personalized")
            and target_angle_low is not None
            and target_angle_high is not None
            and angle_20_raw is not None
            and angle_20_raw >= (target_angle_low - 0.6)
        ):
            should_add_angle_steeper = False
        if should_add_angle_steeper:
            target_text = _angle_target_for_flat(angle_20_raw, profile=personal_profile)
            _add_action(
                actions_by_key,
                key="angle_steeper",
                score=8,
                text=("Nicht direkt maximal steil gehen: " f"{target_text}"),
            )
    elif window_supports_20 and scorecard.get("phase_10_20") == "zu steil":
        target_angle_low = _num(personal_profile.get("angle_20_target_low"))
        target_angle_high = _num(personal_profile.get("angle_20_target_high"))
        if personal_profile.get("is_personalized"):
            if target_angle_low is not None and target_angle_high is not None and angle_20_raw is not None:
                not_good.append(
                    "Zwischen +10s und +20s ist der Winkel für dein stabiles Niveau oft zu steil "
                    f"(Ist +20s: {angle_20_raw:.1f} deg, Korridor: {target_angle_low:.1f} bis {target_angle_high:.1f})."
                )
            else:
                not_good.append("Zwischen +10s und +20s ist der Winkel für dein stabiles Niveau oft zu steil.")
        else:
            not_good.append("Zwischen +10s und +20s ist der Winkel oft zu steil.")
        target_text = _angle_target_for_steep(angle_20_raw, profile=personal_profile)
        _add_action(
            actions_by_key,
            key="angle_flatter",
            score=7,
            text=target_text,
        )

    if scorecard.get("hot_zone") == "kritisch":
        not_good.append("In der schnellen Phase bricht die waagerechte Geschwindigkeit zu stark ein.")
        peak_text = "In der Peak-Phase Druck ruhiger halten und kleine, frühe Korrekturen machen."
        if personal_profile.get("is_personalized"):
            peak_text = (
                f"{peak_text} Ziel für dich: vHor in der Hot-Zone möglichst über "
                f"{personal_vhor_floor:.1f} km/h halten."
            )
        peak_chain_text = _build_personal_hot_zone_chain(
            profile=personal_profile,
            capability_mode=capability_mode,
            angle_20=angle_20_raw,
            min_vhor_20_25=phase_20_25_min_vhor,
            tail_turns=_num(corridor_tail.get("tail_angle_turns")) if corridor_tail.get("available") else None,
        )
        if peak_chain_text:
            not_good.append(peak_chain_text)
        _add_action(
            actions_by_key,
            key="peak_stability",
            score=8,
            text=peak_text,
        )

    if scorecard.get("kipp_risiko") == "hoch":
        not_good.append("Die Bewegung zeigt Abschnitte mit erhoehtem Kipp-Risiko.")
        kipp_text = "Bei Instabilität Körperspannung früher stabilisieren (Schulter und Hüfte)."
        risk_angle = _num(personal_profile.get("angle_20_risk_above"))
        max_turns = _num(personal_profile.get("angle_turns_20_25_max"))
        if personal_profile.get("is_personalized"):
            detail_parts: list[str] = []
            if risk_angle is not None:
                detail_parts.append(f"ab etwa {risk_angle:.1f} Grad")
            if max_turns is not None:
                detail_parts.append(f"bei mehr als {int(round(max_turns))} Korrekturen")
            if detail_parts:
                kipp_text = f"{kipp_text} Bei dir wird es oft unruhig {' und '.join(detail_parts)}."
        _add_action(
            actions_by_key,
            key="kipp_risk_high",
            score=8,
            text=kipp_text,
        )
    elif scorecard.get("kipp_risiko") == "mittel":
        not_good.append("Es gibt kurze Abschnitte mit mittlerem Stabilitätsrisiko.")
        medium_text = "In schnellen Abschnitten früher kleine Korrekturen setzen, statt spät grob zu korrigieren."
        risk_angle = _num(personal_profile.get("angle_20_risk_above"))
        if personal_profile.get("is_personalized") and risk_angle is not None:
            medium_text = (
                f"{medium_text} Bei dir lohnt es sich, den Winkel ab etwa {risk_angle:.1f} Grad "
                "nur in sehr kleinen Schritten zu veraendern."
            )
        _add_action(
            actions_by_key,
            key="kipp_risk_medium",
            score=6,
            text=medium_text,
        )

    if (
        coaching_end_s is not None
        and coaching_end_s >= 27.5
        and fp28
        and fp28.get("vHor_kmh") is not None
        and float(fp28["vHor_kmh"]) < 25
    ):
        not_good.append("Ab ca. +28s ist die waagerechte Geschwindigkeit sehr niedrig.")
        _add_action(
            actions_by_key,
            key="vhor_tail",
            score=7,
            text="Ab +22s leicht gegensteuern, Ziel: vHor möglichst über 25 km/h halten.",
        )
    elif vhor_24 is not None and vhor_24 < 28:
        not_good.append("Schon um +24s fällt die waagerechte Geschwindigkeit früh ab.")
        _add_action(
            actions_by_key,
            key="vhor_mid",
            score=6,
            text="Bereits ab +20s ruhiger und gleichmäßiger Druck halten, damit vHor später abfällt.",
        )

    if start_vvert is not None and start_vvert < 230:
        not_good.append(f"Bei +10s ist der vertikale Speed zu niedrig ({start_vvert:.1f} km/h).")
        target_low_10 = _num(personal_profile.get("vvert_10_target_low"))
        target_high_10 = _num(personal_profile.get("vvert_10_target_high"))
        if (
            target_low_10 is not None
            and target_high_10 is not None
            and target_high_10 - target_low_10 >= 8.0
        ):
            if capability_mode == "safe":
                safe_hi = target_low_10 + (target_high_10 - target_low_10) * 0.6
                target_text_10 = (
                    f"Ziel bei +10s zuerst stabil {target_low_10:.0f} bis {safe_hi:.0f} km/h "
                    f"(dein Korridor: {target_low_10:.0f} bis {target_high_10:.0f})."
                )
            elif capability_mode == "push":
                push_lo = target_low_10 + (target_high_10 - target_low_10) * 0.5
                target_text_10 = (
                    f"Ziel bei +10s: {target_low_10:.0f} bis {target_high_10:.0f} km/h, "
                    f"bei stabiler Linie obere Haelfte ({push_lo:.0f}+)."
                )
            else:
                target_text_10 = (
                    f"Ziel bei +10s: {target_low_10:.0f} bis {target_high_10:.0f} km/h "
                    f"(dein stabiler Korridor)."
                )
        else:
            target_text_10 = "Ziel bei +10s: 230 bis 260 km/h."
        _add_action(
            actions_by_key,
            key="start_speed_low",
            score=10 if start_vvert < 210 else 8,
            text=f"Startphase früher in eine stabile, entschlossene Linie bringen. {target_text_10}",
        )
    if window_supports_20 and gain_10_20 is not None and gain_10_20 < 90:
        target_gain_low = _num(personal_profile.get("gain_10_20_target_low"))
        target_gain_high = _num(personal_profile.get("gain_10_20_target_high"))
        if (
            personal_profile.get("is_personalized")
            and target_gain_low is not None
            and target_gain_high is not None
            and target_gain_high > target_gain_low
        ):
            not_good.append(
                "Der Speed-Aufbau zwischen +10s und +20s ist zu schwach "
                f"(Ist: +{gain_10_20:.1f} km/h, Korridor: +{target_gain_low:.1f} bis +{target_gain_high:.1f})."
            )
        else:
            not_good.append(f"Der Speed-Aufbau zwischen +10s und +20s ist zu schwach (+{gain_10_20:.1f} km/h).")
        target_gain = _num(personal_profile.get("gain_10_20_target_low"))
        if target_gain_low is not None and target_gain_high is not None and target_gain_high > target_gain_low:
            if capability_mode == "safe":
                target_gain = target_gain_low
            elif capability_mode == "push":
                target_gain = target_gain_low + (target_gain_high - target_gain_low) * 0.7
            else:
                target_gain = target_gain_low + (target_gain_high - target_gain_low) * 0.4
        if target_gain is None:
            target_gain = 90.0
        target_gain = max(90.0, min(180.0, float(target_gain)))
        _add_action(
            actions_by_key,
            key="build_speed_low",
            score=9,
            text=(
                "Zwischen +10s und +20s mehr Druck aufbauen. "
                f"Ziel: in diesem Abschnitt mindestens +{target_gain:.0f} km/h Zuwachs."
            ),
        )
    elif (
        window_supports_20
        and gain_10_20 is not None
        and gain_10_20 > 200
        and scorecard.get("kipp_risiko") in {"mittel", "hoch"}
    ):
        not_good.append(f"Der Aufbau zwischen +10s und +20s ist sehr hart (+{gain_10_20:.1f} km/h).")
        _add_action(
            actions_by_key,
            key="build_too_hard",
            score=6,
            text="Den Speed-Aufbau etwas gleichmäßiger verteilen, damit der Peak stabiler bleibt.",
        )

    if early_acc_mean is not None and early_acc_mean < 1.8:
        not_good.append(f"Die Anfangsbeschleunigung ist eher niedrig ({early_acc_mean:.2f} m/s2 im Mittel).")
        early_low_text = "In den ersten 4 bis 6 Sekunden früher Druck aufbauen, damit der vertikale Speed schneller steigt."
        if capability_mode == "safe":
            early_low_text = (
                f"{early_low_text} Erst sauber in den unteren Zielbereich von "
                f"{_safe_phase_0_10_target(personal_profile)}."
            )
        elif capability_mode == "push":
            early_low_text = (
                f"{early_low_text} Wenn stabil, +10s in die obere Haelfte von "
                f"{_safe_phase_0_10_target(personal_profile)} bringen."
            )
        _add_action(
            actions_by_key,
            key="early_acc_low",
            score=8,
            text=early_low_text,
        )
    elif (
        early_acc_p95 is not None
        and (
            (exit_unsteady and early_acc_p95 > 10.0)
            or (early_acc_p95 > 14.0 and scorecard.get("kipp_risiko") in {"mittel", "hoch"})
        )
    ):
        not_good.append("Der frühe Druckanstieg ist eher hart und kann später Stabilität kosten.")
        _add_action(
            actions_by_key,
            key="early_acc_spike",
            score=6,
            text="Frühen Druckanstieg etwas weicher fahren, damit die Linie stabiler bleibt.",
        )
    if early_acc_peak is not None and early_acc_peak > 20.0 and {"SPEED_SPIKE", "TIME_GAPS"} & quality_flags:
        not_good.append("Die Anfangsbeschleunigung wirkt unplausibel hoch; wegen Datenlücken/Spikes vorsichtig interpretieren.")

    if curve_smoothness["label"] == "unruhig":
        not_good.append("Der Kurvenverlauf springt mehrfach hin und her (viele Nachkorrekturen).")
        for cause in curve_smoothness.get("causes", [])[:2]:
            not_good.append(cause)
        for action in curve_smoothness.get("actions", [])[:3]:
            _add_action(
                actions_by_key,
                key=str(action.get("key")),
                score=int(action.get("score", 6)),
                text=str(action.get("text")),
            )
    elif curve_smoothness["label"] == "leicht_unruhig":
        not_good.append("Der Kurvenverlauf ist nicht durchgehend ruhig (einige Nachkorrekturen sichtbar).")
        for action in curve_smoothness.get("actions", [])[:2]:
            _add_action(
                actions_by_key,
                key=str(action.get("key")),
                score=int(action.get("score", 5)),
                text=str(action.get("text")),
            )

    # TIME_GAPS is represented in data-quality scoring; do not duplicate it in coaching text.
    if "LOW_GPS_FIX" in quality_flags or "HIGH_SPEED_ACCURACY_ERROR" in quality_flags:
        not_good.append("Die GPS-Qualität war nicht durchgehend stabil.")

    if best_compare is not None:
        reference = best_compare.get("reference", {})
        comparison = best_compare.get("comparison", {})
        current_is_reference = str(reference.get("jump_id")) == str(jump.get("jump_id"))
        current_is_comparison = str(comparison.get("jump_id")) == str(jump.get("jump_id"))

        by_label = {row["label"]: row for row in best_compare.get("summary", [])}
        max_row = by_label.get("3s Max (Training)")
        if max_row and max_row.get("delta") is not None:
            delta = float(max_row["delta"])
            if current_is_reference and delta < -0.2:
                good.append(f"Dieser Sprung liegt {abs(delta):.2f} km/h über dem nächstbesten Sprung.")

        fp20_cmp = next(
            (row for row in best_compare.get("fixpoint_rows", []) if abs(float(row.get("t_rel_s", -1)) - 20.0) < 1e-6),
            None,
        )
        if fp20_cmp and fp20_cmp.get("delta_vHor_kmh") is not None:
            d_vhor = float(fp20_cmp["delta_vHor_kmh"])
            if current_is_comparison and d_vhor < -4:
                _add_action(
                    actions_by_key,
                    key="vhor_gap_to_reference",
                    score=7,
                    text=(
                        f"Bei +20s fehlen etwa {abs(d_vhor):.1f} km/h vHor gegenüber dem schnelleren Referenzsprung."
                    ),
                )
            if current_is_comparison and d_vhor > 4:
                good.append(f"Bei +20s ist vHor in diesem Sprung um {d_vhor:.1f} km/h besser als bei der Referenz.")
        fp20_vvert_cmp = next(
            (row for row in best_compare.get("fixpoint_rows", []) if abs(float(row.get("t_rel_s", -1)) - 20.0) < 1e-6),
            None,
        )
        if fp20_vvert_cmp and fp20_vvert_cmp.get("delta_vVert_kmh") is not None:
            d_vvert = float(fp20_vvert_cmp["delta_vVert_kmh"])
            if current_is_comparison and d_vvert < -12:
                _add_action(
                    actions_by_key,
                    key="vvert_gap_20",
                    score=8,
                    text=(
                        f"Bei +20s fehlen gegen die Referenz {abs(d_vvert):.1f} km/h vVert. "
                        "Der größte Hebel liegt im Aufbau bis +20s."
                    ),
                )
            if current_is_comparison and d_vvert > 10:
                good.append(f"Bei +20s ist vVert in diesem Sprung um {d_vvert:.1f} km/h besser als bei der Referenz.")

        charts = best_compare.get("charts", {})
        left_curve = charts.get("left", {}) if isinstance(charts, dict) else {}
        right_curve = charts.get("right", {}) if isinstance(charts, dict) else {}
        compare_window_end = 25.0 if coaching_end_s is None else float(coaching_end_s)
        if current_is_comparison:
            cur_d400 = _duration_above_threshold(
                time_s=right_curve.get("time_s", []),
                values=right_curve.get("vVert_kmh", []),
                threshold=400.0,
                start_s=0.0,
                end_s=compare_window_end,
            )
            ref_d400 = _duration_above_threshold(
                time_s=left_curve.get("time_s", []),
                values=left_curve.get("vVert_kmh", []),
                threshold=400.0,
                start_s=0.0,
                end_s=compare_window_end,
            )
            if ref_d400 is not None and cur_d400 is not None and ref_d400 - cur_d400 >= 1.5:
                not_good.append(
                    f"Die Referenz hält >400 km/h im Coaching-Fenster rund {ref_d400 - cur_d400:.1f}s länger."
                )
                _add_action(
                    actions_by_key,
                    key="hold_400_gap_to_reference",
                    score=7,
                    text="Die Peak-Phase länger stabilisieren, um die >400 km/h Zone näher an die Referenz heranzubringen.",
                )

    reference_compares: list[dict[str, Any]] = []
    if top_reference_compares:
        reference_compares = [item for item in top_reference_compares if isinstance(item, dict)]
    elif external_compare is not None:
        # Backward compatibility: if no top-5 set is provided, keep single external reference behavior.
        reference_compares = [external_compare]

    if reference_compares:
        current_jump_id = str(jump.get("jump_id"))
        gap_values: list[float] = []
        d10_values: list[float] = []
        d15_values: list[float] = []
        hold_gap_values: list[float] = []
        ref_speed_values: list[float] = []
        used_ref_names: list[str] = []

        for ref_compare in reference_compares:
            ext_reference = ref_compare.get("reference", {})
            ext_comparison = ref_compare.get("comparison", {})
            if str(ext_comparison.get("jump_id")) != current_jump_id:
                continue

            ref_speed = _num(ext_reference.get("best_3s_vVert_kmh"))
            if ref_speed is not None:
                ref_speed_values.append(ref_speed)
            ref_name = str(ext_reference.get("file_name") or "").strip()
            if ref_name:
                used_ref_names.append(ref_name)

            by_label = {row["label"]: row for row in ref_compare.get("summary", [])}
            top_row = by_label.get("3s Max (Training)")
            if top_row and top_row.get("delta") is not None and float(top_row["delta"]) < 0.0:
                gap_values.append(abs(float(top_row["delta"])))

            fp10_row = next(
                (row for row in ref_compare.get("fixpoint_rows", []) if abs(float(row.get("t_rel_s", -1)) - 10.0) < 1e-6),
                None,
            )
            fp15_row = next(
                (row for row in ref_compare.get("fixpoint_rows", []) if abs(float(row.get("t_rel_s", -1)) - 15.0) < 1e-6),
                None,
            )
            d10 = None if fp10_row is None else _num(fp10_row.get("delta_vVert_kmh"))
            d15 = None if fp15_row is None else _num(fp15_row.get("delta_vVert_kmh"))
            if d10 is not None:
                d10_values.append(float(d10))
            if d15 is not None:
                d15_values.append(float(d15))

            charts = ref_compare.get("charts", {})
            left_curve = charts.get("left", {}) if isinstance(charts, dict) else {}
            right_curve = charts.get("right", {}) if isinstance(charts, dict) else {}
            compare_window_end = 25.0 if coaching_end_s is None else float(coaching_end_s)
            ext_ref_d390 = _duration_above_threshold(
                time_s=left_curve.get("time_s", []),
                values=left_curve.get("vVert_kmh", []),
                threshold=390.0,
                start_s=0.0,
                end_s=compare_window_end,
            )
            ext_cur_d390 = _duration_above_threshold(
                time_s=right_curve.get("time_s", []),
                values=right_curve.get("vVert_kmh", []),
                threshold=390.0,
                start_s=0.0,
                end_s=compare_window_end,
            )
            if ext_ref_d390 is not None and ext_cur_d390 is not None and ext_ref_d390 > ext_cur_d390:
                hold_gap_values.append(float(ext_ref_d390 - ext_cur_d390))

        if ref_speed_values:
            avg_ref_speed = float(np.mean(ref_speed_values))
            happened.append(
                "Top-5 Benchmark (alle Springer): "
                f"Durchschnitt 3s-Max {avg_ref_speed:.2f} km/h "
                f"(n={len(ref_speed_values)})."
            )

        if gap_values:
            avg_gap = float(np.mean(gap_values))
            if avg_gap <= 2.0:
                good.append(
                    f"Der 3s-Max liegt nahe am Top-5 Niveau (mittlere Abweichung {avg_gap:.2f} km/h)."
                )

        if d10_values and d15_values:
            mean_d10 = float(np.mean(d10_values))
            mean_d15 = float(np.mean(d15_values))
            if mean_d10 < -14.0 and mean_d15 < -20.0:
                not_good.append(
                    "Im Aufbau bis +15s fehlt im Top-5 Vergleich klar vertikaler Speed."
                )
                # Make the tip concrete with measured gap and personal targets when available.
                target_10_low = _num(personal_profile.get("vvert_10_target_low"))
                target_10_high = _num(personal_profile.get("vvert_10_target_high"))
                target_gain_10_15_low = _num(personal_profile.get("phase_10_15_gain_low"))
                gap_text = (
                    f"Bis +15s fehlen im Top-5 Vergleich im Schnitt "
                    f"{abs(mean_d10):.1f} km/h bei +10s und {abs(mean_d15):.1f} km/h bei +15s. "
                    "Fokus: Exit-Druck bis +10s stabil mitnehmen und im Segment +10 bis +15s ohne Gegenkorrektur weiter beschleunigen."
                )
                if target_10_low is not None and target_10_high is not None and target_gain_10_15_low is not None:
                    gap_text = (
                        f"{gap_text} Ziel: +10s stabil in Richtung {target_10_low:.0f} bis {target_10_high:.0f} km/h, "
                        f"dann +10 bis +15s mindestens +{target_gain_10_15_low:.0f} km/h Zuwachs."
                    )

                # Avoid duplicate generic build tips when a stronger build-gap action already exists.
                if "build_speed_low" not in actions_by_key or abs(mean_d15) >= 28.0:
                    _add_action(
                        actions_by_key,
                        key="top5_build_0_15",
                        score=8,
                        text=gap_text,
                    )

        if hold_gap_values:
            mean_hold_gap = float(np.mean(hold_gap_values))
            if mean_hold_gap >= 2.0:
                not_good.append(
                    f"Top-5 Referenzen halten >390 km/h im Schnitt rund {mean_hold_gap:.1f}s länger."
                )
                _add_action(
                    actions_by_key,
                    key="top5_hold_390_gap",
                    score=7,
                    text="Hohe Speed-Bereiche länger stabil halten (über 390 km/h), statt sie früh durch Korrekturen zu verlassen.",
                )

    if not good:
        good.append("Der Sprung ist insgesamt verwertbar und zeigt einen klaren Speed-Verlauf.")
    if not not_good:
        not_good.append("Keine deutlichen Schwachstellen in den Hauptdaten gefunden.")

    improve = _build_priority_actions(actions_by_key, max_items=5, phase_boosts=phase_boosts)
    if not improve:
        improve = ["Priorität 1 (stabil halten): den aktuellen Ablauf möglichst reproduzierbar wiederholen."]

    happened_clean = _compact_lines(happened, max_items=14)
    good_clean = _compact_lines(good, max_items=4)
    not_good_clean = _compact_lines(not_good, max_items=6)

    # Keep output compact and readable.
    return {
        "happened": happened_clean,
        "good": good_clean,
        "not_good": not_good_clean,
        "improve": improve,
    }


def _unique_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _line_topic(text: str) -> str:
    lowered = normalize_german_text(text)
    if "top-5" in lowered:
        return "top5"
    if "referenz" in lowered:
        return "reference"
    if "zeitlücken" in lowered or "gps" in lowered or "daten" in lowered:
        return "data_quality"
    if "kipp" in lowered or "instabil" in lowered or "kurvenverlauf" in lowered:
        return "stability"
    if (
        "+20 bis +25" in lowered
        or "hot-phase" in lowered
        or "hot-zone" in lowered
        or ("+20s" in lowered and ("vhor" in lowered or "vvert" in lowered))
    ):
        return "late_segment"
    if "korridor" in lowered or "letzten 5-8" in lowered or "peak" in lowered:
        return "peak_tail"
    if "aufbau" in lowered:
        return "build"
    if "exit" in lowered or "anfang" in lowered or "+10s" in lowered:
        return "start"
    return f"misc:{lowered[:36]}"


def _compact_lines(items: list[str], *, max_items: int) -> list[str]:
    if max_items <= 0:
        return []
    unique_items = _unique_keep_order(items)
    if len(unique_items) <= max_items:
        return unique_items

    out: list[str] = []
    seen_topics: set[str] = set()
    for item in unique_items:
        topic = _line_topic(item)
        if topic in seen_topics:
            continue
        seen_topics.add(topic)
        out.append(item)
        if len(out) >= max_items:
            return out

    return unique_items[:max_items]


def _fixpoint_at(fixpoints: list[dict[str, Any]], t_rel_s: float) -> dict[str, Any] | None:
    return next((item for item in fixpoints if abs(float(item.get("t_rel_s", -1)) - t_rel_s) < 1e-6), None)


def _num(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coaching_focus_end(*, curve_end_s: float | None, hot_zone_end_s: float | None) -> float | None:
    candidates: list[float] = []
    if hot_zone_end_s is not None:
        candidates.append(float(hot_zone_end_s))
    if curve_end_s is not None:
        candidates.append(float(curve_end_s))
    if not candidates:
        return 25.0
    end_s = min(candidates)
    end_s = min(end_s, 25.0)
    return None if end_s < 9.5 else end_s


def _forward_eval_end_s(*, decel_start_s: float | None, fallback_end_s: float | None) -> float | None:
    if decel_start_s is not None and decel_start_s >= 8.0:
        return float(decel_start_s)
    return fallback_end_s


def _effective_eval_window_end_s(*, notes: dict[str, Any], chart_data: dict[str, Any]) -> float | None:
    candidates: list[float] = []
    for key in ["decel_start_s", "performance_window_end_s", "canopy_open_s"]:
        value = _num(notes.get(key))
        if value is not None and value >= 8.0:
            candidates.append(float(value))
    curve_end = _num(notes.get("curve_window_end_s"))
    if curve_end is not None and curve_end >= 8.0:
        candidates.append(float(curve_end))

    times = [_num(item) for item in (chart_data.get("time_s", []) or [])]
    max_time = max((float(v) for v in times if v is not None), default=None)
    if max_time is not None and max_time >= 8.0 and not candidates:
        candidates.append(float(max_time))
    if not candidates:
        return None
    end_s = float(min(candidates))
    if max_time is not None:
        end_s = min(end_s, float(max_time))
    return end_s if end_s >= 8.0 else None


def _build_phase_detail_lines(
    *,
    chart_data: dict[str, Any],
    focus_end_s: float | None,
    eval_end_s: float | None = None,
) -> tuple[list[str], dict[str, float | None]]:
    lines: list[str] = []
    flags: dict[str, float | None] = {
        "early_vvert_start_mps": None,
        "early_vvert_min_mps": None,
        "early_vvert_drop_mps": None,
        "phase_20_25_min_vhor": None,
        "phase_20_25_max_angle": None,
    }
    if focus_end_s is None:
        return lines, flags

    t = chart_data.get("time_s", []) or []
    vvert_kmh = chart_data.get("vVert_kmh", []) or []
    vhor_kmh = chart_data.get("vHor_kmh", []) or []
    angle_deg = chart_data.get("angle_deg", []) or []
    if not t or len(t) != len(vvert_kmh) or len(t) != len(vhor_kmh) or len(t) != len(angle_deg):
        return lines, flags

    t_arr = np.array([_num(x) for x in t], dtype=float)
    vvert_arr = np.array([_num(x) for x in vvert_kmh], dtype=float)
    vhor_arr = np.array([_num(x) for x in vhor_kmh], dtype=float)
    angle_arr = np.array([_num(x) for x in angle_deg], dtype=float)
    valid = np.isfinite(t_arr) & np.isfinite(vvert_arr) & np.isfinite(vhor_arr) & np.isfinite(angle_arr)
    if valid.sum() < 8:
        return lines, flags
    t_arr = t_arr[valid]
    vvert_arr = vvert_arr[valid]
    vhor_arr = vhor_arr[valid]
    angle_arr = angle_arr[valid]
    if len(t_arr) < 8:
        return lines, flags

    eval_end = float(focus_end_s)
    if eval_end_s is not None:
        eval_end = min(eval_end, float(eval_end_s))
    if eval_end < 8.0:
        return lines, flags

    segments = [
        ("Exit", 0.00, 0.15),
        ("Aufbau", 0.15, 0.45),
        ("Hauptaufbau", 0.45, 0.70),
        ("Hot-Zone", 0.70, 0.90),
    ]
    for label, tau0, tau1 in segments:
        start_s = float(max(0.0, min(eval_end, tau0 * eval_end)))
        end_s = float(max(start_s, min(eval_end, tau1 * eval_end)))
        if end_s <= start_s + 0.4:
            continue
        if t_arr[0] > end_s or t_arr[-1] < start_s:
            continue

        vvert_start = _interp(t_arr, vvert_arr, start_s)
        vvert_end = _interp(t_arr, vvert_arr, end_s)
        vhor_start = _interp(t_arr, vhor_arr, start_s)
        vhor_end = _interp(t_arr, vhor_arr, end_s)
        angle_start = _interp(t_arr, angle_arr, start_s)
        angle_end = _interp(t_arr, angle_arr, end_s)
        if None in {vvert_start, vvert_end, vhor_start, vhor_end, angle_start, angle_end}:
            continue

        mask = (t_arr >= start_s) & (t_arr <= end_s)
        if mask.sum() < 2:
            continue
        min_vhor = float(np.min(vhor_arr[mask]))
        max_angle = float(np.max(angle_arr[mask]))
        delta_vvert_mps = (float(vvert_end) - float(vvert_start)) / 3.6

        lines.append(
            "Phase "
            f"{label} (+{start_s:.1f}s bis +{end_s:.1f}s): "
            f"vVert {vvert_start / 3.6:.1f}->{vvert_end / 3.6:.1f} m/s ({delta_vvert_mps:+.1f}), "
            f"vHor {vhor_start:.1f}->{vhor_end:.1f} km/h, "
            f"Winkel {angle_start:.1f}->{angle_end:.1f} deg, "
            f"vHor-Min {min_vhor:.1f}, Winkel-Max {max_angle:.1f}."
        )
        if label == "Hot-Zone":
            flags["phase_20_25_min_vhor"] = min_vhor
            flags["phase_20_25_max_angle"] = max_angle

    early_end = min(2.0, float(focus_end_s))
    if early_end > 0.8:
        early_mask = (t_arr >= 0.0) & (t_arr <= early_end)
        if early_mask.sum() >= 2:
            early_vvert = vvert_arr[early_mask] / 3.6
            start_mps = float(_interp(t_arr, vvert_arr / 3.6, 0.0) or early_vvert[0])
            min_mps = float(np.min(early_vvert))
            flags["early_vvert_start_mps"] = start_mps
            flags["early_vvert_min_mps"] = min_mps
            flags["early_vvert_drop_mps"] = max(0.0, start_mps - min_mps)

    return lines, flags


def _interp(x: np.ndarray, y: np.ndarray, target: float) -> float | None:
    if len(x) < 2 or target < float(x[0]) or target > float(x[-1]):
        return None
    return float(np.interp(target, x, y))


def _segment_efficiency_analysis(
    *,
    chart_data: dict[str, Any],
    focus_end_s: float | None,
    eval_end_s: float | None = None,
) -> dict[str, Any]:
    if focus_end_s is None:
        return {"available": False}

    t = chart_data.get("time_s", []) or []
    vvert_kmh = chart_data.get("vVert_kmh", []) or []
    vhor_kmh = chart_data.get("vHor_kmh", []) or []
    h_agl = chart_data.get("hAGL_m", []) or []
    if not t or len(t) != len(vvert_kmh) or len(t) != len(vhor_kmh) or len(t) != len(h_agl):
        return {"available": False}

    t_arr = np.array([_num(x) for x in t], dtype=float)
    vvert_arr = np.array([_num(x) for x in vvert_kmh], dtype=float)
    vhor_arr = np.array([_num(x) for x in vhor_kmh], dtype=float)
    h_arr = np.array([_num(x) for x in h_agl], dtype=float)
    valid = np.isfinite(t_arr) & np.isfinite(vvert_arr) & np.isfinite(vhor_arr) & np.isfinite(h_arr)
    if valid.sum() < 12:
        return {"available": False}
    t_arr = t_arr[valid]
    vvert_arr = vvert_arr[valid]
    vhor_arr = vhor_arr[valid]
    h_arr = h_arr[valid]

    eval_end = float(focus_end_s)
    if eval_end_s is not None:
        eval_end = min(eval_end, float(eval_end_s))
    if eval_end < 8.0:
        return {"available": False}

    segments = []
    for label, tau0, tau1 in [
        ("build_early", 0.15, 0.45),
        ("build_late", 0.45, 0.70),
        ("hot", 0.70, 0.90),
    ]:
        s = float(max(0.0, min(eval_end, tau0 * eval_end)))
        e = float(max(s, min(eval_end, tau1 * eval_end)))
        segments.append((label, s, e))
    out_segments: list[dict[str, Any]] = []
    for label, start_s, end_s in segments:
        if end_s <= start_s + 0.4:
            continue
        if start_s < float(t_arr[0]) or end_s > float(t_arr[-1]):
            continue
        v_start = _interp(t_arr, vvert_arr, start_s)
        v_end = _interp(t_arr, vvert_arr, end_s)
        h_start = _interp(t_arr, h_arr, start_s)
        h_end = _interp(t_arr, h_arr, end_s)
        vh_start = _interp(t_arr, vhor_arr, start_s)
        vh_end = _interp(t_arr, vhor_arr, end_s)
        if None in {v_start, v_end, h_start, h_end, vh_start, vh_end}:
            continue
        alt_drop = float(h_start - h_end)
        v_gain = float(v_end - v_start)
        vh_drop = float(vh_start - vh_end)
        gain_per_100m = None if alt_drop <= 20.0 else float(v_gain / (alt_drop / 100.0))
        vhor_price = float(vh_drop / max(v_gain, 1.0))
        out_segments.append(
            {
                "label": label,
                "start_s": start_s,
                "end_s": end_s,
                "vvert_gain_kmh": v_gain,
                "vhor_drop_kmh": vh_drop,
                "alt_drop_m": alt_drop,
                "gain_per_100m_kmh": gain_per_100m,
                "vhor_price": vhor_price,
            }
        )

    if not out_segments:
        return {"available": False}

    by_label = {seg["label"]: seg for seg in out_segments}
    mid_eff_vals = [
        seg.get("gain_per_100m_kmh")
        for key, seg in by_label.items()
        if key in {"build_early", "build_late"} and seg.get("gain_per_100m_kmh") is not None
    ]
    mid_price_vals = [
        seg.get("vhor_price")
        for key, seg in by_label.items()
        if key in {"build_early", "build_late"} and seg.get("vhor_price") is not None
    ]
    eff_20_25 = by_label.get("hot", {}).get("gain_per_100m_kmh")
    price_20_25 = by_label.get("hot", {}).get("vhor_price")
    mid_eff = float(np.mean(mid_eff_vals)) if mid_eff_vals else None
    mid_price = float(np.mean(mid_price_vals)) if mid_price_vals else None

    lines: list[str] = []
    if eff_20_25 is not None and mid_eff is not None:
        hot_seg = by_label.get("hot", {})
        hot_start = _num(hot_seg.get("start_s"))
        hot_end = _num(hot_seg.get("end_s"))
        hot_label = "Hot-Zone"
        if hot_start is not None and hot_end is not None:
            hot_label = f"Hot-Zone (+{hot_start:.1f}s bis +{hot_end:.1f}s)"
        lines.append(
            f"Segment-Effizienz {hot_label}: "
            f"{eff_20_25:.1f} km/h je 100 m "
            f"(Aufbau-Mitte: {mid_eff:.1f})."
        )

    inefficient_20_25 = bool(
        eff_20_25 is not None
        and mid_eff is not None
        and mid_eff > 0.0
        and eff_20_25 < (mid_eff * 0.65)
        and by_label.get("hot", {}).get("alt_drop_m", 0.0) >= 250.0
    )
    high_vhor_price_20_25 = bool(
        price_20_25 is not None
        and mid_price is not None
        and price_20_25 > max(0.9, mid_price * 1.4)
    )

    return {
        "available": True,
        "segments": out_segments,
        "lines": lines,
        "inefficient_20_25": inefficient_20_25,
        "high_vhor_price_20_25": high_vhor_price_20_25,
        "mid_eff_kmh_per_100m": mid_eff,
        "mid_vhor_price": mid_price,
        "hot_eff_kmh_per_100m": eff_20_25,
        "hot_vhor_price": price_20_25,
        "hot_vvert_gain_kmh": by_label.get("hot", {}).get("vvert_gain_kmh"),
        "hot_vhor_drop_kmh": by_label.get("hot", {}).get("vhor_drop_kmh"),
        "hot_alt_drop_m": by_label.get("hot", {}).get("alt_drop_m"),
        "hot_start_s": by_label.get("hot", {}).get("start_s"),
        "hot_end_s": by_label.get("hot", {}).get("end_s"),
    }


def _peak_hold_quality(
    *,
    chart_data: dict[str, Any],
    focus_end_s: float | None,
) -> dict[str, Any]:
    if focus_end_s is None:
        return {"available": False}
    dur_390 = _duration_above_threshold(
        time_s=chart_data.get("time_s", []) or [],
        values=chart_data.get("vVert_kmh", []) or [],
        threshold=390.0,
        start_s=0.0,
        end_s=float(focus_end_s),
    )
    dur_400 = _duration_above_threshold(
        time_s=chart_data.get("time_s", []) or [],
        values=chart_data.get("vVert_kmh", []) or [],
        threshold=400.0,
        start_s=0.0,
        end_s=float(focus_end_s),
    )
    if dur_390 is None or dur_400 is None:
        return {"available": False}
    return {
        "available": True,
        "dur_above_390_s": dur_390,
        "dur_above_400_s": dur_400,
        "short_hold_above_390": dur_390 < 3.0,
        "short_hold_above_400": dur_400 < 1.0,
    }


def _duration_above_threshold(
    *,
    time_s: list[Any],
    values: list[Any],
    threshold: float,
    start_s: float,
    end_s: float,
) -> float | None:
    if not time_s or not values or len(time_s) != len(values):
        return None
    t_arr = np.array([_num(x) for x in time_s], dtype=float)
    v_arr = np.array([_num(x) for x in values], dtype=float)
    valid = np.isfinite(t_arr) & np.isfinite(v_arr)
    if valid.sum() < 2:
        return None
    t_arr = t_arr[valid]
    v_arr = v_arr[valid]
    if end_s <= start_s or end_s < float(t_arr[0]) or start_s > float(t_arr[-1]):
        return None
    start = max(start_s, float(t_arr[0]))
    end = min(end_s, float(t_arr[-1]))
    mask = (t_arr >= start) & (t_arr <= end)
    if mask.sum() < 2:
        return None
    t_w = t_arr[mask]
    v_w = v_arr[mask]
    dur = 0.0
    for i in range(1, len(t_w)):
        if v_w[i - 1] >= threshold and v_w[i] >= threshold:
            dur += float(t_w[i] - t_w[i - 1])
    return dur


def _exit_acc_carryover_analysis(*, chart_data: dict[str, Any]) -> dict[str, Any]:
    t = chart_data.get("time_s", []) or []
    acc = chart_data.get("accVert_mps2", []) or []
    vvert_kmh = chart_data.get("vVert_kmh", []) or []
    if not t or len(t) != len(acc) or len(t) != len(vvert_kmh):
        return {"available": False}

    t_arr = np.array([_num(x) for x in t], dtype=float)
    acc_arr = np.array([_num(x) for x in acc], dtype=float)
    vvert_arr = np.array([_num(x) for x in vvert_kmh], dtype=float)
    valid = np.isfinite(t_arr) & np.isfinite(acc_arr) & np.isfinite(vvert_arr)
    if valid.sum() < 10:
        return {"available": False}
    t_arr = t_arr[valid]
    acc_arr = acc_arr[valid]
    vvert_arr = vvert_arr[valid]

    if t_arr[0] > 0.2 or t_arr[-1] < 10.0:
        return {"available": False}

    m0_2 = (t_arr >= 0.0) & (t_arr <= 2.0)
    m2_6 = (t_arr > 2.0) & (t_arr <= 6.0)
    if m0_2.sum() < 3 or m2_6.sum() < 3:
        return {"available": False}

    acc_mean_0_2 = float(np.mean(acc_arr[m0_2]))
    acc_mean_2_6 = float(np.mean(acc_arr[m2_6]))
    if abs(acc_mean_0_2) < 1e-6:
        return {"available": False}
    carry_ratio = float(acc_mean_2_6 / acc_mean_0_2)

    v0 = _interp(t_arr, vvert_arr, 0.0)
    v10 = _interp(t_arr, vvert_arr, 10.0)
    if v0 is None or v10 is None:
        return {"available": False}
    vvert_gain_0_10_mps = float((v10 - v0) / 3.6)

    return {
        "available": True,
        "acc_mean_0_2": acc_mean_0_2,
        "acc_mean_2_6": acc_mean_2_6,
        "carry_ratio": carry_ratio,
        "vvert_gain_0_10_mps": vvert_gain_0_10_mps,
    }


def _corridor_tail_analysis(
    *,
    chart_data: dict[str, Any],
    corridor_end_s: float | None,
) -> dict[str, Any]:
    if corridor_end_s is None:
        return {"available": False}

    t = chart_data.get("time_s", []) or []
    vvert_kmh = chart_data.get("vVert_kmh", []) or []
    angle_deg = chart_data.get("angle_deg", []) or []
    h_agl = chart_data.get("hAGL_m", []) or []
    if not t or len(t) != len(vvert_kmh) or len(t) != len(angle_deg):
        return {"available": False}

    t_arr = np.array([_num(x) for x in t], dtype=float)
    vvert_arr = np.array([_num(x) for x in vvert_kmh], dtype=float)
    angle_arr = np.array([_num(x) for x in angle_deg], dtype=float)
    hagl_arr = np.array([_num(x) for x in h_agl], dtype=float) if len(h_agl) == len(t) else None
    valid = np.isfinite(t_arr) & np.isfinite(vvert_arr) & np.isfinite(angle_arr)
    if valid.sum() < 10:
        return {"available": False}
    t_arr = t_arr[valid]
    vvert_arr = vvert_arr[valid]
    angle_arr = angle_arr[valid]
    if hagl_arr is not None:
        hagl_arr = hagl_arr[valid]

    end_s = float(corridor_end_s)
    if end_s <= float(t_arr[0]) + 5.0 or end_s > float(t_arr[-1]):
        return {"available": False}
    tail_start_s = max(0.0, end_s - 8.0)
    if end_s - tail_start_s < 4.0:
        return {"available": False}

    v_start = _interp(t_arr, vvert_arr, tail_start_s)
    v_end = _interp(t_arr, vvert_arr, end_s)
    if v_start is None or v_end is None:
        return {"available": False}

    tail_mask = (t_arr >= tail_start_s) & (t_arr <= end_s)
    if tail_mask.sum() < 5:
        return {"available": False}
    tail_t = t_arr[tail_mask]
    tail_angle = angle_arr[tail_mask]
    tail_vvert = vvert_arr[tail_mask]
    tail_peak_idx = int(np.argmax(tail_vvert))
    tail_peak_vvert = float(tail_vvert[tail_peak_idx])
    tail_peak_t = float(tail_t[tail_peak_idx])
    drop_peak_to_end = float(tail_peak_vvert - v_end)

    turn_count = 0
    prev = 0
    for d in np.diff(tail_angle):
        sgn = 1 if d > 0.35 else -1 if d < -0.35 else 0
        if sgn == 0:
            continue
        if prev != 0 and sgn != prev:
            turn_count += 1
        prev = sgn

    early_release = bool(
        drop_peak_to_end >= 30.0
        and tail_peak_t <= end_s - 1.0
        and float(_interp(t_arr, angle_arr, end_s) or tail_angle[-1]) <= (float(np.max(tail_angle)) - 4.0)
    )

    hagl_end = None
    if hagl_arr is not None and np.isfinite(hagl_arr).any():
        hagl_end = _interp(t_arr, hagl_arr, end_s)

    return {
        "available": True,
        "corridor_end_s": end_s,
        "tail_window_s": float(end_s - tail_start_s),
        "tail_vvert_start_kmh": float(v_start),
        "tail_vvert_end_kmh": float(v_end),
        "tail_vvert_gain_kmh": float(v_end - v_start),
        "tail_angle_turns": int(turn_count),
        "tail_peak_vvert_kmh": tail_peak_vvert,
        "tail_peak_t_s": tail_peak_t,
        "drop_peak_to_end_kmh": drop_peak_to_end,
        "early_release_before_corridor_end": early_release,
        "hagl_end_m": None if hagl_end is None else float(hagl_end),
    }


def _early_acc_stats(
    chart_data: dict[str, Any],
    *,
    start_s: float,
    end_s: float,
) -> tuple[float | None, float | None]:
    time_s = chart_data.get("time_s", []) or []
    acc = chart_data.get("accVert_mps2", []) or []
    if not time_s or not acc or len(time_s) != len(acc):
        return None, None

    values: list[float] = []
    for i, t in enumerate(time_s):
        t_val = _num(t)
        a_val = _num(acc[i])
        if t_val is None or a_val is None:
            continue
        if start_s <= t_val <= end_s:
            values.append(a_val)
    if not values:
        return None, None
    mean_val = float(sum(values) / len(values))
    peak_val = float(max(values))
    return mean_val, peak_val


def _add_action(
    actions_by_key: dict[str, dict[str, Any]],
    *,
    key: str,
    score: int,
    text: str,
) -> None:
    existing = actions_by_key.get(key)
    if existing is None or int(score) > int(existing.get("score", 0)):
        actions_by_key[key] = {"score": int(score), "text": text}


def _action_phase_rank(*, key: str, text: str) -> int:
    key_l = normalize_german_text(key)
    text_l = normalize_german_text(text)

    # 0: Exit / Start (0-10s)
    if any(
        token in key_l
        for token in ["exit", "start_speed_low", "early_acc", "early_vvert", "carryover"]
    ):
        return 0
    if any(normalize_german_text(token) in text_l for token in ["absprung", "nach dem exit", "ersten 2 sekunden", "startphase"]):
        return 0

    # 1: Aufbau (10-20s)
    if any(token in key_l for token in ["build", "angle_", "top5_build", "vvert_gap_20"]):
        return 1
    if "aufbau" in text_l or ("+10s" in text_l and "+20s" in text_l):
        return 1

    # 2: Hot-Zone / späte Phase (20-25s bis Korridorende)
    if any(
        token in key_l
        for token in [
            "segment_20_25",
            "vhor_mid",
            "vhor_tail",
            "vhor_price_20_25",
            "efficiency_20_25",
            "lateral_",
            "forward_track",
            "peak",
            "hold_",
            "tail",
            "corridor",
            "top5_hold",
            "vhor_gap_to_reference",
        ]
    ):
        return 2
    if any(
        normalize_german_text(token) in text_l
        for token in [
            "+20 bis +25",
            "hot-phase",
            "hot-zone",
            "peak-phase",
            "peak-bereich",
            "über 390",
            "über 400",
            "schlussteil",
            "seitbewegung",
            "seitlinie",
        ]
    ):
        return 2

    # 3: Stabilität / global
    if any(token in key_l for token in ["kipp", "curve", "stability"]):
        return 3
    if any(normalize_german_text(token) in text_l for token in ["stabilität", "kipp", "kurvenverlauf", "körperspannung"]):
        return 3

    # 4: Referenz-/Benchmark- oder sonstige Hinweise
    if "top5" in key_l or "top-5" in text_l or "reference" in key_l or "referenz" in text_l:
        return 4
    return 5


def _forward_track_behavior(
    *,
    chart_data: dict[str, Any],
    focus_end_s: float | None,
) -> dict[str, Any]:
    time_s = chart_data.get("time_s", []) or []
    forward_m = chart_data.get("forward_m", []) or []
    if not time_s or not forward_m or len(time_s) != len(forward_m):
        return {"available": False}

    pairs: list[tuple[float, float]] = []
    for i, raw_t in enumerate(time_s):
        t_val = _num(raw_t)
        f_val = _num(forward_m[i])
        if t_val is None or f_val is None:
            continue
        if t_val < 0.0:
            continue
        if focus_end_s is not None and t_val > float(focus_end_s):
            continue
        pairs.append((float(t_val), float(f_val)))
    if len(pairs) < 5:
        return {"available": False}

    t = [item[0] for item in pairs]
    fwd = [item[1] for item in pairs]
    running_max: list[float] = []
    backtrack: list[float] = []
    cur_max = float("-inf")
    for value in fwd:
        cur_max = max(cur_max, float(value))
        running_max.append(cur_max)
        backtrack.append(float(cur_max - float(value)))

    max_forward = float(max(fwd)) if fwd else 0.0
    max_backtrack = float(max(backtrack)) if backtrack else 0.0
    if max_forward < 10.0:
        return {"available": False}

    backtrack_ratio_pct = float((max_backtrack / max(max_forward, 1e-6)) * 100.0)
    threshold = max(8.0, 0.08 * max_forward)
    drift_start_s = None
    for i in range(len(t)):
        if t[i] < 8.0:
            continue
        if backtrack[i] >= threshold:
            drift_start_s = float(t[i])
            break

    label = "stabil"
    if max_backtrack >= 28.0 and backtrack_ratio_pct >= 12.0:
        label = "negativ"
    elif max_backtrack >= 14.0 and backtrack_ratio_pct >= 7.0:
        label = "leicht_negativ"

    return {
        "available": True,
        "label": label,
        "max_forward_m": max_forward,
        "max_backtrack_m": max_backtrack,
        "backtrack_ratio_pct": backtrack_ratio_pct,
        "drift_start_s": drift_start_s,
    }


def _build_priority_actions(
    actions_by_key: dict[str, dict[str, Any]],
    *,
    max_items: int = 5,
    phase_boosts: dict[int, int] | None = None,
) -> list[str]:
    if not actions_by_key:
        return []

    boosts = phase_boosts or {}
    ordered = sorted(
        actions_by_key.items(),
        key=lambda item: (
            _action_timeline_rank(key=str(item[0]), text=str(item[1].get("text", ""))),
            _action_phase_rank(key=str(item[0]), text=str(item[1].get("text", ""))),
            -(
                int(item[1].get("score", 0))
                + int(
                    boosts.get(
                        _action_phase_rank(key=str(item[0]), text=str(item[1].get("text", ""))),
                        0,
                    )
                )
            ),
            str(item[0]).lower(),
        ),
    )
    out: list[str] = []
    seen_texts: set[str] = set()
    for key, item in ordered:
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        if text in seen_texts:
            continue
        seen_texts.add(text)
        idx = len(out) + 1
        out.append(f"Priorität {idx}: {text}")
        if len(out) >= max_items:
            break
    return out


def _action_timeline_rank(*, key: str, text: str) -> int:
    key_l = normalize_german_text(key)
    text_l = normalize_german_text(text)

    if any(token in key_l for token in ["exit", "start_speed_low", "early_acc", "early_vvert", "carryover"]):
        return 0
    if any(token in key_l for token in ["phase_10_15", "top5_build_0_15"]):
        return 2
    if any(token in key_l for token in ["phase_15_20", "build_", "angle_", "vvert_gap_20"]):
        return 3
    if any(
        token in key_l
        for token in [
            "segment_20_25",
            "lateral_",
            "high_speed_stability",
            "peak",
            "efficiency_20_25",
            "vhor_price_20_25",
            "hold_",
            "top5_hold",
            "vhor_gap_to_reference",
        ]
    ):
        return 4
    if any(token in key_l for token in ["tail", "corridor", "forward_track", "vhor_tail", "vhor_mid"]):
        return 5
    if any(token in key_l for token in ["kipp", "curve", "stability"]):
        return 6

    time_marks = [float(m.group(1)) for m in re.finditer(r"\+([0-9]+(?:\.[0-9]+)?)s", text_l)]
    if time_marks:
        first_t = min(time_marks)
        if first_t <= 3.0:
            return 0
        if first_t <= 10.0:
            return 1
        if first_t <= 15.0:
            return 2
        if first_t <= 20.0:
            return 3
        if first_t <= 25.0:
            return 4
        return 5

    if any(normalize_german_text(token) in text_l for token in ["hot-phase", "hot-zone", "schlussteil", "peak-phase"]):
        return 4
    if any(normalize_german_text(token) in text_l for token in ["stabilität", "kipp", "körperspannung", "kurvenverlauf"]):
        return 6
    return 7


def _extract_phase_boosts(tip_effect_profile: dict[str, Any] | None) -> dict[int, int]:
    if not isinstance(tip_effect_profile, dict):
        return {}

    phase_aliases = {
        "0": 0,
        "exit": 0,
        "start": 0,
        "1": 1,
        "build": 1,
        "aufbau": 1,
        "2": 2,
        "hot": 2,
        "hot_zone": 2,
        "hotzone": 2,
        "peak": 2,
        "3": 3,
        "stability": 3,
        "stabilität": 3,
        "kipp_risiko": 3,
    }
    phase_aliases = {normalize_german_text(key): value for key, value in phase_aliases.items()}

    out: dict[int, int] = {}
    raw = tip_effect_profile.get("phase_boosts")
    if isinstance(raw, dict):
        for raw_key, raw_value in raw.items():
            key_text = normalize_german_text(str(raw_key).strip())
            phase = phase_aliases.get(key_text)
            if phase is None:
                parsed_idx = _num(raw_key)
                if parsed_idx is None:
                    continue
                phase = int(round(parsed_idx))
            value = _num(raw_value)
            if value is None:
                continue
            out[phase] = int(max(-4, min(6, round(value))))

    focus_phase = tip_effect_profile.get("focus_phase")
    focus_boost = _num(tip_effect_profile.get("focus_boost"))
    if focus_phase is not None and focus_boost is not None:
        key_text = normalize_german_text(str(focus_phase).strip())
        phase = phase_aliases.get(key_text)
        if phase is None:
            parsed_idx = _num(focus_phase)
            if parsed_idx is not None:
                phase = int(round(parsed_idx))
        if phase is not None:
            out[phase] = max(out.get(phase, 0), int(max(0, min(6, round(focus_boost)))))

    return out


def _extract_effect_line(tip_effect_profile: dict[str, Any] | None) -> str | None:
    if not isinstance(tip_effect_profile, dict):
        return None
    summary_line = str(tip_effect_profile.get("summary_line") or "").strip()
    if summary_line:
        return summary_line
    focus_label = str(tip_effect_profile.get("focus_label") or "").strip()
    trend_hint = str(tip_effect_profile.get("trend_hint") or "").strip()
    if focus_label and trend_hint:
        return f"Verlauf letzter Sprünge: {focus_label} ({trend_hint})."
    if focus_label:
        return f"Verlauf letzter Sprünge: Fokus aktuell {focus_label}."
    return None


def _safe_phase_0_10_target(profile: dict[str, Any]) -> str:
    low = _num(profile.get("phase_0_10_vvert_low"))
    high = _num(profile.get("phase_0_10_vvert_high"))
    if low is None:
        low = _num(profile.get("vvert_10_target_low"))
    if high is None:
        high = _num(profile.get("vvert_10_target_high"))

    if low is not None and high is not None and high >= low + 3.0:
        return f"{low:.0f} bis {high:.0f} km/h"
    if low is not None:
        return f"mindestens {low:.0f} km/h"
    return "230 bis 260 km/h"


def _safe_phase_0_10_angle_target(profile: dict[str, Any]) -> tuple[float | None, float | None, str | None]:
    low = _num(profile.get("phase_0_10_angle_low"))
    high = _num(profile.get("phase_0_10_angle_high"))
    if low is None:
        low = _num(profile.get("phase_10_15_angle_low"))
    if high is None:
        high = _num(profile.get("phase_10_15_angle_high"))
    if low is None or high is None or high <= low:
        return None, None, None

    mode = str(profile.get("capability_mode") or "").strip().lower()
    width = float(high - low)
    target_low = float(low)
    target_high = float(high)
    if mode == "safe":
        target_high = float(low + max(0.8, width * 0.65))
    elif mode == "build":
        target_high = float(low + max(1.0, width * 0.80))

    target_high = max(target_low + 0.4, min(float(high), target_high))
    return target_low, target_high, f"{target_low:.1f} bis {target_high:.1f} Grad"


def _build_personal_hot_zone_chain(
    *,
    profile: dict[str, Any],
    capability_mode: str,
    angle_20: float | None,
    min_vhor_20_25: float | None,
    tail_turns: float | None,
) -> str | None:
    if not bool(profile.get("is_personalized")):
        return None

    risk_angle = _num(profile.get("angle_20_risk_above"))
    if risk_angle is None:
        target_high = _num(profile.get("angle_20_target_high"))
        if target_high is not None:
            risk_angle = target_high + 0.8
    vhor_floor = _num(profile.get("vhor_min_20_25_floor"))
    if vhor_floor is None:
        vhor_floor = 25.0
    turn_limit = _num(profile.get("angle_turns_20_25_max"))
    if turn_limit is None:
        turn_limit = 4.0

    angle_issue = bool(risk_angle is not None and angle_20 is not None and angle_20 >= risk_angle)
    vhor_issue = bool(min_vhor_20_25 is not None and min_vhor_20_25 < vhor_floor)
    turn_issue = bool(tail_turns is not None and tail_turns >= turn_limit)
    if not (angle_issue or vhor_issue or turn_issue):
        return None

    parts: list[str] = ["Typische Kette bei dir:"]
    if angle_issue:
        parts.append(
            f"ab etwa {risk_angle:.1f} Grad wird die Linie oft zu steil."
        )
    if vhor_issue:
        parts.append(
            f"Dadurch fällt vHor schnell Richtung {min_vhor_20_25:.1f} km/h (unter deinem stabilen Bereich von ca. {vhor_floor:.1f})."
        )
    if turn_issue:
        parts.append(
            "Danach folgen mehrere Nachkorrekturen im Schlussteil."
        )

    if capability_mode == "safe":
        parts.append("Besser früher klein korrigieren und den Winkel langsamer aufbauen.")
    elif capability_mode == "push":
        parts.append("Push erst dann, wenn die Linie ruhig bleibt und vHor nicht abreißt.")

    return " ".join(parts)


def _extract_personal_tip_profile(stability_reference: dict[str, Any] | None) -> dict[str, Any]:
    out: dict[str, Any] = {"is_personalized": False}
    if not isinstance(stability_reference, dict):
        return out
    if not bool(stability_reference.get("available")):
        return out

    thresholds = stability_reference.get("thresholds", {})
    if isinstance(thresholds, dict):
        for key in [
            "angle_20_target_low",
            "angle_20_target_high",
            "angle_20_risk_above",
            "phase_0_10_angle_low",
            "phase_0_10_angle_high",
            "vhor_min_20_25_floor",
            "angle_turns_20_25_max",
            "gain_10_20_target_low",
            "gain_10_20_target_high",
            "vvert_10_target_low",
            "vvert_10_target_high",
            "phase_0_10_vvert_low",
            "phase_0_10_vvert_high",
            "phase_10_15_gain_low",
            "phase_10_15_gain_high",
            "phase_15_20_gain_low",
            "phase_15_20_gain_high",
            "phase_10_15_angle_low",
            "phase_10_15_angle_high",
            "phase_15_20_angle_low",
            "phase_15_20_angle_high",
            "lateral_hot_abs_target_high",
            "lateral_hot_heading_rate_target_high",
            "lateral_hot_alat_peak_target_high",
            "lateral_hot_sign_changes_max",
        ]:
            value = _num(thresholds.get(key))
            if value is not None:
                out[key] = value

    capability = stability_reference.get("capability_profile", {})
    if isinstance(capability, dict):
        mode = str(capability.get("mode") or "").strip().lower()
        if mode:
            out["capability_mode"] = mode
        text = str(capability.get("text") or "").strip()
        if text:
            out["capability_text"] = text
        ratio = _num(capability.get("stable_ratio_pct"))
        if ratio is not None:
            out["capability_ratio_pct"] = ratio

    performance = stability_reference.get("performance_profile", {})
    if isinstance(performance, dict) and performance.get("available"):
        summary = str(performance.get("summary") or "").strip()
        if summary:
            out["performance_text"] = summary
        band = str(performance.get("performance_band") or "").strip().lower()
        if band:
            out["performance_band"] = band
        confidence = str(performance.get("confidence") or "").strip().lower()
        if confidence:
            out["performance_confidence"] = confidence
        top_avg = _num(performance.get("top_available_avg_kmh"))
        if top_avg is not None:
            out["performance_top_avg_kmh"] = top_avg

    bands = stability_reference.get("bands", {})
    stable = bands.get("stable", {}) if isinstance(bands, dict) else {}
    if isinstance(stable, dict):
        angle20 = stable.get("angle_20s")
        if isinstance(angle20, dict):
            if "angle_20_target_low" not in out:
                out["angle_20_target_low"] = _num(angle20.get("low"))
            if "angle_20_target_high" not in out:
                out["angle_20_target_high"] = _num(angle20.get("high"))
        gain = stable.get("gain_10_20")
        if isinstance(gain, dict) and "gain_10_20_target_low" not in out:
            out["gain_10_20_target_low"] = _num(gain.get("low"))
        v10 = stable.get("vvert_10s")
        if isinstance(v10, dict):
            if "vvert_10_target_low" not in out:
                out["vvert_10_target_low"] = _num(v10.get("low"))
            if "vvert_10_target_high" not in out:
                out["vvert_10_target_high"] = _num(v10.get("high"))
        vhor = stable.get("vhor_min_20_25")
        if isinstance(vhor, dict) and "vhor_min_20_25_floor" not in out:
            out["vhor_min_20_25_floor"] = _num(vhor.get("low"))

    asymmetry = stability_reference.get("asymmetry_profile", {})
    if isinstance(asymmetry, dict) and bool(asymmetry.get("available")):
        direction = str(asymmetry.get("direction") or "").strip().lower()
        if direction in {"rechts", "links"}:
            out["asymmetry_direction"] = direction
            out["asymmetry_available"] = True
        hit_ratio = _num(asymmetry.get("hit_ratio_pct"))
        if hit_ratio is not None:
            out["asymmetry_hit_ratio_pct"] = hit_ratio
        text = str(asymmetry.get("text") or "").strip()
        note = str(asymmetry.get("note") or "").strip()
        if text and note:
            out["asymmetry_text"] = f"{text} {note}"
        elif text:
            out["asymmetry_text"] = text

    out["is_personalized"] = any(
        out.get(key) is not None
        for key in [
            "angle_20_target_low",
            "angle_20_target_high",
            "vhor_min_20_25_floor",
            "gain_10_20_target_low",
            "phase_10_15_gain_low",
            "phase_15_20_gain_low",
            "vvert_10_target_low",
        ]
    )
    return out


def _angle_target_for_flat(angle_20: float | None, *, profile: dict[str, Any] | None = None) -> str:
    target_low = None if profile is None else _num(profile.get("angle_20_target_low"))
    target_high = None if profile is None else _num(profile.get("angle_20_target_high"))
    capability_mode = "" if profile is None else str(profile.get("capability_mode") or "").lower()
    early_low = None if profile is None else _num(profile.get("phase_0_10_angle_low"))
    early_high = None if profile is None else _num(profile.get("phase_0_10_angle_high"))
    if target_low is not None and target_high is not None and target_high > target_low:
        entry_low = max(72.0, target_low - 2.0)
        step_high = min(88.5, target_high + 1.0)
        if capability_mode == "safe":
            safe_high = min(target_high, target_low + max(0.8, (target_high - target_low) * 0.6))
            if early_low is not None and early_high is not None and early_high > early_low:
                early_safe_high = min(early_high, early_low + max(0.8, (early_high - early_low) * 0.65))
                return (
                    f"bis +10s zuerst stabil auf {early_low:.1f} bis {early_safe_high:.1f} Grad kommen, "
                    f"danach bis +20s ruhig in den Bereich {target_low:.1f} bis {safe_high:.1f} Grad gehen."
                )
            return (
                f"zuerst stabil auf {entry_low:.1f} bis {safe_high:.1f} Grad kommen und dort ruhig halten, "
                "ohne sofort weiter zu steilen."
            )
        if capability_mode == "push":
            push_low = max(target_low, target_high - 1.0)
            return (
                f"erst stabil in den Korridor {target_low:.1f} bis {target_high:.1f} Grad gehen, "
                f"dann kontrolliert die obere Zone um {push_low:.1f} bis {target_high:.1f} Grad anpeilen."
            )
        if angle_20 is None or angle_20 < entry_low:
            return (
                f"zuerst auf {entry_low:.1f} bis {target_high:.1f} Grad stabilisieren, "
                f"dann schrittweise Richtung {step_high:.1f} Grad aufbauen."
            )
        if angle_20 < target_low:
            return f"leicht steiler werden und stabil im Bereich {target_low:.1f} bis {target_high:.1f} Grad halten."
        return (
            f"im Bereich {target_low:.1f} bis {target_high:.1f} Grad bleiben und nur in kleinen Schritten "
            "weiter steigern."
        )
    if angle_20 is None:
        return "zuerst auf 78 bis 81 Grad stabilisieren, danach Richtung 82 bis 84 Grad aufbauen."
    if angle_20 < 70:
        return "zuerst auf 74 bis 78 Grad stabilisieren, danach Richtung 80 bis 82 Grad aufbauen."
    if angle_20 < 78:
        return "zuerst auf 78 bis 81 Grad stabilisieren, danach Richtung 82 bis 84 Grad aufbauen."
    if angle_20 < 82:
        return "stabil im Bereich 82 bis 84 Grad halten."
    return "nur leicht um 1 bis 2 Grad steiler machen und Stabilität priorisieren."


def _angle_target_for_steep(angle_20: float | None, *, profile: dict[str, Any] | None = None) -> str:
    target_low = None if profile is None else _num(profile.get("angle_20_target_low"))
    target_high = None if profile is None else _num(profile.get("angle_20_target_high"))
    risk_above = None if profile is None else _num(profile.get("angle_20_risk_above"))
    capability_mode = "" if profile is None else str(profile.get("capability_mode") or "").lower()
    if target_low is not None and target_high is not None and target_high > target_low:
        cap = target_high if risk_above is None else min(target_high, risk_above - 0.2)
        cap = max(target_low + 0.4, cap)
        if capability_mode == "safe":
            cap = min(cap, target_low + max(0.8, (target_high - target_low) * 0.6))
        if angle_20 is not None and risk_above is not None and angle_20 >= risk_above:
            return (
                f"im nächsten Sprung klar flacher planen: bei +20s zuerst den Zielkorridor "
                f"{target_low:.1f} bis {cap:.1f} Grad anpeilen."
            )
        return (
            "im nächsten Sprung etwas flacher bleiben und den Winkel stabil im Zielkorridor "
            f"{target_low:.1f} bis {cap:.1f} Grad halten."
        )
    if angle_20 is None:
        return "Im nächsten Sprung zwischen +10s und +20s etwas flacher bleiben und stabil bei 82 bis 84 Grad halten."
    if angle_20 >= 88:
        return "Im nächsten Sprung zwischen +10s und +20s klar flacher gehen und den Zielkorridor 82 bis 84 Grad anpeilen."
    if angle_20 >= 86:
        return "Im nächsten Sprung zwischen +10s und +20s leicht flacher werden und 83 bis 85 Grad stabil halten."
    return "Im nächsten Sprung etwas flacher bleiben, aber nur in kleinen Schritten (1 bis 2 Grad)."


def _angle_progression_stability_analysis(
    *,
    chart_data: dict[str, Any],
    focus_end_s: float | None,
    eval_end_s: float | None,
    profile: dict[str, Any] | None,
) -> dict[str, Any]:
    end_candidates = [25.0]
    if focus_end_s is not None:
        end_candidates.append(float(focus_end_s))
    if eval_end_s is not None:
        end_candidates.append(float(eval_end_s))
    end_s = min(end_candidates)
    if end_s < 13.0:
        return {"available": False}

    t_angle, angle = _window_series(chart_data, "angle_deg", start_s=0.0, end_s=end_s)
    if len(t_angle) < 10 or len(angle) < 10:
        return {"available": False}

    angle_s = _moving_average(angle, window=5)
    t_arr = np.asarray(t_angle, dtype=float)
    angle_arr = np.asarray(angle_s, dtype=float)

    angle_10 = _interp(t_arr, angle_arr, 10.0)
    angle_15 = _interp(t_arr, angle_arr, 15.0)
    angle_20 = _interp(t_arr, angle_arr, 20.0)

    peak_start_s = 12.0
    peak_start_idx = 0
    for i, t in enumerate(t_angle):
        if t >= peak_start_s:
            peak_start_idx = i
            break

    if peak_start_idx >= len(angle_s) - 2:
        return {"available": False}

    peak_idx = max(range(peak_start_idx, len(angle_s)), key=lambda idx: angle_s[idx])
    t_peak = float(t_angle[peak_idx])
    angle_peak = float(angle_s[peak_idx])
    t_end = float(t_angle[-1])
    angle_end = float(mean(angle_s[-min(3, len(angle_s)) :]))

    rollback_deg = max(0.0, angle_peak - angle_end)
    post_slice = angle_s[peak_idx:]
    turns_after_peak = _turn_count(post_slice, eps=0.16) if len(post_slice) >= 4 else 0

    vvert_drop_after_peak = None
    t_vvert, vvert = _window_series(chart_data, "vVert_kmh", start_s=0.0, end_s=end_s)
    if len(t_vvert) >= 10 and len(t_vvert) == len(vvert):
        vvert_s = _moving_average(vvert, window=5)
        t_vvert_arr = np.asarray(t_vvert, dtype=float)
        vvert_arr = np.asarray(vvert_s, dtype=float)
        vvert_at_peak = _interp(t_vvert_arr, vvert_arr, t_peak)
        vvert_at_end = _interp(t_vvert_arr, vvert_arr, t_end)
        if vvert_at_peak is not None and vvert_at_end is not None:
            vvert_drop_after_peak = max(0.0, vvert_at_peak - vvert_at_end)

    phase_10_15_high = None if profile is None else _num(profile.get("phase_10_15_angle_high"))
    if phase_10_15_high is None and profile is not None:
        phase_10_15_high = _num(profile.get("angle_20_target_high"))
    risk_above = None if profile is None else _num(profile.get("angle_20_risk_above"))
    turns_limit = None if profile is None else _num(profile.get("angle_turns_20_25_max"))
    turn_threshold = max(2.0, (turns_limit + 0.5) if turns_limit is not None else 2.0)

    angle_gain_10_15 = None
    if angle_10 is not None and angle_15 is not None:
        angle_gain_10_15 = angle_15 - angle_10

    steep_too_fast = False
    if angle_15 is not None:
        if phase_10_15_high is not None:
            steep_too_fast = angle_15 >= (phase_10_15_high + 1.0)
        else:
            steep_too_fast = angle_15 >= 83.0
    if angle_gain_10_15 is not None and angle_gain_10_15 >= 12.0:
        steep_too_fast = True
    if risk_above is not None and angle_20 is not None and angle_20 >= risk_above:
        steep_too_fast = True

    has_stable_post_window = (t_end - t_peak) >= 2.0
    rollback_with_instability = (
        has_stable_post_window
        and rollback_deg >= 2.5
        and (
            turns_after_peak >= turn_threshold
            or (vvert_drop_after_peak is not None and vvert_drop_after_peak >= 12.0)
        )
    )
    too_fast_steep_not_hold = steep_too_fast and rollback_with_instability

    return {
        "available": True,
        "angle_10": angle_10,
        "angle_15": angle_15,
        "angle_20": angle_20,
        "angle_peak": angle_peak,
        "angle_end": angle_end,
        "t_peak_s": t_peak,
        "rollback_deg": rollback_deg,
        "turns_after_peak": float(turns_after_peak),
        "vvert_drop_after_peak_kmh": vvert_drop_after_peak,
        "too_fast_steep_not_hold": bool(too_fast_steep_not_hold),
        "rollback_with_instability": bool(rollback_with_instability and not too_fast_steep_not_hold),
    }


def _curve_smoothness_analysis(
    *,
    chart_data: dict[str, Any],
    start_s: float,
    end_s: float | None,
    angle_10: float | None,
    angle_20: float | None,
) -> dict[str, Any]:
    t_vhor, vhor = _window_series(chart_data, "vHor_kmh", start_s=start_s, end_s=end_s)
    t_angle, angle = _window_series(chart_data, "angle_deg", start_s=start_s, end_s=end_s)
    t_vvert, vvert = _window_series(chart_data, "vVert_kmh", start_s=start_s, end_s=end_s)
    if len(vhor) < 12 or len(angle) < 12 or len(vvert) < 12:
        return {"label": "unknown", "causes": [], "actions": []}

    vhor_s = _moving_average(vhor, window=7)
    angle_s = _moving_average(angle, window=7)
    vvert_s = _moving_average(vvert, window=7)

    duration = max(1.0, t_vhor[-1] - t_vhor[0])
    turns_vhor = _turn_count(vhor_s, eps=0.45)
    turns_angle = _turn_count(angle_s, eps=0.18)
    turns_vhor_10s = turns_vhor / duration * 10.0
    turns_angle_10s = turns_angle / duration * 10.0

    dvvert = [vvert_s[i] - vvert_s[i - 1] for i in range(1, len(vvert_s))]
    dvvert_std = pstdev(dvvert) if len(dvvert) >= 2 else 0.0

    dip_pct, rebound_pct, dip_vhor = _vhor_dip_rebound(vhor_s, t_vhor)
    negative_ratio = _negative_segment_ratio(vhor_s, angle_s)

    score = 0
    if turns_vhor_10s > 3.2:
        score += 2
    if turns_angle_10s > 4.2:
        score += 2
    if dvvert_std > 1.6:
        score += 1
    correction_pattern = turns_vhor_10s > 2.8 or turns_angle_10s > 3.8
    if dip_pct > 0.25 and rebound_pct > 0.16 and correction_pattern:
        score += 2
    if negative_ratio > 0.10:
        score += 2

    label = "sauber"
    if score >= 6:
        label = "unruhig"
    elif score >= 3:
        label = "leicht_unruhig"

    causes: list[str] = []
    actions: list[dict[str, Any]] = []

    early_too_steep = bool(angle_10 is not None and angle_10 > 78.0) or max(angle_s[: max(6, len(angle_s) // 6)]) > 87.0
    late_hard_build = bool(
        angle_10 is not None and angle_20 is not None and angle_10 < 76.0 and angle_20 > 84.0
    )

    if label in {"leicht_unruhig", "unruhig"}:
        if early_too_steep and turns_angle_10s > 4.2:
            causes.append("Wahrscheinlicher Grund: zu früh sehr steil geworden, danach mehrmals nachkorrigiert.")
            actions.append(
                {
                    "key": "curve_early_steep",
                    "score": 8 if label == "unruhig" else 6,
                    "text": "Bis etwa +12s etwas ruhiger aufbauen und nicht zu früh maximal steil werden.",
                }
            )
        if late_hard_build and turns_angle_10s > 4.2:
            causes.append("Wahrscheinlicher Grund: zuerst zu flach, dann zu harter Übergang in den Steilflug.")
            actions.append(
                {
                    "key": "curve_late_hard_build",
                    "score": 8 if label == "unruhig" else 6,
                    "text": "Übergang in den Steilflug gleichmäßiger fahren (früher und mit kleineren Schritten).",
                }
            )
        if dip_pct > 0.25 and rebound_pct > 0.16 and correction_pattern:
            causes.append(
                "Wahrscheinlicher Grund: starke vHor-Einbrueche mit Gegenkorrektur (Druck nicht konstant gehalten)."
            )
            actions.append(
                {
                    "key": "curve_vhor_dip_rebound",
                    "score": 7,
                    "text": (
                        "In der Peak-Phase Druck konstanter halten, damit vHor nicht stark einbricht und zurückfedert."
                    ),
                }
            )
        if negative_ratio > 0.10:
            causes.append(
                "Es gibt Abschnitte im sehr steilen/negativen Bereich; das fuehrt oft zu Korrekturketten."
            )
            actions.append(
                {
                    "key": "curve_negative_segments",
                    "score": 7,
                    "text": "Extrem steile Abschnitte kürzer halten und früher kleine Korrekturen setzen.",
                }
            )
        if not causes:
            causes.append("Die Kurven zeigen mehrere Richtungswechsel, der Ablauf wirkt nicht gleichmäßig.")
            actions.append(
                {
                    "key": "curve_general_smooth",
                    "score": 6,
                    "text": "Korrekturen kleiner und früher setzen, damit der Kurvenverlauf ruhiger wird.",
                }
            )

    return {
        "label": label,
        "turns_vhor_10s": turns_vhor_10s,
        "turns_angle_10s": turns_angle_10s,
        "dvvert_std": dvvert_std,
        "dip_pct": dip_pct,
        "rebound_pct": rebound_pct,
        "dip_vhor": dip_vhor,
        "negative_ratio": negative_ratio,
        "causes": causes,
        "actions": actions,
    }


def _window_series(
    chart_data: dict[str, Any],
    key: str,
    *,
    start_s: float,
    end_s: float | None,
) -> tuple[list[float], list[float]]:
    time_s = chart_data.get("time_s", []) or []
    values = chart_data.get(key, []) or []
    if not time_s or not values or len(time_s) != len(values):
        return [], []
    t_out: list[float] = []
    v_out: list[float] = []
    for i, t in enumerate(time_s):
        t_val = _num(t)
        v_val = _num(values[i])
        if t_val is None or v_val is None:
            continue
        if t_val < start_s:
            continue
        if end_s is not None and t_val > end_s:
            continue
        t_out.append(t_val)
        v_out.append(v_val)
    return t_out, v_out


def _moving_average(values: list[float], *, window: int) -> list[float]:
    if window <= 1 or len(values) <= 2:
        return values[:]
    half = max(1, window // 2)
    out: list[float] = []
    n = len(values)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        out.append(float(mean(values[lo:hi])))
    return out


def _turn_count(values: list[float], *, eps: float) -> int:
    if len(values) < 3:
        return 0
    turns = 0
    prev_sign = 0
    for i in range(1, len(values)):
        diff = values[i] - values[i - 1]
        sign = 1 if diff > eps else -1 if diff < -eps else 0
        if sign == 0:
            continue
        if prev_sign != 0 and sign != prev_sign:
            turns += 1
        prev_sign = sign
    return turns


def _vhor_dip_rebound(vhor: list[float], times: list[float]) -> tuple[float, float, float]:
    if len(vhor) < 10 or len(times) != len(vhor):
        return 0.0, 0.0, 0.0
    dip_idx = min(range(len(vhor)), key=lambda i: vhor[i])
    dip = float(vhor[dip_idx])
    pre_max = float(max(vhor[: dip_idx + 1])) if dip_idx > 0 else float(vhor[0])
    dip_pct = 0.0 if pre_max <= 1e-6 else max(0.0, (pre_max - dip) / pre_max)

    if dip_idx >= len(vhor) - 2:
        return dip_pct, 0.0, dip

    dt = (times[-1] - times[0]) / max(1, len(times) - 1)
    future_n = max(3, int(round(6.0 / max(0.08, dt))))
    end_idx = min(len(vhor), dip_idx + 1 + future_n)
    post_max = float(max(vhor[dip_idx + 1 : end_idx]))
    rebound_pct = 0.0 if dip <= 1e-6 else max(0.0, (post_max - dip) / dip)
    return dip_pct, rebound_pct, dip


def _negative_segment_ratio(vhor: list[float], angle: list[float]) -> float:
    if not vhor or not angle or len(vhor) != len(angle):
        return 0.0
    count = 0
    for i in range(len(vhor)):
        if angle[i] >= 87.0 and vhor[i] <= 25.0:
            count += 1
    return count / len(vhor)
