import numpy as np
import math


def update_iob(dt, tau, iob, insulin_delivered):
    alpha = np.exp(-dt / tau)
    iob = alpha * iob + insulin_delivered
    return iob


def calculate_meal_units(CR, meal_cho):
    if meal_cho and meal_cho > 0:
        # print(f"Meal Happened! ${meal_cho}")
        meal_units = (meal_cho) / CR
    else:
        meal_units = 0.0

    return meal_units


def normalize_state(tdi, state):
    state = np.asarray(state, dtype=np.float32)
    normalized = np.zeros_like(state, dtype=np.float32)

    normalized[0] = (state[0] - 140) / 100  # CGM
    normalized[1] = state[1] / 5  # Trend -> already normalized
    normalized[2] = (
        state[2] / tdi
    )  # Insulin on Board / Average Total daily insulin -> might need to change this
    normalized[3] = state[3]  # Time sin -> already normalized
    normalized[4] = state[4]  # Time cos -> already normalized
    normalized[5] = state[5] / 5  # Meal units

    return normalized


def encode_time(current_time):
    hour = current_time.hour + current_time.minute / 60.0

    time_sin = np.sin(2 * np.pi * hour / 24)
    time_cos = np.cos(2 * np.pi * hour / 24)

    return time_sin, time_cos


def calculate_normalised_trend(cgm_history, dt):
    if len(cgm_history) >= 3:
        recent_cgm_history = cgm_history[-3:]
        slope = (recent_cgm_history[2] - recent_cgm_history[0]) / (2 * dt)
        trend = slope
    else:
        trend = 0

    return trend


# def compute_safety_gate(iob, cgm, trend):
#     gate = 1.0
#
#     if cgm <= 80:
#         gate = 0.0
#     elif cgm <= 90:
#         gate = 0.5
#     if trend < -1.5:
#         gate = min(gate, 0.3)
#     if iob > 10.0:
#         gate = min(gate, 0.3)
#     elif iob > 8.0:
#         gate = min(gate, 0.5)
#     return gate


def _sigmoid(x):
    x = max(-60.0, min(60.0, x))  # guard overflow
    return 1.0 / (1.0 + math.exp(-x))


def _smoothstep(x):
    x = max(0.0, min(1.0, x))
    return x * x * (3 - 2 * x)


def compute_safety_gate(
    iob,
    cgm,
    trend,
    tdi,
    cf,
    prev_gate=None,
    horizon_min=25,
    hypo_thresh=70.0,
    low_warn=90.0,
    target_cgm=120,
):
    # --- Adaptive horizon (shorter when clearly hyper and not falling) ---
    h_eff = (
        horizon_min
        if not (cgm > target_cgm + 40.0 and trend >= 0.0)
        else min(horizon_min, 15.0)
    )

    # --- IOB-aware prediction: short-horizon insulin effect fraction (25–35%) ---
    iob_frac_effect = 0.3
    pred_glucose_trend = cgm + trend * h_eff
    pred_glucose_iob = cgm - (iob * cf * iob_frac_effect)
    # be conservative in low-risk checks
    predicted_glucose = min(pred_glucose_trend, pred_glucose_iob)

    # --- Hard hypo guard ---
    if cgm <= hypo_thresh or predicted_glucose <= hypo_thresh:
        return 0.0

    # --- Component factors ---
    # Low-risk factor (now vs short-horizon prediction)
    k_low = 0.15
    f_low_now = _sigmoid(k_low * (cgm - low_warn))
    f_low_pred = _sigmoid(k_low * (predicted_glucose - low_warn))
    f_low = min(f_low_now, f_low_pred)

    # Trend factor (softer near flat; still clamps on real negatives)
    k_trend = 0.5
    f_trend = _sigmoid(k_trend * (trend + 0.3))

    iob_scale = max(120.0, 0.5 * cf * max(10.0, tdi))
    p_iob = _smoothstep(min(1.0, (iob * cf) / iob_scale))

    if trend < -0.3 or cgm < low_warn:
        f_iob = 1.0 - 0.85 * p_iob
    else:
        f_iob = 1.0 - 0.3 * p_iob  # was 0.7 → clamp harder

    base_safety_gate = min(f_low, f_trend, f_iob)

    # --- Recovery floor (allow insulin when safely high), tempered by IOB burden ---
    if trend >= 0.0 or cgm >= target_cgm:
        overshoot = max(0.0, cgm - (target_cgm + 10.0))
        recovery_floor = 0.6 + 0.3 * min(1.0, overshoot / 70.0)

        burden = min(1.0, (iob * cf) / max(1e-6, iob_scale))
        recovery_floor *= 1.0 - 0.25 * burden  # temper more if stacked IOB
        base_safety_gate = max(base_safety_gate, recovery_floor)

    # --- Absolute IOB failsafes (stop stacking into a fall/flat) ---
    if iob >= 12.0 and trend <= 0.5:
        base_safety_gate = min(base_safety_gate, 0.25)
    elif iob >= 10.0 and trend <= 0.2:
        base_safety_gate = min(base_safety_gate, 0.40)

    # --- Asymmetric smoothing: open faster, close slower ---
    if prev_gate is not None:
        opening = base_safety_gate > prev_gate
        alpha_open, alpha_close = 0.3, 0.8
        alpha = alpha_open if opening else alpha_close
        gate = alpha * prev_gate + (1.0 - alpha) * base_safety_gate
    else:
        gate = base_safety_gate

    return float(max(0.0, min(1.0, gate)))


def calculate_insulin_delivered(
    meal_units, gate, correction_residual, max_bolus, dt, basal_rate
):
    meal_units = max(0.0, meal_units)
    meal_bolus_units = min(meal_units, max_bolus)

    # Residual capacity: can add up to remaining headroom; can subtract up to the deterministic meal bolus
    residual_max_capacity = max(0.0, max_bolus - meal_bolus_units)
    residual_min_capacity = -meal_bolus_units

    # Split residual into positive (extra insulin) and negative (subtract insulin)
    pos = np.clip(correction_residual, 0.0, residual_max_capacity) * gate
    neg = -np.clip(-correction_residual, 0.0, -residual_min_capacity)

    bolus_units = np.clip(
        meal_bolus_units + pos + neg, 0.0, max_bolus
    )  # final safety clip
    bolus_rate = bolus_units / dt
    basal_units_delivered = basal_rate * dt
    total_units = bolus_units + basal_units_delivered
    return total_units, bolus_rate, bolus_units
