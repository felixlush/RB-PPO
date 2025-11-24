import gymnasium as gym
import numpy as np
import pandas as pd
import pkg_resources
from simglucose.controller.base import Action
from gymnasium.spaces import Box
from utils import (
    calculate_meal_units,
    calculate_normalised_trend,
    encode_time,
    update_iob,
    compute_safety_gate,
    calculate_insulin_delivered,
)

"""
This class wraps the T1DSimGymnaisumEnv class from simglucose/env/simglucose_gym_env it transforms the observation and action spaces to fit our proposed solution:
Observation space consists of:
    [CGM, CGM_trend, IOB, Time, CR, Meal_Units]
and Action space consists of:
    [basal_rate, correction_residual]

Info returned from step is:
        sample_time
        patient_name
        meal
        patient_state
        time
        bg
        lbgi
        hbgi
        risk
"""


class SimglucoseWrapperNoSafety(gym.Wrapper):
    def __init__(self, env, debug=False):
        super().__init__(env)

        self.core_env = self.env.env.env

        # Get carb ratio from Quest.csv
        CONTROL_QUEST = pkg_resources.resource_filename(
            "simglucose", "params/Quest.csv"
        )
        quest_df = pd.read_csv(CONTROL_QUEST)
        patient_name = self.core_env.patient.name
        if any(quest_df.Name.str.match(patient_name)):
            quest = quest_df[quest_df.Name.str.match(patient_name)]
            self.CR = quest.CR.values.item()
            self.tdi = quest.TDI.values.item()
            self.CF = quest.CF.values.item()
        else:
            self.CR = 15  # Default fallback
            self.tdi = 50.0
            self.CF = 20.0

        self.dt = float(getattr(self.core_env, "sample_time", 5.0))
        self.cgm_history = self.env.env.env.CGM_hist

        # Get Insulin Pump Parameters
        self.pump = self.core_env.pump
        self.pump_params = self.pump._params

        # Max Bolus at one time -> used in calculating insulin delivered
        self.max_bolus = self.pump_params.get("max_bolus", 75.0)
        self.target_cgm = 120.0
        # Max Basal Per Hr and Per Minute (simulator doses in minutes)
        self.max_basal_per_hr = 35.0
        self.max_basal_u_per_min = self.max_basal_per_hr / 60.0

        # Max correction per hour and per step (5min default step)
        self.max_correction_per_hr = self.pump_params.get("max_correction", 12.0)
        self.max_correction_per_step = (self.max_correction_per_hr / 60.0) * self.dt

        self.tau = 180.0  # 3hrs Insulin Action Time
        self.iob = 0.0
        self.prev_gate = 1.0
        self.debug = debug

        self.observation_space = self._build_observation_space()
        self.action_space = self._build_action_space()

    def _build_observation_space(self):
        # Define bounds for each component
        # CGM: reading: 0-1000 mg/dl
        # Trend: -10 to +10 mg/dl/min
        # IOB: 0-100 units
        # Time_sin: -1, 1
        # Time_cos: -1, 1
        # Meal units: 0-20 U
        # Correction Factor: 4-50 mg/dl/U
        # Total Daily Insulin: 10-80 U
        low = np.array([0, -10, 0, -1, -1, 0, 4, 10], dtype=np.float32)
        high = np.array([1000, 10, 100, 1, 1, 20, 50, 80], dtype=np.float32)

        return Box(low=low, high=high, dtype=np.float32)

    def _build_action_space(self):
        # Space should be 2D -> [basal_rate, correction_residual]
        # Basal -> 0 to max_basal
        # Correction -max_correction to +max_correction

        low = np.array([0, -self.max_correction_per_step], dtype=np.float32)
        high = np.array(
            [self.max_basal_u_per_min, self.max_correction_per_step], dtype=np.float32
        )

        return Box(low=low, high=high, dtype=np.float32)

    def _extract_state(self, obs):
        if hasattr(obs, "CGM"):
            cgm_value = obs.CGM
        elif isinstance(obs, np.ndarray):
            cgm_value = obs[0]
        else:
            cgm_value = float(obs)

        current_time = self.core_env.time
        trend = calculate_normalised_trend(self.cgm_history, self.dt)
        iob = self.iob
        time_sin, time_cos = encode_time(current_time)

        # Calculate CHO units based on action
        patient_action = self.core_env.scenario.get_action(current_time)
        meal_cho = 0.0
        if isinstance(patient_action, dict) and "meal" in patient_action:
            meal_cho = float(patient_action["meal"])

        meal_units = calculate_meal_units(self.CR, meal_cho)

        if self.debug:
            print(
                f"obs type: {type(obs)}, shape: {obs.shape if hasattr(obs, 'shape') else 'N/A'}"
            )
            print(f"obs: {obs}")
            print(f"trend: {trend}, type: {type(trend)}")
            print(f"iob: {iob}, type: {type(iob)}")
            print(f"time_sin: {time_sin}, type: {type(time_sin)}")
            print(f"time_cos: {time_cos}, type: {type(time_cos)}")
            print(f"meal_units: {meal_units}, type: {type(meal_units)}")

        return np.array(
            [cgm_value, trend, iob, time_sin, time_cos, meal_units, self.CF, self.tdi],
            dtype=np.float32,
        )

    def step(self, action):
        self.dt = float(getattr(self.core_env, "sample_time", 5.0))
        cgm_now = float(self.cgm_history[-1])
        trend_now = calculate_normalised_trend(self.cgm_history, self.dt)
        gate = 1.0
        # gate = compute_safety_gate(
        #     self.iob, cgm_now, trend_now, self.tdi, self.CF, prev_gate=self.prev_gate
        # )

        # Guard NaNs/Infs in the incoming action
        try:
            arr = np.asarray(action, dtype=np.float32)
        except Exception:
            arr = np.zeros(2, dtype=np.float32)
        if not np.all(np.isfinite(arr)):
            arr = np.nan_to_num(
                arr, nan=0.0, posinf=self.max_basal_u_per_min, neginf=0.0
            )

        # Basal command (u/min before gating)
        basal_cmd = float(arr[0])
        basal_rate = basal_cmd * gate

        # Determine announced meal (deterministic bolus event)
        meal_cho = 0.0
        try:
            patient_action = self.core_env.scenario.get_action(self.core_env.time)
            if isinstance(patient_action, dict) and "meal" in patient_action:
                meal_cho = float(patient_action["meal"])
            elif hasattr(patient_action, "meal"):
                meal_cho = float(patient_action.meal)
            elif hasattr(patient_action, "CHO"):
                meal_cho = float(patient_action.CHO)
        except Exception:
            pass

        meal_units = calculate_meal_units(self.CR, meal_cho)

        if meal_units > 0:
            max_negative = meal_units
        else:
            max_negative = 0.0

        # enforce only negative correction if meal event
        predicted_correction = float(arr[1])
        if max_negative > 0 and predicted_correction < 0:
            predicted_correction = predicted_correction
        elif max_negative <= 0 and predicted_correction < 0:
            predicted_correction = 0.0
        else:
            predicted_correction = predicted_correction

        # Safety Mechanism to reduce correction if cgm is falling and close to target
        if trend_now <= 0.0 and cgm_now <= self.target_cgm + 15:
            predicted_correction = min(
                0.0, predicted_correction
            )  # no positive residual
        # Clip correction to max
        predicted_correction = float(
            np.clip(predicted_correction, -max_negative, self.max_correction_per_step)
        )

        total_insulin_delivered, bolus_rate, bolus_units = calculate_insulin_delivered(
            meal_units,
            gate,
            predicted_correction,
            self.max_bolus,
            self.dt,
            basal_rate,
        )

        if self.debug:
            print(f"Time: {self.core_env.time}")
            print(
                f"Correction: {predicted_correction:.2f} -> gated: {predicted_correction * gate:.2f}"
            )
            print(f"Basal cmd: {basal_cmd:.2f} -> gated: {basal_rate:.2f}")
            print(f"Meal units: {meal_units:.2f} (bolus_event={meal_units > 0.0})")
            print(f"Total bolus: {bolus_units:.2f}")
            print(f"IOB: {self.iob:.2f}")
            print(f"Gate: {gate:.2f}")
            print(f"Sample time (dt): {self.dt} minutes")

        pump_action = Action(basal=basal_rate, bolus=bolus_rate)

        step_result = self.core_env.step(pump_action)
        obs = step_result.observation
        reward = step_result.reward
        terminated = step_result.done
        truncated = False
        info = step_result.info or {}

        info.update(
            {
                "rbppo_gate": gate,
                "rbppo_bolus_event": bool(meal_units > 0.0),
                "rbppo_meal_units": meal_units,
                "rbppo_correction": predicted_correction,
                "rbppo_bolus_units": bolus_units,
                "rbppo_basal_u_per_min": basal_rate,
            }
        )

        if terminated and self.debug:
            print(
                f"Early Termination: Time={self.core_env.time}, CGM={obs.CGM:.1f}, Gate={gate:.2f}, IOB={self.iob:.2f}"
            )

        if obs.CGM < 40:
            if self.debug:
                print(f"SEVERE HYPO: Terminating episode at CGM={obs.CGM:.1f}")
            terminated = True

        self.cgm_history = self.core_env.CGM_hist
        self.iob = update_iob(self.dt, self.tau, self.iob, total_insulin_delivered)
        state = self._extract_state(obs)
        return state, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self.dt = float(getattr(self.core_env, "sample_time", 5.0))
        self.iob = 0.0
        self.cgm_history = self.core_env.CGM_hist
        self.prev_gate = 1.0
        state = self._extract_state(obs)
        return state, info
