import numpy as np
from simglucose.controller.base import Controller, Action
from stable_baselines3 import PPO
from datetime import datetime
import pandas as pd
import pkg_resources
from utils import (
    calculate_meal_units,
    calculate_normalised_trend,
    encode_time,
    update_iob,
    compute_safety_gate,
    calculate_insulin_delivered,
)
import cloudpickle


class PPOController(Controller):
    def __init__(
        self,
        model_path,
        patient_name,
        pump_params,
        vecnorm_path=None,
        debug=False,
        deterministic=True,
        device="auto",
    ):
        self.model = PPO.load(model_path, device=device)
        self.patient_name = patient_name
        # Now derive patients params from patient
        self.debug = debug
        self.CR = 15.0
        self.tdi = 50.0
        self.CF = 20.0
        self.dt = 5.0  # minutes per step
        self.tau = 180.0  # minutes (insulin action time)
        self.iob = 0.0
        self.cgm_history: list[float] = []
        self.last_cgm = 140.0
        self.prev_gate = 1.0
        self.init_time = None
        self.deterministic = deterministic

        if self.patient_name:
            try:
                # Get Patient Params from Quest
                CONTROL_QUEST = pkg_resources.resource_filename(
                    "simglucose", "params/Quest.csv"
                )
                quest_df = pd.read_csv(CONTROL_QUEST)
                if any(quest_df.Name.str.match(self.patient_name)):
                    quest = quest_df[quest_df.Name.str.match(self.patient_name)]
                    self.CR = quest.CR.values.item()
                    self.tdi = quest.TDI.values.item()
                    self.CF = quest.CF.values.item()
                else:
                    self.CR = 15  # Default fallback
                    self.tdi = 50.0
                    self.CF = 20.0
            except Exception:
                # If cant find the file keep the defaults
                pass

        if pump_params:
            self.max_basal_u_per_hr = float(pump_params.get("max_basal"))
            self.max_basal_u_per_min = float(self.max_basal_u_per_hr / 60.0)
            self.max_bolus = float(pump_params.get("max_bolus"))
            self.max_correction_per_hr = float(pump_params.get("max_correction"))
            self.max_correction_per_step = float(
                (self.max_correction_per_hr / 60.0) * self.dt
            )
        else:
            self.max_basal_u_per_hr = 35.0
            self.max_basal_u_per_min = self.max_basal_u_per_hr / 60.0
            self.max_bolus = 75.0
            self.max_correction_per_hr = 12.0
            self.max_correction_per_step = float(
                (self.max_correction_per_hr / 60.0) * self.dt
            )

        # Now deal with normalised observations for SB3 VecNormalise
        self._obs_mean = None
        self._obs_var = None
        self._obs_eps = 1e-8
        if vecnorm_path:
            try:
                with open(vecnorm_path, "rb") as f:
                    vec = cloudpickle.load(f)
                if hasattr(vec, "obs_rms") and vec.obs_rms is not None:
                    self._obs_mean = np.asarray(vec.obs_rms.mean, dtype=np.float32)
                    self._obs_var = np.asarray(vec.obs_rms.var, dtype=np.float32)
                    if self.debug:
                        print(f"Loaded VecNormalize stats from {vecnorm_path}")
            except FileNotFoundError:
                if self.debug:
                    print(f"[WARN] VecNormalize file not found at {vecnorm_path}")
            except Exception as exc:
                if self.debug:
                    print(f"[WARN] Failed to load VecNormalize stats: {exc}")

    def policy(self, observation, reward, done, **info):
        # get CGM value
        self.dt = float(info.get("sample_time", self.dt))
        current_cgm = float(getattr(observation, "CGM", observation))

        # add to history
        self.cgm_history.append(current_cgm)
        if len(self.cgm_history) > 12:
            self.cgm_history.pop(0)

        # convert time
        current_time = info.get("time", datetime.now())

        if self.init_time is None:
            self.init_time = current_time

        meal_cho = float(info.get("meal", 0.0)) if isinstance(info, dict) else 0.0

        # calculate meal units
        meal_units = calculate_meal_units(self.CR, meal_cho)

        # calculate trend
        trend = calculate_normalised_trend(self.cgm_history, self.dt)

        # convert time
        time_sin, time_cos = encode_time(current_time)

        state = np.array(
            [
                current_cgm,
                trend,
                self.iob,
                time_sin,
                time_cos,
                meal_units,
                self.CF,
                self.tdi,
            ],
            dtype=np.float32,
        )

        norms_obs = self._normalize_for_inference(state)

        action, _ = self.model.predict(norms_obs, deterministic=self.deterministic)
        action = np.asarray(action, dtype=np.float32)
        basal_cmd = float(np.clip(action[0], 0.0, self.max_basal_u_per_min))

        if meal_units > 0:
            max_negative = meal_units
        else:
            max_negative = 0.0

        # enforce only negative correction if meal event
        predicted_correction = float(action[1])
        if max_negative > 0 and predicted_correction < 0:
            predicted_correction = predicted_correction
        elif max_negative <= 0 and predicted_correction < 0:
            predicted_correction = 0.0
        else:
            predicted_correction = predicted_correction

        predicted_correction = float(
            np.clip(predicted_correction, -max_negative, self.max_correction_per_step)
        )

        gate = compute_safety_gate(
            self.iob, current_cgm, trend, self.tdi, self.CF, prev_gate=self.prev_gate
        )
        basal_rate = basal_cmd * gate

        total_insulin_delivered, bolus_rate, _ = calculate_insulin_delivered(
            meal_units,
            gate,
            predicted_correction,
            self.max_bolus,
            self.dt,
            basal_rate,
        )
        if self.debug:
            print(
                f"gate={gate:.2f} predicted_correction={predicted_correction} basal_cmd={basal_cmd} basal={basal_rate:.3f} bgm={current_cgm:.1f} trend={trend:.2f} iob={self.iob:.3f} meal_cho={meal_cho:.0f} meal_units={meal_units:.2f} bolus={bolus_rate:.3f} total_insulin={total_insulin_delivered:.3f}"
            )
        # update IOB and previous gate
        self.iob = update_iob(self.dt, self.tau, self.iob, total_insulin_delivered)
        self.prev_gate = gate

        return Action(basal=basal_rate, bolus=bolus_rate)

    def reset(self):
        self.iob = 0.0
        self.cgm_history.clear()
        self.prev_gate = 1.0
        self.init_time = None

    def _normalize_for_inference(self, obs_vec: np.ndarray) -> np.ndarray:
        if self._obs_mean is None or self._obs_var is None:
            return obs_vec
        return (obs_vec - self._obs_mean) / np.sqrt(self._obs_var + self._obs_eps)
