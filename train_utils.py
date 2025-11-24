from stable_baselines3.common.utils import obs_as_tensor
from simglucose_env_wrapper import SimglucoseWrapper
from simglucose_env_wrapper_no_bolus import SimglucoseWrapperNoBolus
from simglucose_env_wrapper_no_gate import SimglucoseWrapperNoSafety
from simglucose.envs import T1DSimGymnaisumEnv
from gymnasium.wrappers import TimeLimit
from pos_reward_func import RBRewardNormalized
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback
import os, copy, numpy as np
from config.patient_splits import patients_train, patients_eval
from itertools import islice
from stable_baselines3.common.vec_env import sync_envs_normalization
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy


def make_env(patient, env_type, seed=None, debug=False):
    def _init():
        env = T1DSimGymnaisumEnv(patient_name=patient)
        if env_type == "deterministic_bolus":
            env = SimglucoseWrapper(env, debug=debug)
        elif env_type == "no_bolus":
            env = SimglucoseWrapperNoBolus(env, debug=debug)
        elif env_type == "no_gate":
            env = SimglucoseWrapperNoSafety(env, debug=debug)
        else:
            raise ValueError(f"Unknown env_type: {env_type}")
        env = TimeLimit(env, max_episode_steps=288)
        env = RBRewardNormalized(env)
        # env = Monitor(env)
        if seed is not None:
            env.reset(seed=seed)
        return env

    return _init


def sync_vecnormalize(src: VecNormalize, dst: VecNormalize):
    dst.obs_rms = copy.deepcopy(src.obs_rms)


class GlucoseMetricsCallback(BaseCallback):
    """
    Logs custom metrics from env 'info':
        - risk, lbgi, hbgi (means)
        - time-in-range % (70-180) via CGM from infos when available
        - mean insulin per step
    """

    def __init__(self, log_every_steps=10_000, verbose=0):
        super().__init__(verbose)
        self.log_every_steps = log_every_steps
        self.buffer = {"risk": [], "lbgi": [], "hbgi": [], "cgm": []}
        self.last_logged_step = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if not isinstance(info, dict):
                continue
            if "risk" in info:
                self.buffer["risk"].append(float(info["risk"]))
            if "lbgi" in info:
                self.buffer["lbgi"].append(float(info["lbgi"]))
            if "hbgi" in info:
                self.buffer["hbgi"].append(float(info["hbgi"]))
            if "bg" in info:
                self.buffer["cgm"].append(float(info["bg"]))  # or CGM if provided

        if self.num_timesteps - self.last_logged_step >= self.log_every_steps:

            def mean_or_nan(x):
                return float(np.mean(x)) if len(x) else float("nan")

            tir = (
                float(np.mean([(70.0 <= v <= 180.0) for v in self.buffer["cgm"]]))
                if self.buffer["cgm"]
                else float("nan")
            )

            pct_low = (
                float(np.mean([v < 70.0 for v in self.buffer["cgm"]]))
                if self.buffer["cgm"]
                else float("nan")
            )
            pct_high = (
                float(np.mean([v > 180.0 for v in self.buffer["cgm"]]))
                if self.buffer["cgm"]
                else float("nan")
            )

            self.logger.record("glucose/risk_mean", mean_or_nan(self.buffer["risk"]))
            self.logger.record("glucose/lbgi_mean", mean_or_nan(self.buffer["lbgi"]))
            self.logger.record("glucose/hbgi_mean", mean_or_nan(self.buffer["hbgi"]))
            self.logger.record("glucose/tir_70_180", tir)
            self.logger.record("glucose/percent_low", pct_low)
            self.logger.record("glucose/percent_high", pct_high)
            self.logger.record("glucose/cgm_mean", mean_or_nan(self.buffer["cgm"]))
            # self.logger.record("glucose/insulin_mean", mean_or_nan(self.buffer["ins"]))
            # clear buffer
            for k in self.buffer:
                self.buffer[k].clear()
            self.last_logged_step = self.num_timesteps
        return True


# ---------- Save VecNormalize with checkpoints ----------
class SaveVecNormCallback(BaseCallback):
    def __init__(self, venv: VecNormalize, save_path: str, save_freq: int = 100_000):
        super().__init__()
        self.venv = venv
        self.save_path = save_path
        self.save_freq = save_freq
        self.last_log_timestep = 0
        os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.num_timesteps - self.last_log_timestep >= self.save_freq:
            self.venv.save(os.path.join(self.save_path, "vecnorm.pkl"))
            self.last_log_timestep = self.num_timesteps
        return True

    def _on_training_end(self) -> None:
        self.venv.save(os.path.join(self.save_path, "vecnorm.pkl"))


class PolicyStatsCallback(BaseCallback):
    def __init__(self, log_every_steps=10_000):
        super().__init__()
        self.every = log_every_steps

    def _on_step(self) -> bool:
        if self.num_timesteps % self.every == 0:
            obs_tensor = obs_as_tensor(
                self.locals["rollout_buffer"].observations[-1], self.model.device
            )
            dist = self.model.policy.get_distribution(obs_tensor)
            basal_mean = dist.distribution.mean[:, 0].mean()
            basal_std = dist.distribution.stddev[:, 0].mean()
            self.logger.record("policy/basal_mean", float(basal_mean))
            self.logger.record("policy/basal_std", float(basal_std))
        return True


def build_env_factories(
    reward_func, seed=None, debug=False, env_type="deterministic_bolus"
):
    """
    Build env factory lists with optional deterministic seeding.
    """
    if seed is None:
        train_fns = [make_env(p, env_type, debug=debug) for p in patients_train]
        eval_fns = [make_env(p, env_type, debug=debug) for p in patients_eval]
        return train_fns, eval_fns
    train_fns = [
        make_env(p, env_type, seed=seed + i, debug=debug)
        for i, p in enumerate(patients_train)
    ]
    eval_fns = [
        make_env(p, env_type, seed=seed + 10_000 + i, debug=debug)
        for i, p in enumerate(patients_eval)
    ]
    return train_fns, eval_fns


def next_batch(pcycle, k):
    return list(islice(pcycle, k))


def create_eval_env(train_env, debug=False, env_type="deterministic_bolus"):
    eval_env = [
        make_env(
            p,
            env_type,
            seed=1000 + i,
            debug=debug,
        )
        for i, p in enumerate(patients_eval)
    ]
    eval_env = SubprocVecEnv(eval_env)
    eval_env = VecMonitor(eval_env)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)
    eval_env.training = False
    sync_envs_normalization(train_env, eval_env)
    return eval_env


def create_train_env(
    seed=42, debug=False, use_vecnorm=True, env_type="deterministic_bolus"
):
    train_env_fns = [
        make_env(p, env_type, seed=seed + i, debug=debug)
        for i, p in enumerate(patients_train)
    ]
    train_env = SubprocVecEnv(train_env_fns)
    train_env = VecMonitor(train_env)
    if use_vecnorm:
        train_env = VecNormalize(
            train_env,
            norm_obs=True,
            norm_reward=False,  # RBRewardNormalized returns 0..1
            training=True,
        )
    return train_env


class CustomEvalCallback(EvalCallback):
    """Eval callback that keeps VecNormalize stats aligned and saves them with best model."""

    def __init__(self, train_env, eval_env, *args, **kwargs):
        best_dir = kwargs.get("best_model_save_path", "")
        super().__init__(eval_env, *args, **kwargs)
        self.train_env = train_env
        self.best_vecnorm_path = (
            os.path.join(best_dir, "vecnorm.pkl") if best_dir else None
        )
        if best_dir:
            os.makedirs(best_dir, exist_ok=True)
        if self.best_vecnorm_path:
            os.makedirs(os.path.dirname(self.best_vecnorm_path), exist_ok=True)
        self.best_mean_tir = float("-inf")
        self.last_mean_tir = float("nan")
        self._init_eval_metric_buffers()

    def _init_eval_metric_buffers(self) -> None:
        n_envs = getattr(self.eval_env, "num_envs", 1)
        self._eval_counts_total = np.zeros(n_envs, dtype=np.int32)
        self._eval_counts_in_range = np.zeros(n_envs, dtype=np.int32)
        self._eval_counts_low = np.zeros(n_envs, dtype=np.int32)
        self._eval_counts_high = np.zeros(n_envs, dtype=np.int32)
        self._eval_episode_tir: list[float] = []
        self._eval_episode_low: list[float] = []
        self._eval_episode_high: list[float] = []

    def _reset_eval_metric_buffers(self) -> None:
        self._init_eval_metric_buffers()

    def _eval_metric_callback(self, locals_: dict, globals_: dict) -> None:
        # Preserve default success-rate logging
        self._log_success_callback(locals_, globals_)

        env_idx = locals_.get("i", 0)
        done = bool(locals_.get("done", False))
        info = locals_.get("info") or {}

        bg = info.get("rbppo_bg_mgdl", info.get("bg"))
        if bg is not None:
            bg_val = float(bg)
            self._eval_counts_total[env_idx] += 1
            if bg_val < 70.0:
                self._eval_counts_low[env_idx] += 1
            elif bg_val > 180.0:
                self._eval_counts_high[env_idx] += 1
            else:
                self._eval_counts_in_range[env_idx] += 1

        if done and self._eval_counts_total[env_idx] > 0:
            total = self._eval_counts_total[env_idx]
            self._eval_episode_tir.append(self._eval_counts_in_range[env_idx] / total)
            self._eval_episode_low.append(self._eval_counts_low[env_idx] / total)
            self._eval_episode_high.append(self._eval_counts_high[env_idx] / total)
            self._eval_counts_total[env_idx] = 0
            self._eval_counts_in_range[env_idx] = 0
            self._eval_counts_low[env_idx] = 0
            self._eval_counts_high[env_idx] = 0

    def _on_step(self) -> bool:
        continue_training = True
        should_eval = self.eval_freq > 0 and (self.n_calls % self.eval_freq == 0)

        if should_eval:
            if self.model.get_vec_normalize_env() is not None:
                try:
                    sync_envs_normalization(self.train_env, self.eval_env)
                except AttributeError as err:
                    raise AssertionError(
                        "Training and eval env are not wrapped the same way."
                    ) from err

            self._reset_eval_metric_buffers()
            self._is_success_buffer = []

            episode_rewards, episode_lengths = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=self.warn,
                callback=self._eval_metric_callback,
            )

            if self.log_path is not None:
                assert isinstance(episode_rewards, list)
                assert isinstance(episode_lengths, list)
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(episode_rewards)
                self.evaluations_length.append(episode_lengths)

                kwargs = {}
                if len(self._is_success_buffer) > 0:
                    self.evaluations_successes.append(self._is_success_buffer)
                    kwargs = dict(successes=self.evaluations_successes)

                np.savez(
                    self.log_path,
                    timesteps=self.evaluations_timesteps,
                    results=self.evaluations_results,
                    ep_lengths=self.evaluations_length,
                    **kwargs,
                )

            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = (
                np.mean(episode_lengths),
                np.std(episode_lengths),
            )
            self.last_mean_reward = float(mean_reward)
            mean_tir = (
                float(np.mean(self._eval_episode_tir))
                if self._eval_episode_tir
                else float("nan")
            )
            self.last_mean_tir = mean_tir

            if self.verbose >= 1:
                print(
                    f"Eval num_timesteps={self.num_timesteps}, "
                    f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}"
                )
                print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")

            self.logger.record("eval/mean_reward", float(mean_reward))
            self.logger.record("eval/mean_ep_length", mean_ep_length)

            if self._eval_episode_tir:
                self.logger.record(
                    "eval/tir_70_180", float(np.mean(self._eval_episode_tir))
                )
                self.logger.record(
                    "eval/percent_low", float(np.mean(self._eval_episode_low))
                )
                self.logger.record(
                    "eval/percent_high", float(np.mean(self._eval_episode_high))
                )
            if not np.isnan(mean_tir):
                self.logger.record("eval/mean_tir_70_180", mean_tir)

            if len(self._is_success_buffer) > 0:
                success_rate = np.mean(self._is_success_buffer)
                if self.verbose >= 1:
                    print(f"Success rate: {100 * success_rate:.2f}%")
                self.logger.record("eval/success_rate", success_rate)

            self.logger.record(
                "time/total_timesteps", self.num_timesteps, exclude="tensorboard"
            )
            self.logger.dump(self.num_timesteps)

            improvement_metric = None
            if not np.isnan(mean_tir):
                if mean_tir > self.best_mean_tir:
                    improvement_metric = ("tir", mean_tir)
            elif mean_reward > self.best_mean_reward:
                improvement_metric = ("reward", float(mean_reward))

            if improvement_metric is not None:
                metric_name, metric_value = improvement_metric
                if self.verbose >= 1:
                    print(f"New best mean {metric_name}!")
                if self.best_model_save_path is not None:
                    self.model.save(
                        os.path.join(self.best_model_save_path, "best_model")
                    )
                if self.best_vecnorm_path:
                    self.train_env.save(self.best_vecnorm_path)
                if metric_name == "tir":
                    self.best_mean_tir = metric_value
                else:
                    self.best_mean_reward = metric_value
                if self.callback_on_new_best is not None:
                    continue_training = self.callback_on_new_best.on_step()

            if self.callback is not None:
                continue_training = continue_training and self._on_event()

        return continue_training
