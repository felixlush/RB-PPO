import os
import sys
from typing import List

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from train_utils import make_env  # noqa: E402


class BasalTracker(BaseCallback):
    """Collect basal commands during learning to verify exploration."""

    def __init__(self, log_every: int = 1024, verbose: int = 0):
        super().__init__(verbose)
        self.log_every = log_every
        self._raw_buffer: List[float] = []
        self._gated_buffer: List[float] = []
        self._gate_buffer: List[float] = []
        self.raw_history: List[float] = []
        self.gated_history: List[float] = []
        self.gate_coeff_history: List[float] = []

    def _record_stats(self, name: str, data: List[float]) -> None:
        if not data:
            return
        arr = np.asarray(data, dtype=np.float32)
        min_a, max_a = float(arr.min()), float(arr.max())
        std_a = float(arr.std())
        unique_count = float(np.unique(np.round(arr, 4)).size)
        self.logger.record(f"{name}/min", min_a)
        self.logger.record(f"{name}/max", max_a)
        self.logger.record(f"{name}/std", std_a)
        self.logger.record(f"{name}/unique_count", unique_count)
        if self.verbose:
            print(
                f"[BasalTracker:{name}] range={min_a:.4f}–{max_a:.4f} "
                f"std={std_a:.4f} unique≈{int(unique_count)}"
            )

    def _flush(self) -> None:
        if self._raw_buffer:
            self.raw_history.extend(self._raw_buffer)
            self._record_stats("basal_raw", self._raw_buffer)
        if self._gated_buffer:
            self.gated_history.extend(self._gated_buffer)
            self._record_stats("basal_gated", self._gated_buffer)
        if self._gate_buffer:
            self.gate_coeff_history.extend(self._gate_buffer)
            self._record_stats("gate_coeff", self._gate_buffer)
        self._raw_buffer.clear()
        self._gated_buffer.clear()
        self._gate_buffer.clear()

    def _on_step(self) -> bool:
        actions = self.locals.get("actions")
        infos = self.locals.get("infos")
        if actions is not None:
            self._raw_buffer.extend(actions[:, 0].tolist())
        if infos:
            for info in infos:
                if isinstance(info, dict):
                    gated = info.get("rbppo_basal_u_per_min")
                    if gated is not None:
                        self._gated_buffer.append(float(gated))
                    gate_coeff = info.get("rbppo_gate")
                    if gate_coeff is not None:
                        self._gate_buffer.append(float(gate_coeff))
        if len(self._raw_buffer) >= self.log_every:
            self._flush()
        return True

    def _on_training_end(self) -> None:
        self._flush()

    def plot(self, save_path: str = None) -> None:
        if not (self.gated_history and self.gate_coeff_history):
            if self.verbose:
                print("[BasalTracker] Nothing to plot.")
            return
        steps = np.arange(len(self.gated_history))
        fig, ax1 = plt.subplots(figsize=(10, 4))
        ax1.plot(
            steps, self.gated_history, label="Basal delivered (U/min)", color="tab:blue"
        )
        ax1.set_ylabel("Delivered basal (U/min)", color="tab:blue")
        ax1.tick_params(axis="y", labelcolor="tab:blue")
        ax1.set_xlabel("Training step samples")

        ax2 = ax1.twinx()
        ax2.plot(
            steps,
            self.gate_coeff_history[: len(steps)],
            label="Gate coefficient",
            color="tab:orange",
            alpha=0.7,
        )
        ax2.set_ylabel("Gate coefficient", color="tab:orange")
        ax2.tick_params(axis="y", labelcolor="tab:orange")

        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=200)
        else:
            plt.show()
        plt.close(fig)


def evaluate_policy(
    model: PPO,
    episodes: int = 1,
    deterministic: bool = True,
    seed: int = 123,
    save_path: str = None,
) -> None:
    eval_env = make_env("adolescent#001", seed=seed, debug=False)()
    raw_history: List[float] = []
    clipped_history: List[float] = []
    delivered_history: List[float] = []
    gate_history: List[float] = []

    for ep in range(episodes):
        obs, info = eval_env.reset(seed=seed + ep)
        done = False
        truncated = False
        while not (done or truncated):
            obs_arr = np.asarray(obs, dtype=np.float32)
            action, _ = model.predict(obs_arr, deterministic=deterministic)
            action = np.asarray(action, dtype=np.float32).reshape(-1)
            raw_history.append(float(action[0]))
            clipped = np.clip(
                action, eval_env.action_space.low, eval_env.action_space.high
            )
            clipped_history.append(float(clipped[0]))
            obs, reward, done, truncated, info = eval_env.step(clipped)
            delivered_history.append(float(info.get("rbppo_basal_u_per_min", np.nan)))
            gate_history.append(float(info.get("rbppo_gate", np.nan)))

    eval_env.close()

    if not delivered_history:
        print("[Eval] No data collected.")
        return

    print(
        "[Eval] raw range {:.4f}–{:.4f}, clipped range {:.4f}–{:.4f}, delivered range {:.4f}–{:.4f}".format(
            float(np.nanmin(raw_history)),
            float(np.nanmax(raw_history)),
            float(np.nanmin(clipped_history)),
            float(np.nanmax(clipped_history)),
            float(np.nanmin(delivered_history)),
            float(np.nanmax(delivered_history)),
        )
    )
    print(
        "[Eval] gate coeff range {:.4f}–{:.4f}, mean {:.4f}".format(
            float(np.nanmin(gate_history)),
            float(np.nanmax(gate_history)),
            float(np.nanmean(gate_history)),
        )
    )

    steps = np.arange(len(delivered_history))
    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(
        steps, delivered_history, color="tab:blue", label="Delivered basal (U/min)"
    )
    ax1.set_ylabel("Delivered basal (U/min)", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.set_xlabel("Evaluation step")

    ax2 = ax1.twinx()
    ax2.plot(
        steps, gate_history, color="tab:orange", alpha=0.7, label="Gate coefficient"
    )
    ax2.set_ylabel("Gate coefficient", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200)
    else:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    # Single-env DummyVecEnv keeps everything in-process and easy to inspect
    env = DummyVecEnv([make_env("adolescent#001", seed=42, debug=False)])

    policy_kwargs = dict(net_arch=[256, 256], log_std_init=-3.0)

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=576,
        batch_size=504,
        n_epochs=8,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.15,
        ent_coef=0.001,
        verbose=1,
        policy_kwargs=policy_kwargs,
    )

    tracker = BasalTracker(log_every=512, verbose=1)
    model.learn(total_timesteps=4_096, callback=tracker, progress_bar=True)
    tracker.plot(save_path="basal_gate_plot.png")
    evaluate_policy(
        model,
        episodes=1,
        deterministic=True,
        seed=123,
        save_path="eval_basal_gate_plot.png",
    )
