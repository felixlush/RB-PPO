import argparse
import json
import os
import platform
import random
import subprocess
from datetime import datetime

import numpy as np
import stable_baselines3 as sb3
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    StopTrainingOnNoModelImprovement,
)
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import VecNormalize
from config.patient_splits import patients_eval, patients_train
from train_utils import (
    GlucoseMetricsCallback,
    PolicyStatsCallback,
    create_eval_env,
    create_train_env,
    CustomEvalCallback,
)


def set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    set_random_seed(seed)
    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def create_run_config(args, train_env_seeds, eval_env_seeds) -> dict:
    args_dict = {k: getattr(args, k) for k in vars(args)}
    config = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "args": args_dict,
        "seeds": {
            "global": int(args.seed),
            "train_env": list(train_env_seeds),
            "eval_env": list(eval_env_seeds),
            "eval_env_base": eval_env_seeds[0] if eval_env_seeds else None,
        },
        "package_versions": {
            "python": platform.python_version(),
            "stable_baselines3": sb3.__version__,
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
    }
    return config


def save_run_config(run_dir: str, config: dict, overwrite: bool = True) -> None:
    os.makedirs(run_dir, exist_ok=True)
    path = os.path.join(run_dir, "config.json")
    if not overwrite and os.path.exists(path):
        return
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2, sort_keys=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the RB-PPO model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--type",
        type=str,
        default="fresh",
        help="fresh or resume: defines if starting training from scratch or resuming",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="runs/ckpts/ppo_5000000_steps.zip",
        help="Latest checkpoint file path",
    )
    parser.add_argument(
        "--vecnorm",
        type=str,
        default="runs/vecnorm.pkl",
        help="Path for the VecNormalise.pkl file",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Global random seed for reproducibility",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=1_000_000,
        help="Number of time steps (or additional steps) to be used in training",
    )
    parser.add_argument("--run_dir", type=str, help="Path to save output and models")
    parser.add_argument(
        "--run_name",
        type=str,
        default="RB-PPO",
        help="Run name for TB logging - date will be automatically appended",
    )
    parser.add_argument(
        "--debug",
        default=False,
        action="store_true",
        help="Enable debug mode",
    )
    parser.add_argument(
        "--save_freq",
        type=int,
        default=100_000,
        help="training chunk size for each learning call",
    )
    parser.add_argument(
        "--eval_freq",
        type=int,
        default=100_000,
        help="Frequency of evaluations during training",
    )
    parser.add_argument(
        "--env_type",
        type=str,
        default="deterministic_bolus",
        choices=("deterministic_bolus", "no_bolus", "no_gate"),
        help="Environment wrapper variant for training (clinical ablations).",
    )
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    training_type = args.type
    ckpt_path = args.ckpt
    vecnorm_path = args.vecnorm
    max_timesteps = args.steps
    run_dir = args.run_dir
    seed = int(args.seed)
    run_name = args.run_name
    debug = args.debug
    save_freq = int(args.save_freq)
    eval_freq = int(args.eval_freq)
    env_type = args.env_type

    set_global_seeds(seed)

    train_env_seeds = [seed + i for i in range(len(patients_train))]
    eval_env_base = 1000
    eval_env_seeds = [eval_env_base + i for i in range(len(patients_eval))]

    now = datetime.now().strftime("%Y%m%d-%H%M%S")

    if not run_name:
        raise ValueError("Run name must be specified when starting training")
    log_dir = os.path.join("logs", f"{now}_{run_name}")
    os.makedirs(log_dir, exist_ok=True)

    # Set up directories for logging and saving models
    # Now turn environment into SubprocVecEnv and VecNormalie

    if training_type == "fresh":
        if not run_dir:
            run_dir = os.path.join("runs", f"{now}_{run_name}")
        train_env = create_train_env(seed=seed, debug=debug, env_type=env_type)
        eval_env = create_eval_env(train_env, debug=debug, env_type=env_type)

        # Set policy arguments, reduce log std to reduce large random jumps at start and encourage algo to move mean from 0
        policy_kwargs = dict(net_arch=[256, 256], log_std_init=-3.0)

        model = PPO(
            "MlpPolicy",
            train_env,
            learning_rate=3e-4,
            n_steps=576,
            batch_size=576,
            n_epochs=8,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.15,
            ent_coef=0.001,
            verbose=1,
            tensorboard_log=log_dir,
            policy_kwargs=policy_kwargs,
            use_sde=True,
            sde_sample_freq=1,
        )
        model.set_random_seed(seed)
    if training_type == "resume":
        if not ckpt_path:
            raise ValueError("Checkpoint path must be specified when resuming training")
        if not run_dir:
            run_dir = f"runs/{now}_resumed"
        if not vecnorm_path:
            raise ValueError(
                "VecNormalize path must be specified when resuming training"
            )
        if not os.path.exists(vecnorm_path):
            raise FileNotFoundError(f"VecNormalize stats not found at {vecnorm_path}")

        base_env = create_train_env(
            seed=seed, debug=debug, use_vecnorm=False, env_type=env_type
        )
        train_env = VecNormalize.load(vecnorm_path, base_env)
        train_env.training = True
        train_env.norm_reward = False

        eval_env = create_eval_env(train_env, debug=debug, env_type=env_type)

        model = PPO.load(
            ckpt_path, env=train_env, device="auto", print_system_info=True
        )
        model.set_random_seed(seed)

    args.run_dir = run_dir
    run_config = create_run_config(args, train_env_seeds, eval_env_seeds)

    os.makedirs(run_dir, exist_ok=True)
    save_run_config(run_dir, run_config, overwrite=training_type == "fresh")

    final_dir = os.path.join(run_dir, "final")
    ckpt_dir = os.path.join(run_dir, "ckpts")

    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(final_dir, exist_ok=True)

    # Set up callbacks
    ckpt_freq = max(save_freq // train_env.num_envs, 1)
    ckpt = CheckpointCallback(
        save_freq=ckpt_freq,
        save_path=ckpt_dir,
        name_prefix="ppo",
        save_vecnormalize=True,
    )

    stop_train = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=15, min_evals=20, verbose=1
    )

    eval_freq = max(eval_freq // train_env.num_envs, 1)
    eval_cb = CustomEvalCallback(
        train_env,
        eval_env,
        n_eval_episodes=20,
        eval_freq=eval_freq,
        deterministic=True,
        render=False,
        best_model_save_path=os.path.join(run_dir, "best"),
    )

    metrics_cb = GlucoseMetricsCallback(log_every_steps=10_000, verbose=0)

    policy_stats_freq = max(10_000 // train_env.num_envs, 1)
    policy_cb = PolicyStatsCallback(log_every_steps=policy_stats_freq)

    callbacklist = CallbackList([ckpt, eval_cb, metrics_cb, policy_cb])

    learn_kwargs = dict(
        total_timesteps=max_timesteps, callback=callbacklist, progress_bar=True
    )

    if training_type == "resume":
        learn_kwargs["reset_num_timesteps"] = False

    model.learn(**learn_kwargs)

    model.save(os.path.join(final_dir, f"final_model_{model.num_timesteps}_steps.zip"))
    train_env.save(os.path.join(final_dir, "final_model_vecnorm.pkl"))
