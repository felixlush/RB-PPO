# Simglucose PPO Controller

## Project Overview

This repository contains the code base for my Honours project on learning a fully closed-loop insulin controller for the **simglucose** Type 1 Diabetes simulator. The controller is a risk-bounded Proximal Policy Optimisation (RB-PPO) agent that learns basal delivery and safety-limited correction boluses while observing continuous glucose monitor (CGM) data, insulin-on-board (IOB) estimates, and meal information. Training runs are orchestrated with Stable-Baselines3 and PyTorch, and the learned policies can be evaluated either in bulk on the simulator or exported as a `PPOController` compatible with the `simglucose` framework.

## Repository Layout

- `train_ppo_v2.py` – main entry point for training or resuming an RB-PPO policy (supports deterministic bolus, no-bolus, and no-gate ablations via `--env_type`).
- `train_multi_seed*.py` – launch helpers that repeat training across multiple random seeds for reproducibility studies.
- `simglucose_env_wrapper*.py` – environment wrappers that reshape observations/actions, implement the insulin safety gate, and expose ablations (no bolus announcements or no correction gate).
- `train_utils.py` – environment factories, reward shaping (`RBRewardNormalized`), vector-normalisation utilities, and custom callbacks for glucose metrics, policy diagnostics, and evaluation.
- `ppo_controller.py` – deployment wrapper that loads a trained PPO policy plus VecNormalize statistics to act as a `simglucose` controller.
- `eval_scripts/` – scripts for offline evaluation, batch report generation, and comparison against classical controllers.
- `config/` – patient splits for train/eval cohorts and TensorBoard scalar definitions used during analysis.
- `test_scripts/` – pytest-based smoke tests for wrappers, reward shaping, and the PPO controller logic.
- `runs/`, `logs/`, `results/` – artefact folders for checkpoints, TensorBoard output, and CSV summaries (these grow quickly when running experiments).

## Getting Started

1. **Environment**  
   - Python ≥ 3.10.  
   - Recommended to create a virtual environment: `python -m venv .venv && source .venv/bin/activate`.  
   - Install the main dependencies (install CUDA-specific PyTorch wheels if needed):  

     ```bash
     pip install --upgrade pip
     pip install torch stable-baselines3 gymnasium simglucose numpy pandas tensorboard cloudpickle
     ```

   - The project also relies on `pkg_resources`, `matplotlib`, and `pytest` which will be pulled transitively or can be added explicitly.
2. **Simulator assets**  
   - `simglucose` ships with the patient Quest parameters and meal scenarios required by the wrappers. No additional datasets are needed.
3. **Folder hygiene**  
   - Keep `runs/` and `logs/` under version control ignore (large binary artefacts).  
   - Use meaningful `--run_name` strings to track experiments, e.g. `RB-PPO_seed108_safety_update`.

## Training Workflows

### Single run

```bash
python train_ppo_v2.py \
  --type fresh \
  --seed 108 \
  --steps 5_000_000 \
  --env_type deterministic_bolus \
  --run_name RB-PPO_seed108
```

Key flags:

- `--type` can be `fresh` or `resume` (resume expects `--ckpt` and `--vecnorm` paths).
- `--env_type` selects the wrapper variant (`deterministic_bolus`, `no_bolus`, `no_gate`) for the clinical ablation studies.
- `--save_freq` and `--eval_freq` control checkpointing and validation cadence.

Training writes checkpoints to `runs/<timestamp_run>/ckpts/`, evaluation-best models to `runs/<...>/best/`, and the final policy plus VecNormalize statistics to `runs/<...>/final/`. TensorBoard event files are placed in `logs/<timestamp_run_name>/`.

### Multi-seed sweeps

Use `train_multi_seed.py` (or the `_no_bolus`/`_no_gate` variants) to replicate experiments across the predefined patient-seeded cohorts:

```bash
python train_multi_seed.py --steps 5_000_000 --seeds 108 193 202 -- --env_type no_gate --debug
```

Everything after `--` is forwarded to `train_ppo_v2.py`. Each run is stamped with a descriptive name so that downstream aggregation scripts in `eval_scripts/` can discover them automatically.

## Evaluation and Reporting

- `eval_scripts/evaluate_model.py` loads a checkpoint plus VecNormalize stats and runs multi-patient rollouts, writing CSV summaries to `results/`.
- `eval_scripts/evaluate_batch.py` iterates across a folder of runs (useful for comparing seeds or ablation variants).
- `eval_scripts/concat_final_results.py` merges per-patient CSVs into a single table for figures.  
- Classical controller baselines (e.g., PID) are collected through `eval_scripts/evaluate_classic_controllers.py` for head-to-head comparisons.
- For in-simulator deployment, load the trained weights with `ppo_controller.PPOController` and register it with the `simglucose` experiment runner.

## Testing

Lightweight tests live in `test_scripts/` and can be executed with:

```bash
pytest test_scripts
```

They focus on wrapper behaviours (safety gate, meal logic) and the reward implementation. Extend these tests when adding new wrappers or controllers to keep regression coverage.

## Honours Project Notes

- All experiments share the train/eval patient cohorts defined in `config/patient_splits.py` to ensure fair comparison.
- Logged glucose metrics include risk, LBGI/HBGI, time-in-range, hypo/hyper percentages, and policy distribution stats; these provide the quantitative evidence reported in the thesis.
- Keep a record of the `config.json` emitted in each `runs/<...>/` directory. It captures random seeds, environment variants, and library versions, which is essential for reproducibility and for the methods chapter.
