from datetime import datetime, timedelta
from simglucose.simulation.env import T1DSimEnv
from simglucose.sensor.cgm import CGMSensor
from simglucose.actuator.pump import InsulinPump
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.simulation.scenario_gen import RandomScenario
from simglucose.simulation.sim_engine import SimObj, sim
import argparse
from pathlib import Path
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from ppo_controller import PPOController


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Test the PPOController",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained PPO model (.zip file)",
    )
    parser.add_argument(
        "--vecnorm_path",
        type=str,
        default=None,
        help="Path to the VecNormalize.pkl file (if used during training)",
    )
    parser.add_argument(
        "--patient_name",
        type=str,
        default="adult#010",
        help="Name of the patient model to use",
    )
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    patient_name = args.patient_name
    model_path = args.model_path
    vecnorm_path = args.vecnorm_path

    start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

    # Build a classic (non-Gym) simglucose env
    patient = T1DPatient.withName(patient_name, seed=1)
    sensor = CGMSensor.withName("Dexcom", seed=1)
    pump = InsulinPump.withName("Insulet")
    scen = RandomScenario(start_time=start, seed=100)
    env = T1DSimEnv(patient, sensor, pump, scen)

    # Mirror pump caps inside the controller (keeps actions realistic)
    pump_limits = {
        "max_basal": float(pump._params.get("max_basal", 3.0)),
        "max_bolus": float(pump._params.get("max_bolus", 75.0)),
        "max_correction": float(pump._params.get("max_correction", 12.0)),
    }

    ctrl = PPOController(
        model_path=model_path,
        patient_name=patient.name,
        pump_params=pump_limits,
        vecnorm_path=vecnorm_path,
        debug=True,  # optional; if trained with VecNormalize
    )

    # Simulate for 24h; set animate=False for headless runs
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)

    s = SimObj(env, ctrl, timedelta(days=1), animate=True, path=str(out_dir))
    results_df = sim(s)  # returns a pandas DataFrame
    csv_path = out_dir / "test_ppo_controller_results.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"[INFO] Saved raw results to {csv_path}")

    try:
        from simglucose.analysis.report import report

        report_dir = out_dir / "reports"
        report_dir.mkdir(exist_ok=True)
        report(results_df, str(report_dir))
        print(f"[INFO] Generated report in {report_dir}")
    except Exception as exc:
        print(f"[WARN] Could not generate report: {exc}")
