from datetime import datetime, timedelta
from simglucose.simulation.env import T1DSimEnv
from simglucose.sensor.cgm import CGMSensor
from simglucose.actuator.pump import InsulinPump
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.simulation.scenario_gen import RandomScenario
from simglucose.simulation.sim_engine import SimObj, sim
from simglucose.controller.pid_ctrller import PIDController
from pathlib import Path
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

patient_name = "adult#010"
start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

patient = T1DPatient.withName(patient_name, seed=1)
sensor = CGMSensor.withName("Dexcom", seed=1)
insulin_pump = InsulinPump.withName("Insulet")
scen = RandomScenario(start_time=start, seed=2)
env = T1DSimEnv(patient, sensor, insulin_pump, scen)
ctrl = PIDController(P=0.001, I=0.00001, D=0.001, target=140)

out_dir = Path("results")
out_dir.mkdir(exist_ok=True)

s = SimObj(env, ctrl, timedelta(days=1), animate=True, path=str(out_dir))
results_df = sim(s)
csv_path = out_dir / f"{patient_name}_pid_results.csv"
results_df.to_csv(csv_path, index=False)
print(f"Results saved to {csv_path}")
