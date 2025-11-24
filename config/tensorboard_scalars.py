from tensorboard.backend.event_processing import event_accumulator
import pandas as pd

log_dir = "logs/20251022-132234_RB-PPO/PPO_1"

event_acc = event_accumulator.EventAccumulator(log_dir)
event_acc.Reload()

rows = []

for tag in event_acc.Tags()["scalars"]:
    scalar_events = event_acc.Scalars(tag)
    for event in scalar_events:
        rows.append(
            {
                "wall_time": event.wall_time,
                "step": event.step,
                "tag": tag,
                "value": event.value,
            }
        )

df = pd.DataFrame(rows)
df = df.sort_values(by=["step", "tag"]).reset_index(drop=True)
df.to_csv("tensorboard_scalars_20251022-132234_Model_4.csv", index=False)
