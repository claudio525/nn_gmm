import os
import glob
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def get_event_ids(dirs):
    event_ffps = np.concatenate([list(glob.glob(os.path.join(cur_dir, "*.tfrecord"))) for cur_dir in
                    dirs])
    return np.stack(np.char.split(np.stack(np.char.split(event_ffps, "/"))[:, -1], "."))[:,
    0]

def get_closest_event_ind(mag_array, mag_values):
    return [np.argmin(np.abs(mag_array - mag_val)) for mag_val in mag_values]

source_params_dir = Path("/home/cbs51/dev/work/data/nn_gmm/source_params")
train_dirs = [
    "/home/cbs51/dev/work/data/nn_gmm/sample_files/cybershake_v20p4/train",
    "/home/cbs51/dev/work/data/nn_gmm/sample_files/validation_v20p5p8/train",
    "/home/cbs51/dev/work/data/nn_gmm/sample_files/validation_v20p6p0/train",
]
val_dirs = [
    "/home/cbs51/dev/work/data/nn_gmm/sample_files/cybershake_v20p4/val",
    "/home/cbs51/dev/work/data/nn_gmm/sample_files/validation_v20p5p8/val",
    "/home/cbs51/dev/work/data/nn_gmm/sample_files/validation_v20p6p0/val",
]


event_dfs_dict = {os.path.basename(cur_file).split(".")[0]: pd.read_csv(cur_file, index_col="fault") for cur_file in source_params_dir.glob("*.csv")}
events_df = pd.concat(list(event_dfs_dict.values()))

train_events = get_event_ids(train_dirs)
val_events = get_event_ids(val_dirs)


percentiles = [10, 25, 45, 55, 75, 90]
percentile_values = np.percentile(events_df.mag, percentiles)

plt.figure(figsize=(18, 13.5))

for event_name, cur_event_df in event_dfs_dict.items():
    plt.hist(cur_event_df.mag, bins=15, label=event_name)

for p in percentile_values:
    plt.axvline(p, c='k')

plt.xlabel("mag")
plt.ylabel("count")
plt.grid(linestyle="--", linewidth=0.25)
plt.legend()


train_rep_events = events_df.loc[train_events].iloc[get_closest_event_ind(events_df.loc[train_events].mag.values, percentile_values)]
val_rep_events = events_df.loc[val_events].iloc[get_closest_event_ind(events_df.loc[val_events].mag.values, percentile_values)]

for (_, train_event), (_, val_event) in zip(train_rep_events.iterrows(), val_rep_events.iterrows()):
    plt.axvline(train_event.mag, c="g")
    plt.axvline(val_event.mag, c="r")

plt.show()

print("Training rep events: ", list(train_rep_events.index.values))
print("Validation rep events: ", list(val_rep_events.index.values))

exit()
