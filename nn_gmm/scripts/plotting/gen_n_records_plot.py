"""Script for generating number of records spatial maps"""
import src.plotting_funcs
from pathlib import Path

import nn_gmm

plot_items_ffp = (
    "/home/claudy/dev/work/code/visualization/visualization/gmt/plot_items.py"
)

train_data_dirs = [
    "/home/cbs51/dev/work/data/nn_gmm/input_data/sample_files/train",
]
val_data_dirs = [
    "/home/cbs51/dev/work/data/nn_gmm/input_data/sample_files/val",
]
data_dirs = train_data_dirs + val_data_dirs
output_dir = Path("/home/claudy/dev/work/data/nn_gmm/results/keep/n_records")

print(f"Generating all records map")
src.plotting_funcs.plot_n_records_map(data_dirs, plot_items_ffp, str(output_dir / "n_records"))

print(f"Generating training records map")
src.plotting_funcs.plot_n_records_map(
    train_data_dirs,
    plot_items_ffp,
    str(output_dir / "n_records_train"),
    title="Training:Number-of-records",
)

print(f"Generating validation records map")
src.plotting_funcs.plot_n_records_map(
    val_data_dirs,
    plot_items_ffp,
    str(output_dir / "n_records_val"),
    title="Validation:Number-of-records",
)
