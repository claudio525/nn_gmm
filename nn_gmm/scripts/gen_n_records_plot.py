"""Script for generating number of records spatial maps"""
from pathlib import Path

import nn_gmm

plot_items_ffp = (
    "/home/claudy/dev/work/code/visualization/visualization/gmt/plot_items.py"
)

train_data_dirs = [
    "/home/cbs51/dev/work/data/nn_gmm/sample_files/cybershake_v20p4/train",
    "/home/cbs51/dev/work/data/nn_gmm/sample_files/validation_v20p5p8/train",
    "/home/cbs51/dev/work/data/nn_gmm/sample_files/validation_v20p6p0/train",
]
val_data_dirs = [
    "/home/cbs51/dev/work/data/nn_gmm/sample_files/cybershake_v20p4/val",
    "/home/cbs51/dev/work/data/nn_gmm/sample_files/validation_v20p5p8/val",
    "/home/cbs51/dev/work/data/nn_gmm/sample_files/validation_v20p6p0/val",
]
data_dirs = train_data_dirs + val_data_dirs
output_dir = Path("/home/claudy/dev/work/data/nn_gmm/results/keep/n_records")

print(f"Generating all records map")
nn_gmm.plot_n_records_map(data_dirs, plot_items_ffp, str(output_dir / "n_records"))

print(f"Generating training records map")
nn_gmm.plot_n_records_map(
    train_data_dirs,
    plot_items_ffp,
    str(output_dir / "n_records_train"),
    title="Training:Number-of-records",
)

print(f"Generating validation records map")
nn_gmm.plot_n_records_map(
    val_data_dirs,
    plot_items_ffp,
    str(output_dir / "n_records_val"),
    title="Validation:Number-of-records",
)
