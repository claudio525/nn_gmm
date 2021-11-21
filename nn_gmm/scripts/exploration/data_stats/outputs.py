#%% Imports
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import nn_gmm

#%% Load data
tf_record_ffp = Path("/mnt/data/work/data/nn_gmm/input_data/sample_files/train/AlpineF2K.tfrecord")

data_df = nn_gmm.load_tfrecord(str(tf_record_ffp), nn_gmm.load_feature_details(tf_record_ffp.parent))

#%% PGA
plt.figure(figsize=(16, 10), dpi=200)
plt.hist(data_df.PGA, bins=25)
plt.xlabel("ln(PGA)")
plt.ylabel("Count")
plt.grid(alpha=0.6, linestyle="--", linewidth=0.5)
plt.tight_layout()
plt.show()

#%% PGA -- Standard
plt.figure(figsize=(16, 10), dpi=200)
plt.hist((data_df.PGA - data_df.PGA.mean()) / data_df.PGA.std(), bins=25)
plt.xlabel("ln(PGA)")
plt.ylabel("Count")
plt.grid(alpha=0.6, linestyle="--", linewidth=0.5)
plt.tight_layout()
plt.show()



