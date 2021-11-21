#%% Imports
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import nn_gmm

#%% Load data
site_params_ffp = "/mnt/data/work/data/nn_gmm/input_data/site_params/site_params.csv"
distance_db_ffp = Path("/mnt/data/work/data/nn_gmm/input_data/distance_data/cybershake_v20p4.db")

site_df = pd.read_csv(site_params_ffp, index_col="station")
dist_df = nn_gmm.load_distance_df(site_df, [distance_db_ffp], n_procs=8)

#%% Rrrup
plt.figure(figsize=(16, 10), dpi=200)
plt.hist(dist_df.rrup, bins=20)
plt.xlabel("$R_{rup}$ (km)")
plt.ylabel("Count")
plt.grid(alpha=0.6, linestyle="--", linewidth=0.5)
plt.tight_layout()
plt.show()

#%% Rrrup -- Standard
plt.figure(figsize=(16, 10), dpi=200)
plt.hist((dist_df.rrup - dist_df.rrup.mean()) / dist_df.rrup.std(), bins=20)
plt.xlabel("$R_{rup}$ (km)")
plt.ylabel("Count")
plt.grid(alpha=0.6, linestyle="--", linewidth=0.5)
plt.tight_layout()
plt.show()

#%% Rrrup - LN
plt.figure(figsize=(16, 10), dpi=200)
plt.hist(dist_df.rrup[dist_df.rrup > 10].apply(np.log), bins=20)
plt.xlabel("$ln(R_{rup})$")
plt.ylabel("Count")
plt.grid(alpha=0.6, linestyle="--", linewidth=0.5)
plt.tight_layout()
plt.show()

#%% Rjb
plt.figure(figsize=(16, 10), dpi=200)
plt.hist(dist_df.rjb, bins=20)
plt.xlabel("$R_{JB}$ (km)")
plt.ylabel("Count")
plt.grid(alpha=0.6, linestyle="--", linewidth=0.5)
plt.tight_layout()
plt.show()

#%% Rx
plt.figure(figsize=(16, 10), dpi=200)
plt.hist(dist_df.rx, bins=20)
plt.xlabel("$R_{X}$ (km)")
plt.ylabel("Count")
plt.grid(alpha=0.6, linestyle="--", linewidth=0.5)
plt.tight_layout()
plt.show()

#%% Rtvz
plt.figure(figsize=(16, 10), dpi=200)
plt.hist(dist_df.rtvz.loc[~dist_df.rtvz.isna()], bins=20)
plt.xlabel("Rtvz (km)")
plt.ylabel("Count")
plt.grid(alpha=0.6, linestyle="--", linewidth=0.5)
plt.tight_layout()
plt.show()
