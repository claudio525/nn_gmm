#%% Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import minmax_scale

#%% Load the data
site_params_ffp = "/mnt/data/work/data/nn_gmm/input_data/site_params/site_params.csv"
site_df = pd.read_csv(site_params_ffp, index_col=0)

#%% Vs30
plt.figure(figsize=(16, 10), dpi=200)
plt.hist(site_df.vs30, bins=20)
plt.xlabel("$V_{S30}$ (m/s)")
plt.ylabel("Count")
plt.grid(alpha=0.6, linestyle="--", linewidth=0.5)
plt.tight_layout()
plt.show()

#%% Vs30 -- Standard
plt.figure(figsize=(16, 10), dpi=200)
plt.hist((site_df.vs30 - site_df.vs30.mean()) / site_df.vs30.std(), bins=20)
plt.xlabel("$V_{S30}$ (m/s)")
plt.ylabel("Count")
plt.grid(alpha=0.6, linestyle="--", linewidth=0.5)
plt.tight_layout()
plt.show()

#%% Vs30 -- Min/Max
plt.figure(figsize=(16, 10), dpi=200)
vs30_min_max = minmax_scale(site_df.vs30, (-1, 1))
plt.hist(vs30_min_max, bins=20)
plt.xlabel("$V_{S30}$ (m/s)")
plt.ylabel("Count")
plt.grid(alpha=0.6, linestyle="--", linewidth=0.5)
plt.tight_layout()
plt.show()

#%% Vs500
plt.figure(figsize=(16, 10), dpi=200)
plt.hist(site_df.vs500, bins=20)
plt.xlabel("$V_{S500}$ (m/s)")
plt.ylabel("Count")
plt.grid(alpha=0.6, linestyle="--", linewidth=0.5)
plt.tight_layout()
plt.show()

#%% Z1.0
plt.figure(figsize=(16, 10), dpi=200)
plt.hist(site_df.z1p0, bins=20)
plt.xlabel("Z1.0 (km)")
plt.ylabel("Count")
plt.xlim(0.0, None)
plt.grid(alpha=0.6, linestyle="--", linewidth=0.5)
plt.tight_layout()
plt.show()

#%% Z2.5
plt.figure(figsize=(16, 10), dpi=200)
plt.hist(site_df.z2p5, bins=20)
plt.xlabel("Z2.5 (km)")
plt.ylabel("Count")
plt.xlim(0.0, None)
plt.grid(alpha=0.6, linestyle="--", linewidth=0.5)
plt.tight_layout()
plt.show()
