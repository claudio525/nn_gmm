#%% Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import minmax_scale

#%% Load the data
source_params_ffp = "/mnt/data/work/data/nn_gmm/input_data/source_rel_params/cybershake_v20p4_200.csv"
source_df = pd.read_csv(source_params_ffp, index_col=0).sort_index()
source_df["fault"] = np.stack(np.char.rsplit(source_df.index.values.astype(str), "_", maxsplit=1))[:, 0]
# fault_df = source_df.groupby("fault").first()

#%% Dbottom

#%% Dip
plt.figure(figsize=(16, 10), dpi=200)
plt.hist(source_df.dip)
plt.xlabel("Dip (km)")
plt.ylabel("Count")
plt.grid(alpha=0.6, linestyle="--", linewidth=0.5)
plt.tight_layout()
plt.show()

#%% Dip -- Standard
plt.figure(figsize=(16, 10), dpi=200)
plt.hist((source_df.dip - source_df.dip.mean()) /source_df.dip.std() , bins=20)
plt.xlabel("Dip (km)")
plt.ylabel("Count")
plt.grid(alpha=0.6, linestyle="--", linewidth=0.5)
plt.tight_layout()
plt.show()

#%% Dip -- Min/Max
plt.figure(figsize=(16, 10), dpi=200)
dip_min_max = minmax_scale(source_df.dip, (-1, 1))
plt.hist(dip_min_max, bins=20)
plt.xlabel("Dip (km)")
plt.ylabel("Count")
plt.grid(alpha=0.6, linestyle="--", linewidth=0.5)
plt.tight_layout()
plt.show()

#%% Dtop
plt.figure(figsize=(16, 10), dpi=200)
plt.hist(source_df.dtop)
plt.xlabel("Dtop (km)")
plt.ylabel("Count")
plt.grid(alpha=0.6, linestyle="--", linewidth=0.5)
plt.tight_layout()
plt.show()

#%% Fault Length
plt.figure(figsize=(16, 10), dpi=200)
plt.hist(source_df.length, bins=20)
plt.xlabel("Fault Length (km)")
plt.ylabel("Count")
plt.grid(alpha=0.6, linestyle="--", linewidth=0.5)
plt.tight_layout()
plt.show()

#%% Magnitude
plt.figure(figsize=(16, 10), dpi=200)
plt.hist(source_df.mag, bins=20)
plt.xlabel("Magnitude, $M_W$")
plt.ylabel("Count")
plt.grid(alpha=0.6, linestyle="--", linewidth=0.5)
plt.tight_layout()
plt.show()

#%% Magnitude -- Standard
plt.figure(figsize=(16, 10), dpi=200)
plt.hist((source_df.mag - source_df.mag.mean()) / source_df.mag.std(), bins=20)
plt.xlabel("Magnitude, $M_W$")
plt.ylabel("Count")
plt.grid(alpha=0.6, linestyle="--", linewidth=0.5)
plt.tight_layout()
plt.show()


#%% Rake
plt.figure(figsize=(16, 10), dpi=200)
plt.xlabel("Rake")
plt.ylabel("Count")
plt.grid(alpha=0.6, linestyle="--", linewidth=0.5)

plot_props = dict(c="k", linewidth=0.75, linestyle="-.")
text_height = 1850
plt.hist(source_df.rake, bins=20)

# Strike slip 2
plt.axvline(150, **plot_props)
plt.gca().text(165, text_height, "SS", verticalalignment="center", horizontalalignment="center")
plt.gca().text(-165, text_height, "SS", verticalalignment="center", horizontalalignment="center")
plt.axvline(-150, **plot_props)

# Strike slip 2
plt.axvline(-30, **plot_props)
plt.gca().text(0, text_height, "SS", verticalalignment="center", horizontalalignment="center")
plt.axvline(30, **plot_props)

# Reverse
plt.axvline(60, **plot_props)
plt.gca().text(90, text_height, "Reverse", verticalalignment="center", horizontalalignment="center")
plt.axvline(120, **plot_props)

# Normal
plt.axvline(-120, **plot_props)
plt.gca().text(-90, text_height, "Normal", verticalalignment="center", horizontalalignment="center")
plt.axvline(-60, **plot_props)

plt.xlim(-180, 180)

plt.tight_layout()
plt.show()

#%% Strike
plt.figure(figsize=(16, 10), dpi=200)
plt.hist(source_df.strike, bins=20)
plt.xlabel("Strike")
plt.ylabel("Count")
plt.grid(alpha=0.6, linestyle="--", linewidth=0.5)
plt.tight_layout()
plt.show()

#%% Tectonic Type
plt.figure(figsize=(16, 10), dpi=200)
plt.hist(source_df.tect_type)
plt.xlabel("Tectonic Type")
plt.ylabel("Count")
# plt.grid(alpha=0.6, linestyle="--", linewidth=0.5)
plt.tight_layout()
plt.show()

#%% Fault Width
plt.figure(figsize=(16, 10), dpi=200)
plt.hist(source_df.width, bins=20)
plt.xlabel("Fault width")
plt.ylabel("Count")
plt.grid(alpha=0.6, linestyle="--", linewidth=0.5)
plt.tight_layout()
plt.show()