import time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import pygmt

# Grow the GPU memory usage as needed
gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices("GPU")
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

import nn_gmm
from qcore.uncertainties import mag_scaling
from qcore import constants


# def set_size(width, fraction=1):
#     """Set figure dimensions to avoid scaling in LaTeX.
#
#     Parameters
#     ----------
#     width: float
#             Document textwidth or columnwidth in pts
#     fraction: float, optional
#             Fraction of the width which you wish the figure to occupy
#
#     Returns
#     -------
#     fig_dim: tuple
#             Dimensions of figure in inches
#     """
#     # Width of figure (in pts)
#     fig_width_pt = width * fraction
#
#     # Convert from pt to inches
#     inches_per_pt = 1 / 72.27
#
#     # Golden ratio to set aesthetic figure height
#     # https://disq.us/p/2940ij3
#     golden_ratio = (5**.5 - 1) / 2
#
#     # Figure width in inches
#     fig_width_in = fig_width_pt * inches_per_pt
#     # Figure height in inches
#     fig_height_in = fig_width_in * golden_ratio
#
#     fig_dim = (fig_width_in, fig_height_in)
#
#     return fig_dim
#
# fig_dim = set_size(345)
# fig_dim = (16, 10)

fig_dim = (8, 6)

# Data config
source_params_dir = Path("/home/claudy/dev/work/data/nn_gmm/input_data/source_rel_params")
site_params_dir = Path("/home/claudy/dev/work/data/nn_gmm/input_data/site_params")

distance_dir = Path("/home/claudy/dev/work/data/nn_gmm/input_data/distance_data")


# paper_plots_dir = Path("/home/claudy/dev/work/data/nn_gmm/results/keep/paper_plots")
paper_plots_dir = Path("/home/claudy/dev/work/tmp/paper_plots")

# Load source params
rel_df = nn_gmm.load_dfs(
    list(source_params_dir.glob("*.csv")), index_col="realisation"
)

# Load site data
site_df = nn_gmm.load_dfs(list(site_params_dir.glob("*.csv")))

# Check & Drop duplicates
n_unique_stations = np.unique(site_df.station.values.astype(str)).shape[0]
site_df = site_df.drop_duplicates()
assert n_unique_stations == site_df.shape[0]
site_df.set_index("station", inplace=True)
site_df.drop_duplicates(subset={"lat", "lon"}, inplace=True)

# Load distance data
distance_db_ffps = list(distance_dir.glob("*.db"))
distance_df = nn_gmm.load_distance_df(site_df, distance_db_ffps, n_procs=12)

print("")

# ---------------------------------------------------

# Source params plot
fig = plt.figure(figsize=fig_dim)

# Magnitude
ax = fig.add_subplot(1, 2, 1)

ax.hist(rel_df.mag, bins=25)
# ax.set_title(f"Magnitude")
ax.set_xlabel(r"Magnitude, $M_w$")
ax.set_ylabel(f"Count")
ax.grid(linewidth=0.5, alpha=0.5, linestyle="--")

# Rake
ax = fig.add_subplot(1, 2, 2)

ax.hist(rel_df.rake, bins=25)

# ax.set_title(f"Rake")
ax.set_xlabel(r"Rake, $\lambda$")
ax.set_ylabel(f"Number of ruptures")
ax.grid(linewidth=0.5, alpha=0.5, linestyle="--")

# Normal
ax.axvline(-120, linestyle="--", linewidth=0.75, c="k")
ax.axvline(-60, linestyle="--", linewidth=0.75, c="k")
ax.text(-90, ax.get_ylim()[1] - 20, f"Normal", horizontalalignment="center", verticalalignment="top")


# Strike Slip
ax.axvline(-30, linestyle="--", linewidth=0.75, c="k")
ax.axvline(30, linestyle="--", linewidth=0.75, c="k")
ax.text(0, ax.get_ylim()[1] - 20, f"SS", horizontalalignment="center", verticalalignment="top")

# Reverse
ax.axvline(120, linestyle="--", linewidth=0.75, c="k")
ax.axvline(60, linestyle="--", linewidth=0.75, c="k")
ax.text(90, ax.get_ylim()[1] - 20, f"Reverse", horizontalalignment="center", verticalalignment="top")

# Strike Slip
ax.axvline(150, linestyle="--", linewidth=0.75, c="k")
ax.axvline(-150, linestyle="--", linewidth=0.75, c="k")
ax.text(165, ax.get_ylim()[1] - 20, f"SS", horizontalalignment="center", verticalalignment="top")
ax.text(-165, ax.get_ylim()[1] - 20, f"SS", horizontalalignment="center", verticalalignment="top")


ax.set_xlim(-180, 180)

fig.tight_layout()
fig.savefig(paper_plots_dir / "mag_rake_dist.png", format="png", bbox_inches='tight')
fig.savefig(paper_plots_dir / "mag_rake_dist.pdf", bbox_inches='tight')

# ---------------------------------------------------

# Rrup (Should this be ruptures instead of sources?)
fig = plt.figure(figsize=fig_dim)
ax = fig.add_subplot(1, 1, 1)

ax.hist(distance_df.rrup, bins=25)
ax.set_xlabel(r"Distance, $R_{Rup}$")
ax.set_ylabel(f"Number of site-source pairs")
ax.grid(linewidth=0.5, alpha=0.5, linestyle="--")

fig.tight_layout()
fig.savefig(paper_plots_dir / "rrup.png", format="png", bbox_inches='tight')
fig.savefig(paper_plots_dir / "rrup.pdf", bbox_inches='tight')

# Log x-scale?
# logbins = np.logspace(np.log10(10), np.log10(500), 25)
#
# fig = plt.figure(figsize=fig_dim)
# ax = fig.add_subplot(1, 1, 1)
#
# ax.hist(distance_df.rrup, bins=logbins)
# ax.set_xlabel(r"Distance, $R_{Rup}$")
# ax.set_ylabel(f"Number of site-source pairs")
# ax.grid(linewidth=0.5, alpha=0.5, linestyle="--")
# ax.semilogx()
#
# fig.tight_layout()



# ---------------------------------------------------


# Site
fig = plt.figure(figsize=fig_dim)

ax = fig.add_subplot(1, 2, 1)
ax.hist(site_df.vs30, bins=25)
ax.set_xlabel(r"$V_{S30}$")
ax.set_ylabel(f"Number of sites")
ax.grid(linewidth=0.5, alpha=0.5, linestyle="--")

ax = fig.add_subplot(1, 2, 2)
ax.hist(site_df.z1p0, bins=25)
ax.set_xlabel(r"$Z_{1.0}$")
ax.set_ylabel(f"Number of sites")
ax.grid(linewidth=0.5, alpha=0.5, linestyle="--")

fig.tight_layout()
fig.savefig(paper_plots_dir / "vs30_z1p0.png", format="png", bbox_inches='tight')
fig.savefig(paper_plots_dir / "vs30_z1p0.pdf", bbox_inches='tight')