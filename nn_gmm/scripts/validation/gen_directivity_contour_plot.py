"""Creates a contour plot of theta and s for the given srf"""
import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

import nn_gmm
from qcore import srf, geo

matplotlib.use("Agg")


def main(srf_ffp: Path, srf_info_ffp: Path, output_ffp: Path, vector_dist: float = 10):

    def _plot_other(ax):
        ax.scatter(
            srf_points[:, 0][::2],
            srf_points[:, 1][::2],
            c=srf_points[:, 2][::2],
            label="srf points",
            s=1.0,
        )
        ax.scatter(
            hypo_lon,
            hypo_lat,
            label="Hypocentre",
            marker="x",
            c="k",
            s=50.0,
        )
        rake_lat, rake_lon = geo.ll_shift(hypo_lat, hypo_lon, vector_dist, hypo_rake_bearing)
        ax.plot(
            [hypo_lon, rake_lon],
            [hypo_lat, rake_lat],
            label="Rake bearing",
            c="b",
            linestyle="-",
            linewidth=1.0,
        )
        rake_lat, rake_lon = geo.ll_shift(
            hypo_lat,
            hypo_lon,
            vector_dist,
            hypo_rake_bearing + 180 if hypo_rake_bearing + 180 < 360 else hypo_rake_bearing - 180,
        )
        ax.plot([hypo_lon, rake_lon], [hypo_lat, rake_lat], c="b", linestyle="-", linewidth=1.0)

    with h5py.File(srf_info_ffp, "r") as f:
        srf_info = dict(f.attrs)

    seg_bounds = srf.get_bounds(str(srf_ffp))
    srf_points = srf.read_srf_points(str(srf_ffp))  # Lon, lat, depth
    hypo_lon, hypo_lat = srf_info["hlon"], srf_info["hlat"]
    strike_values, rake = srf_info["strike"], srf_info["rake"]

    hypo_strike = strike_values[nn_gmm.get_hypo_seg_ix(seg_bounds, hypo_lon, hypo_lat)]
    hypo_rake_bearing = hypo_strike - rake if hypo_strike - rake > 0 else 360 - (hypo_strike - rake)

    # Create the mesh
    lon_values = np.linspace(
        srf_points[:, 0].min() - 1, srf_points[:, 0].max() + 1, 100
    )
    lat_values = np.linspace(
        srf_points[:, 1].min() - 1, srf_points[:, 1].max() + 1, 100
    )

    x, y = np.meshgrid(lon_values, lat_values)
    site_coords = np.stack((x, y), axis=2).reshape(-1, 2)

    result = nn_gmm.compute_theta_s(
        seg_bounds,
        strike_values,
        rake,
        (hypo_lon, hypo_lat),
        site_coords,
        n_procs=8,
    )

    theta_values = np.asarray(result[0]).reshape(100, 100)
    s_values = np.asarray(result[1]).reshape(100, 100)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(21, 13.5), dpi=144)

    levels = 20
    ct = ax1.contourf(x, y, theta_values, cmap="Reds_r", levels=levels)
    ax1.set_title("Theta")
    fig.colorbar(ct, ax=ax1, pad=0.01)
    _plot_other(ax1)
    ax1.legend()

    ct = ax2.contourf(x, y, s_values, cmap="Reds", levels=levels)
    ax2.set_title("s")
    fig.colorbar(ct, ax=ax2, pad=0.01)
    _plot_other(ax2)

    fig.suptitle(srf_ffp.name.split(".")[0])

    fig.tight_layout()
    fig.savefig(output_ffp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("srf_ffp", type=str, help="Path to the srf file")
    parser.add_argument("srf_info_ffp", type=str, help="Path to the srf info file")

    parser.add_argument("output_ffp", type=str, help="File path for the outputs.")

    args = parser.parse_args()

    main(Path(args.srf_ffp), Path(args.srf_info_ffp), Path(args.output_ffp))
