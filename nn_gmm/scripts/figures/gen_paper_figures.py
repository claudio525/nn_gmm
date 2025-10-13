import io
import os
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import typer

import nn_gmm as nng

app = typer.Typer()


def _fig_settings():
    for cur_env_key in os.environ.keys():
        if cur_env_key.startswith("fig_"):
            print("Using figure parameter:", cur_env_key, "=", os.environ[cur_env_key])

    # Update font size
    if nng.constants.FIG_FONT_SIZE is not None:
        plt.rcParams.update(
            {
                "font.size": nng.constants.FIG_FONT_SIZE,
            }
        )


@app.command("mag-rrup-scatter")
def mag_rrup_scatter(
    imdb_ffp: Path,
    output_dir: Path,
    n_rrup_bins: int = 20,
    n_mag_bins: int = 20,
    color: str = "#4363d8",
):
    """
    Magnitude Rrup scatter plot.
    Note: Each points represents a site-event pair, NOT a single record,
    as there are multiple records for a given site-event pair
    due to the simulation realisations
    """
    logger = nng.utils.setup_logging()
    _fig_settings()

    with nng.DuckIMDB(imdb_ffp, readonly=True) as imdb:
        event_df = imdb.get_event_df()
        record_info_df = imdb.get_record_info_df()

        # Get site-event pairs
        site_event_pairs = record_info_df[
            ["site_int_id", "event_int_id"]
        ].drop_duplicates()
        site_event_int_ids = nng.utils.get_site_event_int_id(
            site_event_pairs.site_int_id.values, site_event_pairs.event_int_id.values
        )

        site_event_df = imdb.get_site_event_df(
            site_event_int_ids=site_event_int_ids
        ).sort_index()

    site_event_df["magnitude"] = event_df.loc[
        site_event_df.event_int_id.values, "magnitude"
    ].values

    fig, axs = plt.subplot_mosaic(
        [["histx", "."], ["scatter", "histy"]],
        width_ratios=(4, 1),
        height_ratios=(1, 4),
        layout="constrained",
        figsize=nng.constants.FIG_SIZE,
        dpi=nng.constants.FIG_DPI,
    )
    ax_scatter = axs["scatter"]
    ax_histx = axs["histx"]
    ax_histy = axs["histy"]

    # Scatter plot
    ax_scatter.scatter(
        site_event_df.rrup,
        site_event_df.magnitude,
        s=2.0,
        c=color,
        alpha=0.75,
    )

    ax_scatter.set_xlabel("Source-to-site distance, $R_{rup}$ (km)")
    ax_scatter.set_ylabel("Magnitude, $M_{W}$")
    ax_scatter.set_xscale("log")
    ax_scatter.set_xlim(0.1, 1000)
    ax_scatter.grid(which="both", linewidth=0.5, alpha=0.5, linestyle="--")

    ax_histx.hist(
        site_event_df.rrup,
        bins=np.logspace(np.log10(0.1), np.log10(1000), n_rrup_bins),
        color=color,
        edgecolor="black",
        log=True,
    )
    ax_histx.set_xscale("log")
    ax_histx.set_xlim(0.1, 1000)
    ax_histx.spines[["top", "right"]].set_visible(False)
    ax_histx.set_xticklabels([])

    ax_histy.hist(
        site_event_df.magnitude,
        bins=n_mag_bins,
        color=color,
        orientation="horizontal",
        edgecolor="black",
        log=True,
    )
    ax_histy.set_ylim(ax_scatter.get_ylim())
    ax_histy.spines[["top", "right"]].set_visible(False)
    ax_histy.set_yticklabels([])

    plt.savefig(output_dir / f"rrup_vs_mag.{nng.constants.FIG_FORMAT}")


@app.command("nzgmdb-mag-rrup-scatter")
def nzgmdb_mag_rrup_scatter(
    nzgmdb_ffp: Path, output_dir: Path, n_rrup_bins: int = 20, n_mag_bins: int = 20
):
    """
    Creates a scatter of rrup vs magnitude
    with the marginal distributions on the sides
    """
    logger = nng.utils.setup_logging()
    _fig_settings()

    obs_data = nng.obs_data.ObservedData.from_nzgmdb_flat(nzgmdb_ffp)
    obs_data = obs_data.to_event_site_index()

    # Load basic filtered data
    filtered_obs_data = nng.obs_data.load_obs_nzgmdb(
        nzgmdb_ffp, apply_mag_distance_filter=True
    )

    fig, axs = plt.subplot_mosaic(
        [["histx", "."], ["scatter", "histy"]],
        width_ratios=(4, 1),
        height_ratios=(1, 4),
        layout="constrained",
        figsize=nng.constants.FIG_SIZE,
        dpi=nng.constants.FIG_DPI,
    )
    ax_scatter = axs["scatter"]
    ax_histx = axs["histx"]
    ax_histy = axs["histy"]

    # Scatter plot
    ax_scatter.scatter(
        obs_data.record_df["rrup"],
        obs_data.record_df["mag"],
        s=2.0,
        c="grey",
        alpha=0.75,
        label=f"All (N={obs_data.n_records:,})",
    )
    ax_scatter.scatter(
        filtered_obs_data.record_df["rrup"],
        filtered_obs_data.record_df["mag"],
        s=2.0,
        c="red",
        alpha=0.75,
        label=f"Filtered (N={filtered_obs_data.n_records:,})",
    )
    ax_scatter.plot(
        nng.constants.MW_RRUP_LIMITS[:, 1],
        nng.constants.MW_RRUP_LIMITS[:, 0],
        c="blue",
        label="Magnitude-distance filter",
        linewidth=nng.constants.FIG_LINEWIDTH,
    )

    ax_scatter.set_xlabel("Source-to-site distance, $R_{rup}$ (km)")
    ax_scatter.set_ylabel("Magnitude, $M_{W}$")
    ax_scatter.legend()
    ax_scatter.set_xscale("log")
    ax_scatter.set_xlim(0.1, 1000)
    ax_scatter.grid(which="both", linewidth=0.5, alpha=0.5, linestyle="--")

    ax_histx.hist(
        filtered_obs_data.record_df["rrup"],
        bins=np.logspace(np.log10(0.1), np.log10(1000), n_rrup_bins),
        color="red",
        edgecolor="black",
    )
    ax_histx.set_xscale("log")
    ax_histx.set_xlim(0.1, 1000)
    ax_histx.spines[["top", "right"]].set_visible(False)
    ax_histx.set_xticklabels([])
    ax_histx.set_yscale("log")

    ax_histy.hist(
        filtered_obs_data.record_df["mag"],
        bins=n_mag_bins,
        color="red",
        orientation="horizontal",
        edgecolor="black",
    )
    ax_histy.set_ylim(ax_scatter.get_ylim())
    ax_histy.spines[["top", "right"]].set_visible(False)
    ax_histy.set_yticklabels([])
    ax_histy.set_xscale("log")

    plt.savefig(output_dir / f"nzgmdb_rrup_vs_mag.{nng.constants.FIG_FORMAT}")


@app.command("fault-basin-map")
def fault_basin_map(output_dir: Path):
    """
    Plot fault and basin map of NZ
    """
    logger = nng.utils.setup_logging()

    spatial_plots = (
        nng.plots_spatial.SpatialPlot()
        .plot_faults(pen="1p,black")
        .plot_basin_boundaries(pen="0.5p,red")
    )

    spatial_plots.save(output_dir / f"nz_faults_and_basins.{nng.constants.FIG_FORMAT}")


@app.command("nzgmdb-map")
def nzgmdb_map(nzgmdb_ffp: Path, output_dir: Path):
    """
    Plot historic events and strong motion stations from NZGMDB
    """
    obs_data = nng.obs_data.load_obs_nzgmdb(nzgmdb_ffp)

    event_df = obs_data.event_df.copy()
    event_df = event_df[event_df.mag >= 4.0]

    # Split historic events into magnitude bins
    mag_bins = [4, 5, 6, 9]
    labels = ["blue", "orange", "red"]
    event_df.loc[:, "mag_bin"] = pd.cut(
        event_df.mag, bins=mag_bins, include_lowest=True, labels=labels
    )

    spatial_plot = nng.plots_spatial.SpatialPlot()

    # Plot the historic events
    for i, (_, cur_row) in enumerate(event_df.sort_values("mag").iterrows()):
        spatial_plot.fig.meca(
            spec={
                "strike": cur_row.strike,
                "dip": cur_row.dip,
                "rake": cur_row.rake,
                "magnitude": cur_row.mag,
                "depth": cur_row.depth,
            },
            scale=f"{0.04 * cur_row.mag}c",
            longitude=cur_row.lon,
            latitude=cur_row.lat,
            pen="0.05p,black,solid",
            compressionfill=cur_row.mag_bin,
        )

    spatial_plot.plot_sites(
        obs_data.site_df, style="t0.05c", pen="0.05p,black,solid", fill="green3"
    )

    legend_spec = io.StringIO()
    legend_spec.write("H 12p,Helvetica-Bold Historic Events\n")
    legend_spec.write("D 0.1i 1p\n")
    legend_spec.write("S 0.1i c 0.15c blue 0.05p,black 0.4i Magnitude 4.0-5.0\n")
    legend_spec.write("S 0.1i c 0.20c orange 0.05p,black 0.4i Magnitude 5.0-6.0\n")
    legend_spec.write("S 0.1i c 0.25c red 0.05p,black 0.4i Magnitude 6.0+\n")

    spatial_plot.fig.legend(
        spec=legend_spec,
        box="+gwhite+p1p",
    )

    spatial_plot.save(
        output_dir / f"nzgmdb_events_and_stations.{nng.constants.FIG_FORMAT}"
    )


if __name__ == "__main__":
    app()
