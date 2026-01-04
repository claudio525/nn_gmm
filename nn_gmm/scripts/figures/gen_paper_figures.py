import io
import os
from pathlib import Path

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import typer

import nn_gmm as nng
from mera import MeraResults
import ml_tools as mlt
from qcore import nhm


device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
if torch.mps.is_available():
    device = "mps"
print(f"Using device: {device.upper()}")

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
    nng.utils.setup_logging()
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


@app.command("fault-basin-map")
def fault_basin_map(output_dir: Path):
    """
    Plot fault and basin map of NZ
    """
    nng.utils.setup_logging()

    spatial_plots = (
        nng.plots_spatial.SpatialPlot()
        .plot_fault_traces(pen="1p,black")
        .plot_basin_boundaries(pen="0.5p,red")
    )

    spatial_plots.save(output_dir / f"nz_faults_and_basins.{nng.constants.FIG_FORMAT}")


@app.command("fault-map")
def fault_map(output_dir: Path, imdb_ffp: Path):
    """
    Plot fault map of NZ
    """
    nng.utils.setup_logging()

    with nng.DuckIMDB(imdb_ffp, readonly=True) as imdb:
        event_df = imdb.get_event_df()
    
    # Ignore point sources
    event_df = event_df.loc[event_df.fault_type != "DS_POINT_SOURCE"]

    region = [165.4, 179.6, -47.4, -36.2]

    spatial_plot = nng.plots_spatial.SpatialPlot(
        plot_topo=True, plot_highways=True, region=region
    )

    # Plot simulated crustal faults
    spatial_plot.plot_fault_traces(
        event_df.loc[event_df.tect_type == "ACTIVE_SHALLOW"].event_id.values.astype(
            str
        ),
        label="Simulated Active Shallow Sources",
        pen="0.75p,red",
    )

    # Plot simulated crustal faults
    spatial_plot.plot_fault_traces(
        event_df.loc[event_df.tect_type == "VOLCANIC"].event_id.values.astype(
            str
        ),
        label="Simulated Volcanic Sources",
        pen="0.75p,maroon",
    )

    # Plot simulated subduction faults
    spatial_plot.plot_fault_traces(
        event_df.loc[event_df.tect_type == "SUBDUCTION_INTERFACE"].event_id.values.astype(
            str
        ),
        label="Simulated Subduction Interface Sources",
        pen="0.75p,blue",
    )

    # Not simulated faults from NHM
    nhm_df = nhm.load_nhm_df(str(nng.constants.NHM_FAULT_FFP))
    not_simulated_faults = nhm_df.loc[~nhm_df.name.isin(event_df.event_id)].name.values.astype(str)
    spatial_plot.plot_fault_traces( 
        not_simulated_faults,
        label="Sources not Simulated",
        pen="0.5p,black",
    )

    # Add legend
    spatial_plot.fig.legend(position="JTL+jTL+o0.2c", box="+gwhite+p1p")
    
    spatial_plot.save(output_dir / f"nz_faults.{nng.constants.FIG_FORMAT}")


@app.command("mera-model-bias-std")
def mera_model_bias_std(
    base_model_results_dir: Path,
    loc_adj_results_dir: Path,
    output_dir: Path,
):
    nng.utils.setup_logging()
    _fig_settings()

    # Check that MERA results exist
    if not (
        base_mera_results_dir := base_model_results_dir / "mera_site_term"
    ).exists():
        raise FileNotFoundError(
            f"MERA results directory not found: {base_mera_results_dir}"
        )

    if not (
        loc_adj_mera_results_dir := loc_adj_results_dir / "mera_site_term"
    ).exists():
        raise FileNotFoundError(
            f"MERA results directory not found: {loc_adj_mera_results_dir}"
        )

    base_mera_results = MeraResults.load_from_parquet(base_mera_results_dir)
    ims = base_mera_results.bias_std_df.index.values.astype(str)
    periods = [nng.utils.get_pSA_period(im) for im in ims]

    loc_adj_mera_results = MeraResults.load_from_parquet(loc_adj_mera_results_dir)
    assert np.array_equal(
        loc_adj_mera_results.bias_std_df.index.values,
        base_mera_results.bias_std_df.index.values,
    )

    bias_std_plot = nng.plots.BiasStdPlot(
        figsize=nng.constants.FIG_SIZE,
        dpi=nng.constants.FIG_DPI,
        bias_ylim=(-0.25, 0.25),
        std_ylim=(0, 0.5),
        main_wspace=0.175,
        left=0.07,
    )
    bias_std_plot.ax1.set_ylabel("Model Bias", labelpad=-2)
    bias_std_plot.ax3.set_ylabel("Standard Deviation")

    base_std_df = base_mera_results.bias_std_df.copy()
    base_std_df.index = periods

    loc_adj_std_df = loc_adj_mera_results.bias_std_df.copy()
    loc_adj_std_df.index = periods

    # Model bias
    bias_std_plot.add_bias(base_std_df["bias"], c="blue", label="Base model")
    bias_std_plot.add_bias(loc_adj_std_df["bias"], c="red", label="Location model")
    bias_std_plot.add_legend(bias_std_plot.ax1)

    # Between-event standard deviation
    bias_std_plot.add_std(base_std_df["tau"], c="blue", label="Between-event std.")
    bias_std_plot.add_std(loc_adj_std_df["tau"], c="red")

    # Site-term standard deviation
    bias_std_plot.add_std(
        base_std_df["phi_S2S"], c="blue", linestyle="dotted", label="Site-term std."
    )
    bias_std_plot.add_std(loc_adj_std_df["phi_S2S"], c="red", linestyle="dotted")

    # Remaining standard deviation
    bias_std_plot.add_std(
        base_std_df["phi_w"], c="blue", linestyle="dashed", label="Within-event std."
    )
    bias_std_plot.add_std(loc_adj_std_df["phi_w"], c="red", linestyle="dashed")

    bias_std_plot.add_legend(bias_std_plot.ax3)

    bias_std_plot.fig.savefig(
        output_dir / f"mera_model_bias_std.{nng.constants.FIG_FORMAT}"
    )
    plt.close(bias_std_plot.fig)


@app.command("model-trends")
def model_trends(
    base_model_dir: Path,
    loc_model_dir: Path,
    config_ffp: Path,
    output_dir: Path,
    ims: list[str],
    locations: list[str] = None,
):
    nng.utils.setup_logging()
    _fig_settings()

    base_run_config = nng.nn_gmm.load_config(base_model_dir / "run_config.yaml")
    loc_run_config = nng.nn_gmm.load_config(loc_model_dir / "run_config.yaml")
    assert loc_run_config.base_model_dir == base_model_dir

    val_record_ids = pd.read_parquet(
        base_model_dir / "val_results.parquet"
    ).index.values
    with nng.DuckIMDB(base_run_config.imdb_ffp, readonly=True) as imdb:
        sim_df = imdb.get_im_data(base_run_config.ims, record_int_ids=val_record_ids)
        record_info_df = imdb.get_record_info_df(record_int_ids=val_record_ids)
        rel_df = imdb.get_rel_df(rel_int_ids=record_info_df.rel_int_id.unique())
        sim_df["rel_int_id"] = record_info_df.loc[sim_df.index, "rel_int_id"]
        sim_df["magnitude"] = rel_df.loc[sim_df.rel_int_id.values, "magnitude"].values

    legend_handles = []
    config = mlt.utils.load_yaml(config_ffp)
    if config["type"] == "mag":
        fig, axs = nng.plots.magnitude_trend_plot(
            base_model_dir,
            config["fixed_inputs"],
            config["records_range"],
            device,
            record_int_ids=val_record_ids,
            simulation_df=sim_df,
            cv=True,
            ims=ims,
            plot_ind_cv=False,
            major_line_width=nng.constants.FIG_LINEWIDTH,
            minor_line_width=nng.constants.FIG_GROUP_LINEWIDTH,
            dpi=nng.constants.FIG_DPI,
            ind_fig_size=nng.constants.FIG_SIZE,
            legend=False,
            legend_labels=False,
        )

        legend_handles.append(mlines.Line2D([], [], color="blue", label="Base GMM"))
        legend_handles.append(mlines.Line2D([], [], color="g", label="Empirical GMM"))

        colors = ["red", "orange", "yellow"]
        if locations is not None:
            for i, loc in enumerate(locations):
                if loc not in nng.constants.NZ_SITE_LOCATIONS:
                    raise ValueError(f"Unknown NZ location: {loc}")
                nztm_coords = nng.constants.NZ_SITE_LOCATIONS[loc]["nztm"]

                nng.plots.magnitude_trend_plot(
                    loc_model_dir,
                    config["fixed_inputs"]
                    | {"nztm_x": nztm_coords[0], "nztm_y": nztm_coords[1]},
                    config["records_range"],
                    device,
                    cv=True,
                    ims=ims,
                    plot_ind_cv=False,
                    major_line_width=nng.constants.FIG_LINEWIDTH,
                    minor_line_width=nng.constants.FIG_GROUP_LINEWIDTH,
                    dpi=nng.constants.FIG_DPI,
                    ind_fig_size=nng.constants.FIG_SIZE,
                    plot_empirical=False,
                    nn_color=colors[i],
                    axs=axs,
                    legend=False,
                    legend_labels=False,
                )

                legend_handles.append(
                    mlines.Line2D(
                        [], [], color=colors[i], label=f"Location GMM - {loc}"
                    )
                )

        axs[0].legend(handles=legend_handles)
        axs[0].set_xticklabels([])
        axs[1].set_xticklabels([])
        axs[1].set_yticklabels([])
        axs[3].set_yticklabels([])

        axs[0].set_xlabel(None)
        axs[1].set_xlabel(None)

        for ax, im in zip(axs, ims):
            ax.text(
                0.025,
                0.975,
                nng.utils.get_nice_im_name(im),
                transform=ax.transAxes,
                horizontalalignment="left",
                verticalalignment="top",
                fontweight="bold",
            )
            ax.set_ylim(1e-5, 0.5)

        fig.subplots_adjust(
            left=0.06, right=0.99, top=0.99, bottom=0.07, wspace=0.025, hspace=0.025
        )

        fig.savefig(output_dir / f"{config_ffp.stem}.{nng.constants.FIG_FORMAT}")
    elif config["type"] == "rrup":
        pass
    else:
        raise ValueError(f"Unknown trend type: {config['type']}")


if __name__ == "__main__":
    app()
