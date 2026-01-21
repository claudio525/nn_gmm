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
        event_df.loc[event_df.tect_type == "VOLCANIC"].event_id.values.astype(str),
        label="Simulated Volcanic Sources",
        pen="0.75p,maroon",
    )

    # Plot simulated subduction faults
    spatial_plot.plot_fault_traces(
        event_df.loc[
            event_df.tect_type == "SUBDUCTION_INTERFACE"
        ].event_id.values.astype(str),
        label="Simulated Subduction Interface Sources",
        pen="0.75p,blue",
    )

    # Not simulated faults from NHM
    nhm_df = nhm.load_nhm_df(str(nng.constants.NHM_FAULT_FFP))
    not_simulated_faults = nhm_df.loc[
        ~nhm_df.name.isin(event_df.event_id)
    ].name.values.astype(str)
    spatial_plot.plot_fault_traces(
        not_simulated_faults,
        label="Sources not Simulated",
        pen="0.5p,black",
    )

    # Add legend
    spatial_plot.fig.legend(position="JTL+jTL+o0.2c", box="+gwhite+p1p")

    spatial_plot.save(output_dir / f"nz_faults.{nng.constants.FIG_FORMAT}")


@app.command("mera-basin-site-term")
def mera_basin_site_term(
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

    run_config_1 = nng.nn_gmm.load_config(base_model_results_dir / "run_config.yaml")
    with nng.DuckIMDB(
        run_config_1.imdb_ffp,
        readonly=True,
    ) as imdb:
        site_df = imdb.get_site_df().set_index("site_id")

    base_mera_results = MeraResults.load_from_parquet(base_mera_results_dir)
    loc_mera_results = MeraResults.load_from_parquet(loc_adj_mera_results_dir)
    assert base_mera_results.bias_std_df.index.equals(
        loc_mera_results.bias_std_df.index
    ), "IMs do not match between the two MERA results"
    ims = base_mera_results.bias_std_df.index.values
    periods = [nng.utils.get_pSA_period(im) for im in ims]

    base_site_term = base_mera_results.site_res_df.sort_index()
    base_site_term.loc[:, ["lon", "lat"]] = site_df.loc[
        base_site_term.index, ["lon", "lat"]
    ]
    base_site_term = nng.utils.add_basin_column(base_site_term)
    loc_site_term = loc_mera_results.site_res_df.sort_index()
    loc_site_term.loc[:, ["lon", "lat"]] = site_df.loc[
        loc_site_term.index, ["lon", "lat"]
    ]
    loc_site_term = nng.utils.add_basin_column(loc_site_term)

    assert np.all(
        base_site_term.index.values.astype(str)
        == loc_site_term.index.values.astype(str)
    ), "Site mismatch between the two MERA results"

    # Basin grouping
    north_canterbury_mask = (
        (base_site_term.basin == "NorthCanterbury")
        | (base_site_term.basin == "Kaikoura")
        | (base_site_term.basin == "Hanmer")
        | (base_site_term.basin == "Cheviot")
    )
    nelson_marlborough_wellington_mask = (
        (base_site_term.basin == "Nelson")
        | (base_site_term.basin == "Marlborough")
        | (base_site_term.basin == "Wellington")
    )

    base_site_term["basin_group"] = base_site_term.basin.values.astype(str)
    base_site_term.loc[north_canterbury_mask, "basin_group"] = "NorthCanterbury"
    base_site_term.loc[nelson_marlborough_wellington_mask, "basin_group"] = (
        "NelsonMarlboroughWellington"
    )
    base_site_term["basin_group"] = base_site_term["basin_group"].astype("category")

    loc_site_term["basin_group"] = base_site_term.basin_group.copy()

    group_keys = [
        "NiB",
        "Canterbury",
        "NorthCanterbury",
        "NelsonMarlboroughWellington",
        "WaikatoHauraki",
    ]
    group_colors = ["k", "b", "magenta", "green", "red"]
    group_labels = [
        "Not in Basin",
        "Canterbury",
        "North Canterbury\nBasins",
        "Nelson, Marlborough,\nWellington",
        "Waikato, Hauraki",
    ]
    bias_std_plot = nng.plots.GroupedBiasStdPlot(
        "basin_group",
        None,
        group_keys,
        group_colors,
        group_labels=group_labels,
        pSA_keys_periods=(ims, periods),
        bias_ylim=(-0.2, 0.2),
        std_ylim=(0, 0.3),
        figsize=nng.constants.FIG_SIZE,
        dpi=nng.constants.FIG_DPI,
        main_wspace=0.175,
        left=0.07,
    )

    bias_std_plot.add_categorial_results(base_site_term, True)
    bias_std_plot.add_categorial_results(loc_site_term, False, linestyle="--")
    bias_std_plot.add_legend(bias_std_plot.ax3)

    legend_elements = {
        mlines.Line2D(
            [], [], color="black", linestyle="--", label="Location-specific model"
        ),
        mlines.Line2D([], [], color="black", linestyle="-", label="Base model"),
    }
    bias_std_plot.ax1.legend(handles=legend_elements)

    bias_std_plot.ax1.set_ylabel("Mean Site-to-Site Residual", labelpad=-5.0)
    bias_std_plot.ax3.set_ylabel(
        "Site-to-Site Residual Standard Deviation", labelpad=1.0
    )

    bias_std_plot.ax1.yaxis.set_minor_locator(plt.MultipleLocator(0.05))
    bias_std_plot.ax1.yaxis.set_major_locator(plt.MultipleLocator(0.1))
    bias_std_plot.ax3.yaxis.set_minor_locator(plt.MultipleLocator(0.05))
    bias_std_plot.ax3.yaxis.set_major_locator(plt.MultipleLocator(0.1))

    bias_std_plot.fig.savefig(output_dir / "mera_basin_site_term_comparison.png")
    plt.close(bias_std_plot.fig)


@app.command("mera-site-term")
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
    bias_std_plot.ax1.set_ylabel("Model prediction bias, a", labelpad=-2)
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
    bias_std_plot.add_std(base_std_df["tau"], c="blue", label=r"Between-event, $\tau$")
    bias_std_plot.add_std(loc_adj_std_df["tau"], c="red")

    # Site-term standard deviation
    bias_std_plot.add_std(
        base_std_df["phi_S2S"],
        c="blue",
        linestyle="dotted",
        label=r"Site-to-site, $\phi_{S2S}$",
    )
    bias_std_plot.add_std(loc_adj_std_df["phi_S2S"], c="red", linestyle="dotted")

    # Remaining standard deviation
    bias_std_plot.add_std(
        base_std_df["phi_w"], c="blue", linestyle="dashed", label=r"Remaining, $\phi_w$"
    )
    bias_std_plot.add_std(loc_adj_std_df["phi_w"], c="red", linestyle="dashed")

    bias_std_plot.add_legend(bias_std_plot.ax3)

    output_ffp = output_dir / f"mera_model_bias_std.{nng.constants.FIG_FORMAT}"
    bias_std_plot.fig.savefig(output_ffp)
    plt.close(bias_std_plot.fig)

    # Metadata
    mlt.utils.write_to_yaml(
        dict(
            type="mera-model-bias-std",
            model_dir_1=str(base_model_results_dir),
            model_dir_2=str(loc_adj_results_dir),
            is_mera=True,
        ),
        output_ffp.with_suffix(".yaml"),
    )


@app.command("nn-gmm-site-term-map")
def nn_gmm_site_term_map(
    model_dir: Path,
    output_dir: Path,
    im: str,
    prefix: str,
    grid_spacing: str = "500e/500e",
):
    """
    Generate site term map for the specified IM.
    I.e. map of delta_S2S
    """
    logger = nng.utils.setup_logging()

    site_terms_ffp = model_dir / "mera_site_term/site_res_df.parquet"
    if not site_terms_ffp.exists():
        raise FileNotFoundError(
            f"Site terms file not found: {site_terms_ffp}. "
            "Please run MERA analysis with site terms first."
        )

    run_config = nng.nn_gmm.load_config(model_dir / "run_config.yaml")
    logger.info(f"Loading IMDB data from {run_config.imdb_ffp}")
    with nng.DuckIMDB(run_config.imdb_ffp, readonly=True) as imdb:
        site_df = imdb.get_site_df()

    site_res_df = pd.read_parquet(site_terms_ffp)
    site_res_df = site_res_df.join(
        site_df[["site_id", "lon", "lat"]].set_index("site_id"), how="left"
    )

    plt_kwargs = {"water_color": "white"}
    spatial_plot = nng.plots_spatial.SpatialPlot(plot_kwargs=plt_kwargs)

    spatial_plot.plot_ratio(
        site_res_df,
        im,
        grid_spacing=grid_spacing,
        cmap_limits=(-0.5, 0.5, 1.0 / 10),
        cb_label=f"{nng.utils.get_nice_im_name(im)} Site Term",
        transparency=25,
    )

    spatial_plot.save(output_dir / f"{prefix}_site_term_map_{im}.png")


@app.command("ds-hazard-map")
def ds_hazard_map(
    imdb_ffp: Path,
    hazard_results_dir: Path,
    output_dir: Path,
    im: str,
    rp: int,
    title: str,
    filename_prefix: str,
    grid_spacing: str,
):
    """DS hazard map for a specified IM at a given return period."""
    nng.utils.setup_logging()

    with nng.DuckIMDB(imdb_ffp, readonly=True) as imdb:
        site_df = imdb.get_site_df().set_index("site_id")

    nng.plots_spatial.hazard_map(
        site_df,
        hazard_results_dir,
        im,
        [rp],
        output_dir,
        title=title,
        grid_spacing=grid_spacing,
        filename_prefix=filename_prefix,
    )


@app.command("total-hazard-map")
def total_hazard_map(
    imdb_ffp: Path,
    hazard_results_dir: Path,
    output_dir: Path,
    im: str,
    rp: int,
    title: str,
    filename_prefix: str,
    grid_spacing: str,
):
    """
    Total hazard map for a specified IM at a given return period,
    using CS25.6 for the fault component of hazard.
    """
    nng.utils.setup_logging()

    with nng.DuckIMDB(imdb_ffp, readonly=True) as imdb:
        site_df = imdb.get_site_df().set_index("site_id")

    nng.plots_spatial.hazard_map(
        site_df,
        hazard_results_dir,
        im,
        [rp],
        output_dir,
        add_flt_hazard=True,
        grid_spacing=grid_spacing,
        title=title,
        filename_prefix=filename_prefix,
    )


@app.command("hazard-ratio-map")
def hazard_ratio_map(
    results_dir_1: Path,
    results_dir_2: Path,
    imdb_ffp: Path,
    output_dir: Path,
    im: str,
    rp: int,
    cb_label_suffix: str,
    filename_prefix: str,
    grid_spacing: str,
    cb_max: float = 2.0,
):
    nng.utils.setup_logging()

    with nng.DuckIMDB(imdb_ffp, readonly=True) as imdb:
        site_df = imdb.get_site_df().set_index("site_id")

    nng.plots_spatial.hazard_ratio_map(
        site_df,
        results_dir_1,
        results_dir_2,
        im,
        [rp],
        output_dir,
        cb_label_suffix,
        filename_prefix,
        grid_spacing=grid_spacing,
        cb_max=cb_max,
    )


@app.command("site-uhs-plot")
def site_uhs_plot(
    base_nn_ds_hazard_dir: Path,
    loc_nn_ds_hazard_dir: Path,
    emp_ds_results_dir: Path,
    out_dir: Path,
):
    """UHS plots for NN-GMM model and empirical DS hazard for two sites and two IMs"""
    nng.utils.setup_logging()
    _fig_settings()

    site1, site2 = "HORC", "LHUS"
    rp1, rp2 = 475, 2475

    sites = [site1, site2]
    rps = [rp1, rp2]

    # CS fault UHS
    cs_flt_uhs = nng.hazard.compute_cs_flt_uhs(sites, rps)
    assert cs_flt_uhs.coords["im"].values.tolist() == nng.constants.PSA_KEYS

    # Base NN-GMM DS UHS
    base_nn_ds_uhs = nng.hazard.compute_nn_ds_uhs(
        base_nn_ds_hazard_dir, rps, sites=sites
    )
    assert base_nn_ds_uhs.coords["im"].values.tolist() == nng.constants.PSA_KEYS

    # Location NN-GMM DS UHS
    loc_nn_ds_uhs = nng.hazard.compute_nn_ds_uhs(loc_nn_ds_hazard_dir, rps, sites=sites)
    assert loc_nn_ds_uhs.coords["im"].values.tolist() == nng.constants.PSA_KEYS

    # Empirical DS UHS
    emp_ds_uhs = nng.hazard.compute_emp_ds_uhs(emp_ds_results_dir, rps, sites=sites)
    assert emp_ds_uhs.coords["im"].values.tolist() == nng.constants.PSA_KEYS

    site_txt_offset = (0.03, 0.97)
    bbox_dict = dict(boxstyle="round", fc="w", ec="0.8", alpha=0.8)
    fig, (ax1, ax2, ax3, ax4) = mlt.plotting.get_fig_axes(
        4,
        2,
        2,
        ind_figsize=nng.constants.FIG_SIZE,
        dpi=nng.constants.FIG_DPI,
    )

    # Site 1 - RP 1
    ax1.loglog(
        nng.constants.PSA_PERIODS,
        cs_flt_uhs.sel(site=site1, rp=rp1).values,
        label="Cybershake Fault UHS",
        color="black",
        linestyle="dashed",
        linewidth=nng.constants.FIG_LINEWIDTH,
    )
    ax1.loglog(
        nng.constants.PSA_PERIODS,
        emp_ds_uhs.sel(site=site1, rp=rp1).values,
        label="Empirical DS UHS",
        color="green",
        linewidth=nng.constants.FIG_LINEWIDTH,
    )
    ax1.loglog(
        nng.constants.PSA_PERIODS,
        base_nn_ds_uhs.sel(site=site1, rp=rp1).values,
        label="Base NN-GMM DS UHS",
        color="blue",
        linewidth=nng.constants.FIG_LINEWIDTH,
    )
    ax1.loglog(
        nng.constants.PSA_PERIODS,
        loc_nn_ds_uhs.sel(site=site1, rp=rp1).values,
        label="Location NN-GMM DS UHS",
        color="red",
        linewidth=nng.constants.FIG_LINEWIDTH,
    )

    ax1.set_ylim(0.005, 5)
    ax1.set_xlim(0.01, 10.0)
    ax1.grid(which="both", linewidth=0.5, alpha=0.5, linestyle="--")
    ax1.tick_params(labelbottom=False)
    ax1.set_ylabel("pSA [g]")
    ax1.legend(loc="lower left")

    ax1.text(
        site_txt_offset[0],
        site_txt_offset[1],
        f"{site1}\n{nng.utils.rp_to_poe_string(rp1)}",
        transform=ax1.transAxes,
        horizontalalignment="left",
        verticalalignment="top",
        bbox=bbox_dict,
    )

    # Site 1 - RP 2
    ax2.sharey(ax1)

    ax2.loglog(
        nng.constants.PSA_PERIODS,
        cs_flt_uhs.sel(site=site1, rp=rp2).values,
        label="Cybershake Fault UHS",
        color="black",
        linestyle="dashed",
        linewidth=nng.constants.FIG_LINEWIDTH,
    )
    ax2.loglog(
        nng.constants.PSA_PERIODS,
        emp_ds_uhs.sel(site=site1, rp=rp2).values,
        label="Empirical DS UHS",
        color="green",
        linewidth=nng.constants.FIG_LINEWIDTH,
    )
    ax2.loglog(
        nng.constants.PSA_PERIODS,
        base_nn_ds_uhs.sel(site=site1, rp=rp2).values,
        label="Base NN-GMM DS UHS",
        color="blue",
        linewidth=nng.constants.FIG_LINEWIDTH,
    )
    ax2.loglog(
        nng.constants.PSA_PERIODS,
        loc_nn_ds_uhs.sel(site=site1, rp=rp2).values,
        label="Location NN-GMM DS UHS",
        color="red",
        linewidth=nng.constants.FIG_LINEWIDTH,
    )

    ax2.sharex(ax1)
    ax2.sharey(ax1)
    ax2.tick_params(labelbottom=False, labelleft=False)
    ax2.grid(which="both", linewidth=0.5, alpha=0.5, linestyle="--")

    ax2.text(
        site_txt_offset[0],
        site_txt_offset[1],
        f"{site1}\n{nng.utils.rp_to_poe_string(rp2)}",
        transform=ax2.transAxes,
        horizontalalignment="left",
        verticalalignment="top",
        bbox=bbox_dict,
    )

    # Site 2 - RP 1
    ax3.loglog(
        nng.constants.PSA_PERIODS,
        cs_flt_uhs.sel(site=site2, rp=rp1).values,
        label="Cybershake Fault UHS",
        color="black",
        linestyle="dashed",
        linewidth=nng.constants.FIG_LINEWIDTH,
    )
    ax3.loglog(
        nng.constants.PSA_PERIODS,
        emp_ds_uhs.sel(site=site2, rp=rp1).values,
        label="Empirical DS UHS",
        color="green",
        linewidth=nng.constants.FIG_LINEWIDTH,
    )
    ax3.loglog(
        nng.constants.PSA_PERIODS,
        base_nn_ds_uhs.sel(site=site2, rp=rp1).values,
        label="Base NN-GMM DS UHS",
        color="blue",
        linewidth=nng.constants.FIG_LINEWIDTH,
    )
    ax3.loglog(
        nng.constants.PSA_PERIODS,
        loc_nn_ds_uhs.sel(site=site2, rp=rp1).values,
        label="Location NN-GMM DS UHS",
        color="red",
        linewidth=nng.constants.FIG_LINEWIDTH,
    )

    ax3.sharey(ax1)
    ax3.grid(which="both", linewidth=0.5, alpha=0.5, linestyle="--")
    ax3.set_xlabel("Vibration Period [s]")
    ax3.set_ylabel("pSA [g]")

    ax3.set_xlim(ax1.get_xlim())
    ax3.set_xticks([1.0e-02, 1.0e-01, 1.0e00, 1.0e01])
    ax3.set_xticklabels(
        [
            "$\\mathdefault{10^{-2}}$",
            "$\\mathdefault{10^{-1}}$",
            "$\\mathdefault{10^{0}}$",
            "",
        ]
    )

    ax3.text(
        site_txt_offset[0],
        site_txt_offset[1],
        f"{site2}\n{nng.utils.rp_to_poe_string(rp1)}",
        transform=ax3.transAxes,
        horizontalalignment="left",
        verticalalignment="top",
        bbox=bbox_dict,
    )

    # Site 2 - RP 2
    ax4.loglog(
        nng.constants.PSA_PERIODS,
        cs_flt_uhs.sel(site=site2, rp=rp2).values,
        label="Cybershake Fault UHS",
        color="black",
        linestyle="dashed",
        linewidth=nng.constants.FIG_LINEWIDTH,
    )
    ax4.loglog(
        nng.constants.PSA_PERIODS,
        emp_ds_uhs.sel(site=site2, rp=rp2).values,
        label="Empirical DS UHS",
        color="green",
        linewidth=nng.constants.FIG_LINEWIDTH,
    )
    ax4.loglog(
        nng.constants.PSA_PERIODS,
        base_nn_ds_uhs.sel(site=site2, rp=rp2).values,
        label="Base NN-GMM DS UHS",
        color="blue",
        linewidth=nng.constants.FIG_LINEWIDTH,
    )
    ax4.loglog(
        nng.constants.PSA_PERIODS,
        loc_nn_ds_uhs.sel(site=site2, rp=rp2).values,
        label="Location NN-GMM DS UHS",
        color="red",
        linewidth=nng.constants.FIG_LINEWIDTH,
    )

    ax4.sharex(ax1)
    ax4.sharey(ax1)
    ax4.tick_params(labelleft=False)
    ax4.grid(which="both", linewidth=0.5, alpha=0.5, linestyle="--")
    ax4.set_xlabel("Vibration Period [s]")

    ax4.text(
        site_txt_offset[0],
        site_txt_offset[1],
        f"{site2}\n{nng.utils.rp_to_poe_string(rp2)}",
        transform=ax4.transAxes,
        horizontalalignment="left",
        verticalalignment="top",
        bbox=bbox_dict,
    )

    fig.subplots_adjust(
        left=0.06, right=0.98, top=0.99, bottom=0.07, wspace=0.025, hspace=0.025
    )

    fig.savefig(out_dir / f"site_uhs_plot.{nng.constants.FIG_FORMAT}")
    plt.close(fig)


@app.command("site-hazard-plot")
def site_hazard_plot(
    base_nn_ds_dir: Path, loc_nn_ds_dir: Path, emp_ds_dir: Path, out_dir: Path
):
    """Hazard plots for two NN-GMM models and empirical DS hazard for two sites and two IMs"""
    nng.utils.setup_logging()
    _fig_settings()

    site1, site2 = "HORC", "LHUS"
    im1, im2 = "pSA_0.5", "pSA_5.0"

    # Load Cybershake fault hazard
    cs_flt_hazard = pd.read_pickle(
        nng.constants.HAZARD_RESOURCES_DIR / "flt/Cybershake_hazard_data.pkl"
    )

    # Load base NN-GMM DS hazard results
    base_ds_hazard = {
        cur_ffp.stem: pd.read_pickle(base_nn_ds_dir / f"{cur_ffp.stem}.pkl")
        for cur_ffp in base_nn_ds_dir.glob("*.pkl")
    }

    # Load location NN-GMM DS hazard results
    loc_ds_hazard = {
        cur_ffp.stem: pd.read_pickle(loc_nn_ds_dir / f"{cur_ffp.stem}.pkl")
        for cur_ffp in loc_nn_ds_dir.glob("*.pkl")
    }

    # Load empirical DS hazard results
    emp_ds_hazard = {
        cur_ffp.stem: pd.read_pickle(emp_ds_dir / f"{cur_ffp.stem}.pkl")
        for cur_ffp in emp_ds_dir.glob("*.pkl")
    }

    site_txt_offset = (0.03, 0.97)
    bbox_dict = dict(boxstyle="round", fc="w", ec="0.8", alpha=0.8)
    fig, (ax1, ax2, ax3, ax4) = mlt.plotting.get_fig_axes(
        4, 2, 2, ind_figsize=nng.constants.FIG_SIZE, dpi=nng.constants.FIG_DPI
    )

    # Site 1 - IM 1
    ax1.loglog(
        base_ds_hazard[site1]["total"][im1].index,
        base_ds_hazard[site1]["total"][im1].values,
        label="Base NN-GMM DS Hazard",
        color="blue",
        linewidth=nng.constants.FIG_LINEWIDTH,
    )
    ax1.loglog(
        loc_ds_hazard[site1]["total"][im1].index,
        loc_ds_hazard[site1]["total"][im1].values,
        label="Location NN-GMM DS Hazard",
        color="red",
        linewidth=nng.constants.FIG_LINEWIDTH,
    )
    ax1.loglog(
        emp_ds_hazard[site1]["total"][im1].index,
        emp_ds_hazard[site1]["total"][im1].values,
        label="Empirical DS Hazard",
        color="green",
        linewidth=nng.constants.FIG_LINEWIDTH,
    )
    ax1.loglog(
        cs_flt_hazard[im1].loc[site1].index.values,
        cs_flt_hazard[im1].loc[site1].values,
        label="Cybershake Fault Hazard",
        color="black",
        linestyle="dashed",
        linewidth=nng.constants.FIG_LINEWIDTH,
    )

    ax1.set_ylim(1e-4, 2.0)
    ax1.set_xlim(0.01, 2.0)
    ax1.grid(which="both", linewidth=0.5, alpha=0.5, linestyle="--")
    ax1.tick_params(labelbottom=False)
    ax1.set_ylabel("Annual Rate of Exceedance")
    ax1.text(
        site_txt_offset[0],
        site_txt_offset[1],
        site1,
        transform=ax1.transAxes,
        horizontalalignment="left",
        verticalalignment="top",
        # fontweight="bold",
        bbox=bbox_dict,
    )

    # Site 1 - IM 2
    ax2.sharey(ax1)
    ax2.sharex(ax1)

    ax2.loglog(
        base_ds_hazard[site1]["total"][im2].index,
        base_ds_hazard[site1]["total"][im2].values,
        label="Base NN-GMM DS Hazard",
        color="blue",
        linewidth=nng.constants.FIG_LINEWIDTH,
    )
    ax2.loglog(
        loc_ds_hazard[site1]["total"][im2].index,
        loc_ds_hazard[site1]["total"][im2].values,
        label="Location NN-GMM DS Hazard",
        color="red",
        linewidth=nng.constants.FIG_LINEWIDTH,
    )
    ax2.loglog(
        emp_ds_hazard[site1]["total"][im2].index,
        emp_ds_hazard[site1]["total"][im2].values,
        label="Empirical DS Hazard",
        color="green",
        linewidth=nng.constants.FIG_LINEWIDTH,
    )
    ax2.loglog(
        cs_flt_hazard[im2].loc[site1].index.values,
        cs_flt_hazard[im2].loc[site1].values,
        label="Cybershake Fault Hazard",
        color="black",
        linestyle="dashed",
        linewidth=nng.constants.FIG_LINEWIDTH,
    )

    ax2.tick_params(labelbottom=False, labelleft=False)
    ax2.grid(which="both", linewidth=0.5, alpha=0.5, linestyle="--")
    ax2.legend(loc="upper right")

    ax2.text(
        site_txt_offset[0],
        site_txt_offset[1],
        site1,
        transform=ax2.transAxes,
        horizontalalignment="left",
        verticalalignment="top",
        # fontweight="bold",
        bbox=bbox_dict,
    )

    # Site 2 - IM 1
    ax3.sharex(ax1)
    ax3.sharey(ax1)

    ax3.loglog(
        base_ds_hazard[site2]["total"][im1].index,
        base_ds_hazard[site2]["total"][im1].values,
        label="Base NN-GMM DS Hazard",
        color="blue",
        linewidth=nng.constants.FIG_LINEWIDTH,
    )
    ax3.loglog(
        loc_ds_hazard[site2]["total"][im1].index,
        loc_ds_hazard[site2]["total"][im1].values,
        label="Location NN-GMM DS Hazard",
        color="red",
        linewidth=nng.constants.FIG_LINEWIDTH,
    )
    ax3.loglog(
        emp_ds_hazard[site2]["total"][im1].index,
        emp_ds_hazard[site2]["total"][im1].values,
        label="Empirical DS Hazard",
        color="green",
        linewidth=nng.constants.FIG_LINEWIDTH,
    )
    ax3.loglog(
        cs_flt_hazard[im1].loc[site2].index.values,
        cs_flt_hazard[im1].loc[site2].values,
        label="Cybershake Fault Hazard",
        color="black",
        linestyle="dashed",
        linewidth=nng.constants.FIG_LINEWIDTH,
    )

    ax3.set_xlabel(nng.utils.get_nice_im_name(im1))
    ax3.set_ylabel("Annual Rate of Exceedance")
    ax3.grid(which="both", linewidth=0.5, alpha=0.5, linestyle="--")

    ax3.text(
        site_txt_offset[0],
        site_txt_offset[1],
        site2,
        transform=ax3.transAxes,
        horizontalalignment="left",
        verticalalignment="top",
        # fontweight="bold",
        bbox=bbox_dict,
    )

    # Site 2 - IM 2
    ax4.sharex(ax2)
    ax4.sharey(ax3)

    ax4.loglog(
        base_ds_hazard[site2]["total"][im2].index,
        base_ds_hazard[site2]["total"][im2].values,
        label="Base NN-GMM DS Hazard",
        color="blue",
        linewidth=nng.constants.FIG_LINEWIDTH,
    )
    ax4.loglog(
        loc_ds_hazard[site2]["total"][im2].index,
        loc_ds_hazard[site2]["total"][im2].values,
        label="Location NN-GMM DS Hazard",
        color="red",
        linewidth=nng.constants.FIG_LINEWIDTH,
    )
    ax4.loglog(
        emp_ds_hazard[site2]["total"][im2].index,
        emp_ds_hazard[site2]["total"][im2].values,
        label="Empirical DS Hazard",
        color="green",
        linewidth=nng.constants.FIG_LINEWIDTH,
    )
    ax4.loglog(
        cs_flt_hazard[im2].loc[site2].index.values,
        cs_flt_hazard[im2].loc[site2].values,
        label="Cybershake Fault Hazard",
        color="black",
        linestyle="dashed",
        linewidth=nng.constants.FIG_LINEWIDTH,
    )

    ax4.tick_params(labelleft=False)
    ax4.set_xlabel(nng.utils.get_nice_im_name(im2))
    ax4.grid(which="both", linewidth=0.5, alpha=0.5, linestyle="--")
    ax4.text(
        site_txt_offset[0],
        site_txt_offset[1],
        site2,
        transform=ax4.transAxes,
        horizontalalignment="left",
        verticalalignment="top",
        # fontweight="bold",
        bbox=bbox_dict,
    )

    # fig.tight_layout()
    fig.subplots_adjust(
        left=0.06, right=0.99, top=0.99, bottom=0.07, wspace=0.025, hspace=0.025
    )
    fig.savefig(out_dir / f"site_hazard_plot.{nng.constants.FIG_FORMAT}")
    plt.close(fig)


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
