import os
import io
from pathlib import Path

import torch
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.ticker as mticker
from matplotlib.legend_handler import HandlerTuple
import seaborn as sns
import typer

import nn_gmm as nng
import ml_tools as mlt
from qcore import nhm

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
if torch.mps.is_available():
    device = "mps"
print(f"Using device: {device.upper()}")

app = typer.Typer(add_completion=False, pretty_exceptions_short=True)


def _make_line_proxy(line):
    return mlines.Line2D(
        [],
        [],
        color=line.get_color(),
        linestyle=line.get_linestyle(),
        linewidth=1.0,
    )


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


def _interp_cs_hazard(
    cs_flt_hazard: pd.Series, model_ds_hazard: pd.Series
) -> pd.Series:
    """
    Helper function to inerpolate CS fault hazard
    to required IM levels
    """
    assert (cs_flt_hazard.index.values[0] >= model_ds_hazard.index.values[0]) and (
        cs_flt_hazard.index.values[-1] <= model_ds_hazard.index.values[-1]
    ), "CS fault hazard does not cover the same intensity measure levels as the DS hazard results."
    cs_flt_hazard_interp = np.exp(
        np.interp(
            np.log(model_ds_hazard.index.values),
            np.log(cs_flt_hazard.index.values),
            np.log(cs_flt_hazard.values.astype(float)),
        )
    )
    return pd.Series(cs_flt_hazard_interp, index=model_ds_hazard.index)


@app.command("mag-rrup-scatter")
def mag_rrup_scatter(
    imdb_ffp: Path,
    output_dir: Path,
    n_rrup_bins: int = 20,
    n_mag_bins: int = 20,
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
    site_event_df["tect_type"] = event_df.loc[
        site_event_df.event_int_id.values, "tect_type"
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
    crustal_mask = (site_event_df.tect_type == "ACTIVE_SHALLOW") | (
        site_event_df.tect_type == "VOLCANIC"
    )
    ax_scatter.scatter(
        site_event_df.loc[crustal_mask].rrup,
        site_event_df.loc[crustal_mask].magnitude,
        s=2.0,
        c="#4363d8",
        alpha=0.75,
        label="Active Shallow & Volcanic",
    )
    interface_mask = site_event_df.tect_type == "SUBDUCTION_INTERFACE"
    ax_scatter.scatter(
        site_event_df.loc[interface_mask].rrup,
        site_event_df.loc[interface_mask].magnitude,
        s=2.0,
        c="#911eb4",
        alpha=0.75,
        label="Subduction Interface",
    )
    slab_mask = site_event_df.tect_type == "SUBDUCTION_SLAB"
    ax_scatter.scatter(
        site_event_df.loc[slab_mask].rrup,
        site_event_df.loc[slab_mask].magnitude,
        s=2.0,
        c="#e6194B",
        alpha=0.75,
        label="Subduction Slab",
    )
    # ax_scatter.scatter(
    #     site_event_df.rrup,
    #     site_event_df.magnitude,
    #     s=2.0,
    #     c=color,
    #     alpha=0.75,
    # )

    ax_scatter.set_xlabel("Source-to-site distance, $R_{rup}$ (km)")
    ax_scatter.set_ylabel("Magnitude, $M_{W}$")
    ax_scatter.set_xscale("log")
    ax_scatter.set_xlim(0.1, 1000)
    ax_scatter.grid(which="both", linewidth=0.5, alpha=0.5, linestyle="--")

    ax_scatter.legend(loc="lower right")

    rrup_bins = np.logspace(np.log10(0.1), np.log10(1000), n_rrup_bins)

    crustal_counts, _ = np.histogram(site_event_df[crustal_mask].rrup, bins=rrup_bins)
    interface_counts, _ = np.histogram(
        site_event_df[interface_mask].rrup, bins=rrup_bins
    )
    slab_counts, _ = np.histogram(site_event_df[slab_mask].rrup, bins=rrup_bins)

    bin_widths = np.diff(rrup_bins)
    bin_centers = rrup_bins[:-1]

    ax_histx.bar(
        bin_centers,
        crustal_counts,
        width=bin_widths,
        align="edge",
        color="#4363d8",
        edgecolor="black",
        log=True,
    )
    ax_histx.bar(
        bin_centers,
        interface_counts,
        width=bin_widths,
        align="edge",
        color="#911eb4",
        edgecolor="black",
        log=True,
        bottom=crustal_counts,
    )
    ax_histx.bar(
        bin_centers,
        slab_counts,
        width=bin_widths,
        align="edge",
        color="#e6194B",
        edgecolor="black",
        log=True,
        bottom=crustal_counts + interface_counts,
    )

    ax_histx.set_xscale("log")
    ax_histx.set_xlim(0.1, 1000)
    ax_histx.spines[["top", "right"]].set_visible(False)
    ax_histx.set_xticklabels([])

    mag_bins = np.linspace(
        site_event_df.magnitude.min(), site_event_df.magnitude.max(), n_mag_bins + 1
    )

    crustal_mag_counts, _ = np.histogram(
        site_event_df[crustal_mask].magnitude, bins=mag_bins
    )
    interface_mag_counts, _ = np.histogram(
        site_event_df[interface_mask].magnitude, bins=mag_bins
    )
    slab_mag_counts, _ = np.histogram(site_event_df[slab_mask].magnitude, bins=mag_bins)

    bin_heights = np.diff(mag_bins)
    bin_bottoms = mag_bins[:-1]

    ax_histy.barh(
        bin_bottoms,
        crustal_mag_counts,
        height=bin_heights,
        align="edge",
        color="#4363d8",
        edgecolor="black",
        log=True,
    )
    ax_histy.barh(
        bin_bottoms,
        interface_mag_counts,
        height=bin_heights,
        align="edge",
        color="#911eb4",
        edgecolor="black",
        log=True,
        left=crustal_mag_counts,
    )
    ax_histy.barh(
        bin_bottoms,
        slab_mag_counts,
        height=bin_heights,
        align="edge",
        color="#e6194B",
        edgecolor="black",
        log=True,
        left=crustal_mag_counts + interface_mag_counts,
    )

    ax_histy.set_ylim(ax_scatter.get_ylim())
    ax_histy.spines[["top", "right"]].set_visible(False)
    ax_histy.set_yticklabels([])

    plt.savefig(output_dir / f"rrup_vs_mag.{nng.constants.FIG_FORMAT}")


@app.command("model-dev-dataset-hist")
def model_dev_dataset_hist(full_model_dir: Path, output_dir: Path, n_bins: int = 20):
    """
    Plot histogram of event magnitudes and site-to-source distances for model development dataset.
    """
    nng.utils.setup_logging()
    _fig_settings()

    run_config = nng.nn_gmm.load_config(full_model_dir / "run_config.yaml")
    train_record_ids = np.load(full_model_dir / "train_record_ids.npy")

    # Retrieve relevant data
    with nng.DuckIMDB(run_config.imdb_ffp, readonly=True) as imdb:
        record_info_df = imdb.get_record_info_df(record_int_ids=train_record_ids)
        event_df = imdb.get_event_df()
        rel_df = imdb.get_rel_df(rel_int_ids=record_info_df.rel_int_id.unique())
        record_info_df["site_event_int_id"] = nng.utils.get_site_event_int_id(
            record_info_df.site_int_id.values, record_info_df.event_int_id.values
        )
        site_event_df = imdb.get_site_event_df(
            site_event_int_ids=record_info_df.site_event_int_id.unique()
        )

    # Drop test events
    event_df = event_df.loc[~event_df.event_id.isin(run_config.test_events)]

    record_site_source_df = site_event_df.loc[record_info_df.site_event_int_id.values]
    rel_df["tect_type"] = event_df.loc[rel_df.event_int_id.values, "tect_type"].values

    rrup_bins = np.linspace(0, run_config.max_rrup, n_bins + 1)

    # Create the plot
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=nng.constants.FIG_SIZE, dpi=nng.constants.FIG_DPI
    )

    # data_to_plot = [rel_df.loc[rel_df.tect_type == tt, "magnitude"].values for tt in nng.constants.NN_TECT_TYPES]
    data_to_plot = [
        event_df.loc[event_df.tect_type == tt, "magnitude"].values
        for tt in nng.constants.NN_TECT_TYPES
    ]

    ax1.hist(
        data_to_plot,
        bins=n_bins,
        stacked=True,
        edgecolor="black",
        label=nng.constants.NN_TECT_TYPES_SHORT_NAME,
    )
    ax1.legend(loc="upper right")

    # ax1.hist(
    # rel_df.magnitude.values,
    # bins=n_bins,
    # color="#4363d8",
    # edgecolor="black",
    # density=False,
    # )
    ax1.grid(which="both", linewidth=0.5, alpha=0.5, linestyle="--")
    # ax1.set_xlim(rel_df.magnitude.min(), rel_df.magnitude.max())
    ax1.set_xlim(event_df.magnitude.min(), event_df.magnitude.max())
    # ax1.set_ylabel("Simulation Count")
    ax1.set_ylabel("Event Count", labelpad=0.25)
    ax1.set_xlabel("Magnitude")

    ax1.yaxis.set_major_locator(mticker.MultipleLocator(25))
    ax1.yaxis.set_minor_locator(mticker.MultipleLocator(12.5))
    # ax1.yaxis.set_major_formatter(
    # mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k" if x > 0 else 0)
    # )
    # ax1.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    ax2.hist(
        record_site_source_df.rrup.values,
        bins=rrup_bins,
        # color="#4363d8",
        edgecolor="black",
        density=False,
    )
    ax2.grid(linewidth=0.5, alpha=0.5, linestyle="--")
    ax2.set_xlabel("Source-to-site distance, $R_{rup}$ (km)")
    ax2.set_ylabel("Record Count")
    ax2.set_xlim(0, run_config.max_rrup)

    ax2.yaxis.set_major_locator(mticker.MultipleLocator(500_000))
    ax2.yaxis.set_minor_locator(mticker.MultipleLocator(250_000))
    ax2.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x/1_000_000:.1f}M" if x > 0 else 0)
    )
    # ax2.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    # fig.tight_layout()
    fig.subplots_adjust(left=0.065, right=0.98, top=0.98, bottom=0.165, wspace=0.225)
    # fig.subplots_adjust(left=0.07, right=0.98, top=0.98, bottom=0.165, wspace=0.225)
    fig.savefig(output_dir / f"model_dev_dataset_hist.{nng.constants.FIG_FORMAT}")
    plt.close(fig)


@app.command("ds-dataset-hist")
def ds_dataset_hist(full_model_dir: Path, output_dir: Path, n_bins: int = 20):
    """
    Plot histogram of event magnitudes and site-to-source distances for DS dataset.
    """
    nng.utils.setup_logging()
    _fig_settings()

    # Load DS source data
    ds_source_df, ds_erf_df = nng.utils.get_ds_source_data()

    run_config = nng.nn_gmm.load_config(full_model_dir / "run_config.yaml")
    train_record_ids = np.load(full_model_dir / "train_record_ids.npy")
    with nng.DuckIMDB(run_config.imdb_ffp, readonly=True) as imdb:
        record_info_df = imdb.get_record_info_df(record_int_ids=train_record_ids)
        site_df = imdb.get_site_df(add_nztm=True)

    site_df = site_df.loc[record_info_df.site_int_id.unique()]

    # Compute Rrup values for each site-source pair
    site_coords = site_df[["nztm_x", "nztm_y"]].values.astype(np.float32)
    source_coords = ds_source_df[["nztm_x", "nztm_y", "depth"]].values.astype(
        np.float32
    )
    rrup_values = (
        np.sqrt(
            (site_coords[:, 0][:, None] - source_coords[:, 0][None, :]) ** 2
            + (site_coords[:, 1][:, None] - source_coords[:, 1][None, :]) ** 2
            + (source_coords[:, 2][None, :] * 1000) ** 2
        )
        / 1000
    )

    # Drop any large than max_rrup
    rrup_values = rrup_values[rrup_values <= run_config.max_rrup]

    # Magnitude bins
    mag_bins_centres = ds_source_df.mag.unique()
    mag_bin_spacing = np.diff(mag_bins_centres).min()
    mag_bins = np.arange(
        mag_bins_centres.min() - mag_bin_spacing / 2,
        mag_bins_centres.max() + mag_bin_spacing / 2,
        mag_bin_spacing,
    )

    # Rrup bins
    rrup_bins = np.linspace(0, run_config.max_rrup, n_bins + 1)

    # Create the plot
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=nng.constants.FIG_SIZE, dpi=nng.constants.FIG_DPI
    )

    tect_types = ["ACTIVE_SHALLOW", "SUBDUCTION_SLAB"]
    data_to_plot = [
        ds_source_df.loc[ds_source_df.tectonic_type == tt, "mag"].values
        for tt in tect_types
    ]
    labels = ["Crustal", "Slab"]
    colors = ["tab:blue", "tab:red"]

    ax1.hist(
        data_to_plot,
        bins=mag_bins,
        stacked=True,
        edgecolor="black",
        label=labels,
        color=colors,
    )

    # ax1.hist(
    #     ds_source_df.mag.values,
    #     bins=mag_bins,
    #     color="#4363d8",
    #     edgecolor="black",
    #     density=False,
    # )
    ax1.grid(which="both", linewidth=0.5, alpha=0.5, linestyle="--")
    ax1.set_xlim(mag_bins.min(), mag_bins.max())
    ax1.set_ylabel("Event Count")
    ax1.set_xlabel("Magnitude")

    ax1.yaxis.set_major_locator(mticker.MultipleLocator(5000))
    ax1.yaxis.set_minor_locator(mticker.MultipleLocator(25000))
    ax1.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x/1_000:.0f}k" if x > 0 else 0)
    )
    # ax1.yaxis.set_major_formatter(mticker.ScalarFormatter())
    # ax1.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    ax2.hist(
        rrup_values, bins=rrup_bins, color="tab:blue", edgecolor="black", density=False
    )
    ax2.grid(linewidth=0.5, alpha=0.5, linestyle="--")
    ax2.set_xlabel("Source-to-site distance, $R_{rup}$ (km)")
    ax2.set_ylabel("Record Count")
    ax2.set_xlim(0, run_config.max_rrup)

    ax2.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x/1_000_000:.0f}M" if x > 0 else 0)
    )
    # ax2.yaxis.set_major_formatter(mticker.ScalarFormatter())
    # ax2.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    # fig.tight_layout()
    fig.subplots_adjust(left=0.075, right=0.98, top=0.98, bottom=0.165, wspace=0.225)
    fig.savefig(output_dir / f"ds_dataset_hist.{nng.constants.FIG_FORMAT}")
    plt.close(fig)


@app.command("mag-tect-type-dist")
def mag_tect_type_dist(imdb_ffp: Path, output_dir: Path, n_mag_bins: int = 20):
    """
    Creates two-panel figure showing magnitude distribution by tectonic type,
    and sum of recurrence probabilities by magnitude and tectonic type.
    """
    import seismic_hazard_analysis as sha

    nng.utils.setup_logging()
    _fig_settings()

    with nng.DuckIMDB(imdb_ffp, readonly=True) as imdb:
        event_df = imdb.get_event_df()

        slab_events = event_df[
            event_df.tect_type == "SUBDUCTION_SLAB"
        ].event_id.values.astype(str)
        slab_record_info_df = imdb.get_record_info_df(events=slab_events)
        slab_rel_df = imdb.get_rel_df(events=slab_events)
        assert slab_events.size == slab_rel_df.shape[0]

    flt_erf_df = nhm.load_nhm_df(str(nng.constants.NHM_FAULT_FFP))
    ds_erf_df = pd.read_csv(nng.constants.DS_ERF_FFP, index_col="rupture_name")
    ds_source_df = sha.nshm_2010.get_ds_source_df(nng.constants.DS_SOURCE_FFP)

    crustal_mask = (event_df.tect_type == "ACTIVE_SHALLOW") | (
        event_df.tect_type == "VOLCANIC"
    )
    interface_mask = event_df.tect_type == "SUBDUCTION_INTERFACE"
    slab_mask = event_df.tect_type == "SUBDUCTION_SLAB"

    event_df["ds_rupture_name"] = None
    for rel_int_id, row in slab_rel_df.iterrows():
        event_df.loc[row.event_int_id, "ds_rupture_name"] = (
            sha.nshm_2010.utils.create_ds_rupture_name(
                row.hypo_lat,
                row.hypo_lon,
                row.hypo_depth,
                row.magnitude,
                "SUBDUCTION_SLAB",
            )
        )

    event_df["recur_prob"] = np.nan
    flt_mask = event_df.event_id.isin(flt_erf_df.name.values.astype(str))
    event_df.loc[flt_mask, "recur_prob"] = (
        1.0
        / flt_erf_df.loc[
            event_df.loc[flt_mask, "event_id"].values, "recur_int_median"
        ].values
    )
    event_df.loc[slab_mask, "recur_prob"] = ds_erf_df.loc[
        event_df.loc[slab_mask, "ds_rupture_name"].values, "annual_rec_prob"
    ].values
    assert (
        event_df.recur_prob.notna().all()
    ), "Some events are missing recurrence probabilities"

    min_mag, max_mag = 5.5, 8.25
    mag_bins = np.linspace(min_mag, max_mag, n_mag_bins + 1)
    assert (
        event_df.magnitude.min() >= mag_bins[0]
        and event_df.magnitude.max() <= mag_bins[-1]
    ), "Magnitude bins do not cover the range of magnitudes in the event dataframe"

    event_df["mag_bin_ix"] = np.digitize(event_df.magnitude.values, mag_bins) - 1

    crustal_counts_series = (
        event_df.loc[crustal_mask]["mag_bin_ix"].value_counts().sort_index()
    )
    crustal_counts = np.zeros(n_mag_bins, dtype=int)
    crustal_counts[crustal_counts_series.index.values] = crustal_counts_series.values

    interface_counts_series = (
        event_df.loc[interface_mask]["mag_bin_ix"].value_counts().sort_index()
    )
    interface_counts = np.zeros(n_mag_bins, dtype=int)
    interface_counts[interface_counts_series.index.values] = (
        interface_counts_series.values
    )

    slab_counts_series = (
        event_df.loc[slab_mask]["mag_bin_ix"].value_counts().sort_index()
    )
    slab_counts = np.zeros(n_mag_bins, dtype=int)
    slab_counts[slab_counts_series.index.values] = slab_counts_series.values

    # crustal_counts, _ = np.histogram(event_df[crustal_mask].magnitude, bins=mag_bins)
    # interface_counts, _ = np.histogram(
    #     event_df[interface_mask].magnitude, bins=mag_bins
    # )
    # slab_counts, _ = np.histogram(event_df[slab_mask].magnitude, bins=mag_bins)

    bin_widths = np.diff(mag_bins)
    bin_centers = mag_bins[:-1]

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=nng.constants.FIG_SIZE, dpi=nng.constants.FIG_DPI
    )

    ax1.bar(
        bin_centers,
        crustal_counts,
        width=bin_widths,
        align="edge",
        label="Active Shallow Crustal & Volcanic",
        color="#4363d8",
        edgecolor="black",
    )
    ax1.bar(
        bin_centers,
        interface_counts,
        width=bin_widths,
        align="edge",
        label="Subduction Interface",
        color="#911eb4",
        edgecolor="black",
        bottom=crustal_counts,
    )
    ax1.bar(
        bin_centers,
        slab_counts,
        width=bin_widths,
        align="edge",
        label="Subduction Slab",
        color="#e6194B",
        edgecolor="black",
        bottom=crustal_counts + interface_counts,
    )
    ax1.set_xlabel("Magnitude")
    ax1.set_ylabel("Event Count")
    ax1.set_xlim(min_mag, max_mag)
    ax1.grid(which="both", linewidth=0.5, alpha=0.5, linestyle="--")
    ax1.legend()

    # formatter = mticker.ScalarFormatter(useMathText=True)
    # formatter.set_scientific(True)
    # formatter.set_powerlimits((0, 0))  # Always use scientific notation
    # ax1.yaxis.set_major_formatter(formatter)

    # Sum of recurrence probabilities for each magnitude bin and tectonic type
    crustal_rate_sum_series = (
        event_df.loc[crustal_mask]
        .groupby("mag_bin_ix")["recur_prob"]
        .sum()
        .sort_index()
    )
    crustal_rate_sums = np.zeros(n_mag_bins, dtype=float)
    crustal_rate_sums[crustal_rate_sum_series.index.values] = (
        crustal_rate_sum_series.values
    )
    interface_rate_sum_series = (
        event_df.loc[interface_mask]
        .groupby("mag_bin_ix")["recur_prob"]
        .sum()
        .sort_index()
    )
    interface_rate_sums = np.zeros(n_mag_bins, dtype=float)
    interface_rate_sums[interface_rate_sum_series.index.values] = (
        interface_rate_sum_series.values
    )
    slab_rate_sum_series = (
        event_df.loc[slab_mask].groupby("mag_bin_ix")["recur_prob"].sum().sort_index()
    )
    slab_rate_sums = np.zeros(n_mag_bins, dtype=float)
    slab_rate_sums[slab_rate_sum_series.index.values] = slab_rate_sum_series.values

    ax2.bar(
        bin_centers,
        crustal_rate_sums,
        width=bin_widths,
        align="edge",
        label="Active Shallow Crustal & Volcanic",
        color="#4363d8",
        edgecolor="black",
    )
    ax2.bar(
        bin_centers,
        interface_rate_sums,
        width=bin_widths,
        align="edge",
        label="Subduction Interface",
        color="#911eb4",
        edgecolor="black",
        bottom=crustal_rate_sums,
    )
    ax2.bar(
        bin_centers,
        slab_rate_sums,
        width=bin_widths,
        align="edge",
        label="Subduction Slab",
        color="#e6194B",
        edgecolor="black",
        bottom=crustal_rate_sums + interface_rate_sums,
    )
    ax2.set_xlabel("Magnitude")
    ax2.set_ylabel("Sum of Recurrence Probabilities")
    ax2.set_xlim(min_mag, max_mag)
    ax2.grid(which="both", linewidth=0.5, alpha=0.5, linestyle="--")
    ax2.set_ylim(0.0, 0.05)

    fig.tight_layout()
    fig.savefig(output_dir / f"mag_tect_type_dist.{nng.constants.FIG_FORMAT}")
    plt.close(fig)


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

    horc_location = (171.9599076, -43.53963553)
    lhus_location = (174.8936358, -41.23084665)

    region = [165.4, 179.6, -47.4, -36.2]
    spatial_plot = nng.plots_spatial.SpatialPlot(
        plot_topo=True,
        plot_highways=True,
        region=region,
        plot_kwargs={"water_color": "white"},
    )

    basin_specs = {
        Path(
            "/Users/claudy/dev/work/code/nn_gmm/nn_gmm/resources/basin_boundaries/Kaikoura_outline_WGS84.txt"
        ): {"fill": "green", "pen": "0.2p,green"},
        Path(
            "/Users/claudy/dev/work/code/nn_gmm/nn_gmm/resources/basin_boundaries/Marlborough_outline_WGS84.txt"
        ): {"fill": "orange", "pen": "0.2p,orange"},
        Path(
            "/Users/claudy/dev/work/code/nn_gmm/nn_gmm/resources/basin_boundaries/Wellington_outline_WGS84.txt"
        ): {"fill": "blue", "pen": "0.2p,blue"},
        Path(
            "/Users/claudy/dev/work/code/nn_gmm/nn_gmm/resources/basin_boundaries/Cheviot_outline_WGS84.txt"
        ): {"fill": "green", "pen": "0.2p,green"},
        Path(
            "/Users/claudy/dev/work/code/nn_gmm/nn_gmm/resources/basin_boundaries/Canterbury_outline_WGS84.txt"
        ): {"fill": "cyan", "pen": "0.2p,cyan"},
        Path(
            "/Users/claudy/dev/work/code/nn_gmm/nn_gmm/resources/basin_boundaries/Hanmer_outline_WGS84_v19p1.txt"
        ): {"fill": "green", "pen": "0.2p,green"},
        Path(
            "/Users/claudy/dev/work/code/nn_gmm/nn_gmm/resources/basin_boundaries/Nelson_outline_WGS84.txt"
        ): {"fill": "purple", "pen": "0.2p,purple"},
        Path(
            "/Users/claudy/dev/work/code/nn_gmm/nn_gmm/resources/basin_boundaries/BanksPeninsulaVolcanics_outline_WGS84.txt"
        ): {"fill": None, "pen": "0.2p,black"},
        Path(
            "/Users/claudy/dev/work/code/nn_gmm/nn_gmm/resources/basin_boundaries/WaikatoHauraki_outline_WGS84.txt"
        ): {"fill": "magenta", "pen": "0.2p,magenta"},
        Path(
            "/Users/claudy/dev/work/code/nn_gmm/nn_gmm/resources/basin_boundaries/NorthCanterbury_outline_WGS84.txt"
        ): {"fill": "green", "pen": "0.2p,green"},
    }

    spatial_plot.plot_basins(basin_specs=basin_specs)

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
        label="Sources Not Simulated",
        pen="0.5p,black",
    )

    spatial_plot.fig.plot(
        x=horc_location[0],
        y=horc_location[1],
        style="t0.25c",
        # fill="#4363d8",
        fill="#911eb4",
        pen="0.25p,black",
        label="HORC Site",
    )
    spatial_plot.fig.plot(
        x=lhus_location[0],
        y=lhus_location[1],
        style="t0.25c",
        # fill="#911eb4",
        fill="#3cb44b",
        pen="0.25p,black",
        label="LHUS Site",
    )

    # Coastline
    spatial_plot.plot_coastline()

    # Add legend
    spatial_plot.fig.legend(position="JTL+jTL+o0.2c", box="+gwhite+p1p")

    # Custom legend for modelled basins
    legend_spec_io = io.StringIO("""
N 2
S 0.1c r 0.25c green 0.2p,green 0.3c North Canterbury
S 0.1c r 0.25c orange 0.2p,orange 0.3c Marlborough
S 0.1c r 0.25c blue 0.2p,blue 0.3c Wellington
S 0.1c r 0.25c cyan 0.2p,cyan 0.3c Canterbury
S 0.1c r 0.25c purple 0.2p,purple 0.3c Nelson
S 0.1c r 0.25c magenta 0.2p,magenta 0.3c Waikato-Hauraki
        """)

    spatial_plot.fig.legend(
        spec=legend_spec_io, position="JBR+jBR+o0.2c+w7.5c", box="+gwhite+p1p"
    )

    spatial_plot.save(output_dir / f"nz_faults.{nng.constants.FIG_FORMAT}")


@app.command("mera-basin-site-term")
def mera_basin_site_term(
    base_model_results_dir: Path,
    loc_adj_results_dir: Path,
    output_dir: Path,
    min_period: float | None = None,
):
    """
    Plot MERA site term bias and standard deviation by basin group,
    for both the base model and location-specific model.
    """
    from mera import MeraResults

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
    periods = np.array([nng.utils.get_pSA_period(im) for im in ims])

    if min_period is not None:
        mask = periods >= min_period
        ims = ims[mask]
        periods = periods[mask]

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
        bias_ylim=(-0.3, 0.3),
        std_ylim=(0, 0.5),
        figsize=nng.constants.FIG_SIZE,
        dpi=nng.constants.FIG_DPI,
        main_wspace=0.175,
        left=0.07,
        x_axis_limits=(0.01 if min_period is None else min_period, 10.0),
    )

    bias_std_plot.add_categorial_results(base_site_term, True)
    bias_std_plot.add_categorial_results(loc_site_term, False, linestyle="--")
    bias_std_plot.add_legend(bias_std_plot.ax3)

    legend_elements = {
        mlines.Line2D([], [], color="black", linestyle="--", label="Location"),
        mlines.Line2D([], [], color="black", linestyle="-", label="Base"),
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


@app.command("mera-model-bias-std")
def mera_model_bias_std(
    base_model_results_dir: Path,
    loc_adj_results_dir: Path,
    output_dir: Path,
    full_base_model_results_dir: Path | None = None,
    full_loc_model_results_dir: Path | None = None,
    event_mera: bool = False,
    show_sim: bool = False,
):
    """
    Generate a plot comparing model bias and standard deviation
    for MERA results of two models.
    """
    from mera import MeraResults

    nng.utils.setup_logging()
    _fig_settings()

    def _load_duipuis(ffp: Path) -> pd.DataFrame:
        df = pd.read_csv(ffp, index_col="imName")
        pSA_ims = [im for im in df.index if im.startswith("pSA_")]
        df = df.loc[pSA_ims]
        df.index = [nng.utils.get_pSA_period(im) for im in df.index]
        return df

    base_mera_results_dir = base_model_results_dir / (
        "event_mera_site_term" if event_mera else "mera_site_term"
    )
    loc_adj_mera_results_dir = loc_adj_results_dir / (
        "event_mera_site_term" if event_mera else "mera_site_term"
    )

    # Check that MERA results exist
    if not (base_mera_results_dir).exists():
        raise FileNotFoundError(
            f"MERA results directory not found: {base_mera_results_dir}"
        )

    if not (loc_adj_mera_results_dir).exists():
        raise FileNotFoundError(
            f"MERA results directory not found: {loc_adj_mera_results_dir}"
        )

    base_mera_results = MeraResults.load_from_parquet(base_mera_results_dir)
    base_periods = [
        nng.utils.get_pSA_period(im)
        for im in base_mera_results.bias_std_df.index.values.astype(str)
    ]

    loc_adj_mera_results = MeraResults.load_from_parquet(loc_adj_mera_results_dir)
    loc_periods = [
        nng.utils.get_pSA_period(im)
        for im in loc_adj_mera_results.bias_std_df.index.values.astype(str)
    ]

    if full_base_model_results_dir is not None:
        full_base_test_mera_results = MeraResults.load_from_parquet(
            full_base_model_results_dir / "test_mera"
        )
        full_base_periods = [
            nng.utils.get_pSA_period(im)
            for im in full_base_test_mera_results.bias_std_df.index.values.astype(str)
        ]
        full_base_test_std_df = full_base_test_mera_results.bias_std_df.copy()
        full_base_test_std_df.index = full_base_periods

    if full_loc_model_results_dir is not None:
        full_loc_test_mera_results = MeraResults.load_from_parquet(
            full_loc_model_results_dir / "test_mera"
        )
        full_loc_periods = [
            nng.utils.get_pSA_period(im)
            for im in full_loc_test_mera_results.bias_std_df.index.values.astype(str)
        ]
        full_loc_test_std_df = full_loc_test_mera_results.bias_std_df.copy()
        full_loc_test_std_df.index = full_loc_periods

    bias_std_plot = nng.plots.BiasStdPlot(
        figsize=nng.constants.FIG_SIZE,
        dpi=nng.constants.FIG_DPI,
        bias_ylim=(-0.5, 0.5),
        # std_ylim=(0, 0.875),
        std_ylim=(0, 0.95),
        main_wspace=0.175,
        left=0.07,
        add_over_under_text=False if event_mera else True,
    )
    if event_mera:
        bias_std_plot.ax1.set_ylabel(
            "Event-level Model Prediction Bias, a", labelpad=-2
        )
        bias_std_plot.ax3.set_ylabel(
            r"Event-level Residual Standard Deviation, $\sigma$"
        )
    else:
        bias_std_plot.ax1.set_ylabel("Model prediction bias, a", labelpad=-2)
        bias_std_plot.ax3.set_ylabel(r"Residual Standard Deviation, $\sigma$")

    base_std_df = base_mera_results.bias_std_df.copy()
    base_std_df.index = base_periods

    loc_adj_std_df = loc_adj_mera_results.bias_std_df.copy()
    loc_adj_std_df.index = loc_periods

    if show_sim:
        # Lee NZ Validation 2022 study
        lee_nz_val_bias_std_df = pd.read_csv(
            nng.constants.LEE_NZ_VAL_BIAS_STD_FFP, index_col="imName"
        )
        lee_nz_val_bias_std_df = lee_nz_val_bias_std_df.loc[
            [row for row in lee_nz_val_bias_std_df.index if row.startswith("pSA")]
        ].set_index("imVal", drop=True)

        bias_std_plot.add_bias(
            lee_nz_val_bias_std_df["bias"],
            c="k",
            label="Lee et al. (2022)",
            linewidth=nng.constants.FIG_GROUP_LINEWIDTH,
        )

        dupuis_small_mag_interface = _load_duipuis(
            nng.constants.DUPUIS_SMALL_INT_BIAS_STD_FFP
        )
        dupuis_small_mag_slab = _load_duipuis(
            nng.constants.DUPUIS_SMALL_SLAB_BIAS_STD_FFP
        )

        bias_std_plot.add_bias(
            dupuis_small_mag_interface["bias"],
            c="gray",
            label="Dupuis et al. (2025) - Interface",
            linestyle="--",
            linewidth=nng.constants.FIG_GROUP_LINEWIDTH,
        )

        bias_std_plot.add_bias(
            dupuis_small_mag_slab["bias"],
            c="gray",
            label="Dupuis et al. (2025) - Slab",
            linestyle="-",
            linewidth=nng.constants.FIG_GROUP_LINEWIDTH,
        )

    # Model bias
    if full_base_model_results_dir is not None:
        bias_std_plot.add_bias(
            full_base_test_std_df["bias"],
            c="gray",
            linestyle="-",
            linewidth=nng.constants.FIG_GROUP_LINEWIDTH,
        )
    if full_loc_model_results_dir is not None:
        bias_std_plot.add_bias(
            full_loc_test_std_df["bias"],
            c="gray",
            linestyle="--",
            linewidth=nng.constants.FIG_GROUP_LINEWIDTH,
        )
    bias_std_plot.add_bias(
        base_std_df["bias"],
        c="blue",
        label="Surrogate",
        linestyle="-",
        linewidth=nng.constants.FIG_LINEWIDTH,
    )
    # bias_std_plot.add_bias(
    #     loc_adj_std_df["bias"],
    #     c="blue",
    #     label="Location",
    #     linestyle="--",
    #     linewidth=nng.constants.FIG_LINEWIDTH,
    # )

    if show_sim:
        sim_handles = [
            mlines.Line2D(
                [],
                [],
                color="k",
                label="Lee et al. (2022)",
                linewidth=nng.constants.FIG_GROUP_LINEWIDTH,
            ),
            mlines.Line2D(
                [],
                [],
                color="gray",
                linestyle="--",
                label="Dupuis et al. (2025) - Interface",
                linewidth=nng.constants.FIG_GROUP_LINEWIDTH,
            ),
            mlines.Line2D(
                [],
                [],
                color="gray",
                linestyle="-",
                label="Dupuis et al. (2025) - Slab",
                linewidth=nng.constants.FIG_GROUP_LINEWIDTH,
            ),
        ]
        sim_legend = bias_std_plot.ax1.legend(
            handles=sim_handles, loc="upper left", title="Simulation Validation Studies",
            # title_fontproperties={"weight": "bold"}
        )
        sim_legend._legend_box.align = "left"
        bias_std_plot.ax1.add_artist(sim_legend)

    surrogate_handles = [
        mlines.Line2D(
            [],
            [],
            color="blue",
            label="Surrogate",
            linestyle="-",
            linewidth=nng.constants.FIG_LINEWIDTH,
        )
    ]
    bias_std_plot.ax1.legend(handles=surrogate_handles, loc="lower right")

    # Total standard deviation
    if show_sim:
        bias_std_plot.add_std(
            lee_nz_val_bias_std_df["sigma"],
            c="k",
            linewidth=nng.constants.FIG_GROUP_LINEWIDTH,
        )

        bias_std_plot.add_std(
            dupuis_small_mag_interface["sigma"],
            c="gray",
            linestyle="--",
            linewidth=nng.constants.FIG_GROUP_LINEWIDTH,
        )

        bias_std_plot.add_std(
            dupuis_small_mag_slab["sigma"],
            c="gray",
            linestyle="-",
            linewidth=nng.constants.FIG_GROUP_LINEWIDTH,
        )

    if full_base_model_results_dir is not None:
        bias_std_plot.add_std(
            full_base_test_std_df["sigma"],
            c="gray",
            label=r"Test: Total, $\sigma$",
            linestyle="-",
            linewidth=nng.constants.FIG_GROUP_LINEWIDTH,
        )
    if full_loc_model_results_dir is not None:
        bias_std_plot.add_std(
            full_loc_test_std_df["sigma"],
            c="gray",
            linestyle="--",
            linewidth=nng.constants.FIG_GROUP_LINEWIDTH,
        )

    bias_std_plot.add_std(
        base_std_df["sigma"],
        c="blue",
        label=r"Total, $\sigma$",
        linestyle="solid",
        linewidth=nng.constants.FIG_LINEWIDTH,
    )
    # bias_std_plot.add_std(
    #     loc_adj_std_df["sigma"],
    #     c="blue",
    #     linestyle="--",
    #     linewidth=nng.constants.FIG_LINEWIDTH,
    # )

    # # Between-event standard deviation
    # bias_std_plot.add_std(
    #     base_std_df["tau"],
    #     c="purple",
    #     label=r"Between-event, $\tau$",
    #     linestyle="solid",
    #     linewidth=nng.constants.FIG_GROUP_LINEWIDTH,
    # )
    # bias_std_plot.add_std(
    #     loc_adj_std_df["tau"],
    #     c="purple",
    #     linestyle="--",
    #     linewidth=nng.constants.FIG_GROUP_LINEWIDTH,
    # )

    # # Site-term standard deviation
    # bias_std_plot.add_std(
    #     base_std_df["phi_S2S"],
    #     c="red",
    #     linestyle="solid",
    #     label=r"Site-to-site, $\phi_{S2S}$",
    #     linewidth=nng.constants.FIG_GROUP_LINEWIDTH,
    # )
    # bias_std_plot.add_std(
    #     loc_adj_std_df["phi_S2S"],
    #     c="red",
    #     linestyle="--",
    #     linewidth=nng.constants.FIG_GROUP_LINEWIDTH,
    # )

    # # Remaining standard deviation
    # bias_std_plot.add_std(
    #     base_std_df["phi_w"], c="green", linestyle="solid", label=r"Remaining, $\phi_w$",
    #     linewidth=nng.constants.FIG_GROUP_LINEWIDTH,
    # )
    # bias_std_plot.add_std(
    #     loc_adj_std_df["phi_w"],
    #     c="green",
    #     linestyle="dashed",
    #     linewidth=nng.constants.FIG_GROUP_LINEWIDTH,
    # )

    # bias_std_plot.add_legend(bias_std_plot.ax3)
    bias_std_plot.ax3.yaxis.set_major_locator(plt.MultipleLocator(0.2))

    plot_type = "event_mera_model_bias_std" if event_mera else "mera_model_bias_std"
    output_ffp = output_dir / f"{plot_type}.{nng.constants.FIG_FORMAT}"
    bias_std_plot.fig.savefig(output_ffp)
    plt.close(bias_std_plot.fig)


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

    spatial_plot = nng.plots_spatial.SpatialPlot(plot_kwargs={"water_color": "white"})

    spatial_plot.plot_ratio(
        site_res_df,
        im,
        grid_spacing=grid_spacing,
        cmap_limits=(-0.5, 0.5, 1.0 / 10),
        cb_label=f"{nng.utils.get_nice_im_name(im)} Site Term",
        transparency=25,
    )
    spatial_plot.add_main_city_labels()

    spatial_plot.fig.text(
        position="TL",
        text=f"IM: {nng.utils.get_nice_im_name(im)}",
        offset="0.5c/-0.5c",
        font=nng.constants.GMT_FIG_FONT_LABEL,
    )

    spatial_plot.save(
        output_dir / f"{prefix}_site_term_map_{nng.utils.get_im_filename(im)}.png"
    )


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
        save_metadata=False,
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
        save_metadata=False,
    )


@app.command("site-uhs-plot")
def site_uhs_plot(
    base_nn_ds_hazard_dir: Path,
    loc_nn_ds_hazard_dir: Path,
    emp_ds_results_dir: Path,
    out_dir: Path,
):
    """
    UHS plots for NN-GMM model and empirical
    DS hazard for two sites and two IMs
    """
    RP_Y_LIMITS = {72: (1e-3, 1.0), 475: (1e-3, 5), 2475: (1e-2, 5)}

    def _plot_uhs(
        ax: plt.Axes,
        cur_flt_uhs: xr.DataArray,
        cur_emp_ds_uhs: xr.DataArray,
        cur_base_ds_uhs: xr.DataArray,
        cur_loc_ds_uhs: xr.DataArray,
    ):
        # Fault
        (flt_line,) = ax.loglog(
            nng.constants.PSA_PERIODS,
            cur_flt_uhs.values,
            color="black",
            linestyle="dashdot",
            linewidth=nng.constants.FIG_LINEWIDTH,
        )
        # DS
        (emp_ds_line,) = ax.loglog(
            nng.constants.PSA_PERIODS,
            cur_emp_ds_uhs.values,
            color="green",
            linestyle="dashed",
            linewidth=nng.constants.FIG_LINEWIDTH,
        )
        (base_ds_line,) = ax.loglog(
            nng.constants.PSA_PERIODS,
            cur_base_ds_uhs.values,
            color="blue",
            linestyle="dashed",
            linewidth=nng.constants.FIG_LINEWIDTH,
        )
        (loc_ds_line,) = ax.loglog(
            nng.constants.PSA_PERIODS,
            cur_loc_ds_uhs.values,
            color="red",
            linestyle="dashed",
            linewidth=nng.constants.FIG_LINEWIDTH,
        )
        # Total
        (emp_total_line,) = ax.loglog(
            nng.constants.PSA_PERIODS,
            (cur_emp_ds_uhs + cur_flt_uhs).values,
            color="green",
            linewidth=nng.constants.FIG_LINEWIDTH,
        )
        (base_total_line,) = ax.loglog(
            nng.constants.PSA_PERIODS,
            (cur_base_ds_uhs + cur_flt_uhs).values,
            color="blue",
            linewidth=nng.constants.FIG_LINEWIDTH,
        )
        (loc_total_line,) = ax.loglog(
            nng.constants.PSA_PERIODS,
            (cur_loc_ds_uhs + cur_flt_uhs).values,
            color="red",
            linewidth=nng.constants.FIG_LINEWIDTH,
        )

        return (
            flt_line,
            emp_ds_line,
            base_ds_line,
            loc_ds_line,
            emp_total_line,
            base_total_line,
            loc_total_line,
        )

    nng.utils.setup_logging()
    _fig_settings()

    sites = site1, site2 = ["HORC", "LHUS"]
    rps = [72, 475, 2475]
    n_rps = len(rps)

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
    fig, (axs) = mlt.plotting.get_fig_axes(
        n_rps * 2,
        2,
        -1,
        ind_figsize=nng.constants.FIG_SIZE,
        dpi=nng.constants.FIG_DPI,
    )

    for i in range(n_rps):
        ax_ix = i * 2

        # Site 1
        cur_ax = axs[ax_ix]
        cur_flt_uhs = cs_flt_uhs.sel(site=site1, rp=rps[i])
        cur_emp_ds_uhs = emp_ds_uhs.sel(site=site1, rp=rps[i])
        cur_base_ds_uhs = base_nn_ds_uhs.sel(site=site1, rp=rps[i])
        cur_loc_ds_uhs = loc_nn_ds_uhs.sel(site=site1, rp=rps[i])

        (
            cur_flt_line,
            cur_emp_ds_line,
            cur_base_ds_line,
            cur_loc_ds_line,
            cur_emp_total_line,
            cur_base_total_line,
            cur_loc_total_line,
        ) = _plot_uhs(
            cur_ax, cur_flt_uhs, cur_emp_ds_uhs, cur_base_ds_uhs, cur_loc_ds_uhs
        )

        cur_ax.set_ylim(*RP_Y_LIMITS[rps[i]])
        cur_ax.set_xlim(0.01, 10.0)
        cur_ax.grid(which="both", linewidth=0.5, alpha=0.5, linestyle="--")
        cur_ax.set_ylabel("pSA [g]")
        if i < n_rps - 1:
            cur_ax.tick_params(labelbottom=False)
        else:
            cur_ax.set_xlabel("Vibration Period [s]")

        cur_ax.text(
            site_txt_offset[0],
            site_txt_offset[1],
            f"{site1}\n{nng.utils.rp_to_poe_string(rps[i])}",
            transform=cur_ax.transAxes,
            horizontalalignment="left",
            verticalalignment="top",
            bbox=bbox_dict,
        )

        if i == 0:
            # Legend
            legend_1 = cur_ax.legend(
                handles=[
                    _make_line_proxy(cur_emp_total_line),
                    _make_line_proxy(cur_base_total_line),
                    _make_line_proxy(cur_loc_total_line),
                ],
                labels=["Empirical GM LT", "Base Model", "Location Model"],
                loc="lower left",
                # bbox_to_anchor=(0.0, 0.35),
            )
            cur_ax.add_artist(legend_1)

            axs[ax_ix + 1].legend(
                handles=[
                    _make_line_proxy(cur_flt_line),
                    (
                        _make_line_proxy(cur_emp_ds_line),
                        _make_line_proxy(cur_base_ds_line),
                        _make_line_proxy(cur_loc_ds_line),
                    ),
                    (
                        _make_line_proxy(cur_emp_total_line),
                        _make_line_proxy(cur_base_total_line),
                        _make_line_proxy(cur_loc_total_line),
                    ),
                ],
                labels=["Fault", "Distributed", "Total"],
                loc="lower left",
                handler_map={tuple: HandlerTuple(ndivide=None, pad=0.6)},
                handlelength=6.0,
            )

        # Site 2
        cur_ax = axs[ax_ix + 1]
        cur_flt_uhs = cs_flt_uhs.sel(site=site2, rp=rps[i])
        cur_emp_ds_uhs = emp_ds_uhs.sel(site=site2, rp=rps[i])
        cur_base_ds_uhs = base_nn_ds_uhs.sel(site=site2, rp=rps[i])
        cur_loc_ds_uhs = loc_nn_ds_uhs.sel(site=site2, rp=rps[i])

        _plot_uhs(cur_ax, cur_flt_uhs, cur_emp_ds_uhs, cur_base_ds_uhs, cur_loc_ds_uhs)

        cur_ax.text(
            site_txt_offset[0],
            site_txt_offset[1],
            f"{site2}\n{nng.utils.rp_to_poe_string(rps[i])}",
            transform=cur_ax.transAxes,
            horizontalalignment="left",
            verticalalignment="top",
            bbox=bbox_dict,
        )

        cur_ax.set_ylim(*RP_Y_LIMITS[rps[i]])
        cur_ax.set_xlim(0.01, 10.0)
        cur_ax.grid(which="both", linewidth=0.5, alpha=0.5, linestyle="--")
        if i < n_rps - 1:
            cur_ax.tick_params(labelbottom=False, labelleft=False)
        else:
            cur_ax.tick_params(labelbottom=True, labelleft=False)
            cur_ax.set_xlabel("Vibration Period [s]")

    fig.subplots_adjust(
        left=0.0825, right=0.98, top=0.99, bottom=0.05, wspace=0.05, hspace=0.025
    )

    visible_labels = [
        label
        for label in axs[-2].get_xticklabels()
        if label.get_visible()
        and axs[-2].get_xlim()[0] <= label.get_position()[0] <= axs[-2].get_xlim()[1]
    ]
    if visible_labels:
        visible_labels[-1].set_visible(False)
    fig.savefig(out_dir / f"site_uhs_plot.{nng.constants.FIG_FORMAT}")
    plt.close(fig)


@app.command("single-site-ds-hazard-plot")
def single_site_ds_hazard_plot(
    site: str,
    im: str,
    base_nn_ds_dir: Path,
    loc_nn_ds_dir: Path,
    emp_ds_dir: Path,
    out_dir: Path,
):
    """
    DS Hazard plot for two NN-GMM models and
    empirical DS hazard for one site and one IM,
    broken down by tectonic source type.
    """
    nng.utils.setup_logging()
    _fig_settings()

    # Load base NN-GMM DS hazard results
    base_ds_hazard = {
        cur_ffp.stem: pd.read_pickle(base_nn_ds_dir / f"{cur_ffp.stem}.pkl")
        for cur_ffp in base_nn_ds_dir.glob("*.pkl")
    }
    # base_ds_hazard = base_ds_hazard[site]["total"][im]

    # Load location NN-GMM DS hazard results
    loc_ds_hazard = {
        cur_ffp.stem: pd.read_pickle(loc_nn_ds_dir / f"{cur_ffp.stem}.pkl")
        for cur_ffp in loc_nn_ds_dir.glob("*.pkl")
    }
    # loc_ds_hazard = loc_ds_hazard[site]["total"][im]

    # Load empirical DS hazard results
    emp_ds_hazard = {
        cur_ffp.stem: pd.read_pickle(emp_ds_dir / f"{cur_ffp.stem}.pkl")
        for cur_ffp in emp_ds_dir.glob("*.pkl")
    }
    # emp_ds_hazard = emp_ds_hazard[site]["total"][im]
    assert emp_ds_hazard[site]["total"][im].index.equals(
        base_ds_hazard[site]["total"][im].index
    ), "Base DS hazard and empirical DS hazard do not have the same intensity measure levels"
    assert emp_ds_hazard[site]["total"][im].index.equals(
        loc_ds_hazard[site]["total"][im].index
    ), "Location DS hazard and empirical DS hazard do not have the same intensity measure levels"

    fig, ax = plt.subplots(figsize=nng.constants.FIG_SIZE, dpi=nng.constants.FIG_DPI)

    _plot_poe_lines(ax)

    # rp_475_prob = sha.utils.rp_to_prob(475)
    # ax.axhline(
    #     rp_475_prob,
    #     linestyle="dashed",
    #     color=nng.constants.DARKGRAY,
    #     linewidth=nng.constants.FIG_MINOR_LINEWIDTH,
    # )
    # ax.text(
    #     0.01,
    #     rp_475_prob * 1.05,
    #     "10% in 50 years",
    #     transform=ax.get_yaxis_transform(),
    #     va="bottom",
    #     ha="left",
    #     color=nng.constants.DARKGRAY,
    # )
    # rp_2475_prob = sha.utils.rp_to_prob(2475)
    # ax.axhline(
    #     rp_2475_prob,
    #     linestyle="dashed",
    #     color=nng.constants.DARKGRAY,
    #     linewidth=nng.constants.FIG_MINOR_LINEWIDTH,
    # )
    # ax.text(
    #     0.01,
    #     rp_2475_prob * 1.05,
    #     "2% in 50 years",
    #     transform=ax.get_yaxis_transform(),
    #     va="bottom",
    #     ha="left",
    #     color=nng.constants.DARKGRAY,
    # )

    (emp_ds_line,) = ax.loglog(
        emp_ds_hazard[site]["total"][im].index.values,
        emp_ds_hazard[site]["total"][im].values,
        label="Empirical",
        color="green",
        linewidth=nng.constants.FIG_LINEWIDTH,
        linestyle="solid",
    )
    (base_ds_line,) = ax.loglog(
        base_ds_hazard[site]["total"][im].index.values,
        base_ds_hazard[site]["total"][im].values,
        label="Base Model",
        color="blue",
        linestyle="solid",
        linewidth=nng.constants.FIG_LINEWIDTH,
    )
    # (loc_ds_line,) = ax.loglog(
    #     loc_ds_hazard[site]["total"][im].index.values,
    #     loc_ds_hazard[site]["total"][im].values,
    #     label="Location Model DS Hazard",
    #     color="red",
    #     linestyle="solid",
    #     linewidth=nng.constants.FIG_LINEWIDTH,
    # )

    # Crustal DS
    (crustal_emp_ds_line,) = ax.loglog(
        emp_ds_hazard[site]["crustal"][im].index.values,
        emp_ds_hazard[site]["crustal"][im].values,
        label="Empirical Crustal",
        color="green",
        # linewidth=nng.constants.FIG_LINEWIDTH,
        linewidth=nng.constants.FIG_GROUP_LINEWIDTH,
        linestyle="dashed",
    )
    (crustal_base_ds_line,) = ax.loglog(
        base_ds_hazard[site]["crustal"][im].index.values,
        base_ds_hazard[site]["crustal"][im].values,
        label="Base Model Crustal DS Hazard",
        color="blue",
        linestyle="dashed",
        linewidth=nng.constants.FIG_GROUP_LINEWIDTH,
        # linewidth=nng.constants.FIG_LINEWIDTH,
    )
    # (crustal_loc_ds_line,) = ax.loglog(
    #     loc_ds_hazard[site]["crustal"][im].index.values,
    #     loc_ds_hazard[site]["crustal"][im].values,
    #     label="Location Model Crustal DS Hazard",
    #     color="red",
    #     linestyle="dashed",
    #     linewidth=nng.constants.FIG_LINEWIDTH,
    # )

    # Subduction Slab
    (slab_emp_ds_line,) = ax.loglog(
        emp_ds_hazard[site]["subduction"][im].index.values,
        emp_ds_hazard[site]["subduction"][im].values,
        label="Empirical Subduction Slab",
        color="green",
        # linewidth=nng.constants.FIG_LINEWIDTH,
        linewidth=nng.constants.FIG_GROUP_LINEWIDTH,
        linestyle="dashdot",
    )
    (slab_base_ds_line,) = ax.loglog(
        base_ds_hazard[site]["subduction_slab"][im].index.values,
        base_ds_hazard[site]["subduction_slab"][im].values,
        label="Base Model Subduction Slab",
        color="blue",
        linestyle="dashdot",
        # linewidth=nng.constants.FIG_LINEWIDTH,
        linewidth=nng.constants.FIG_GROUP_LINEWIDTH,
    )
    # (slab_loc_ds_line,) = ax.loglog(
    #     loc_ds_hazard[site]["subduction_slab"][im].index.values,
    #     loc_ds_hazard[site]["subduction_slab"][im].values,
    #     label="Location Model Subduction Slab DS Hazard",
    #     color="red",
    #     linestyle="dashdot",
    #     linewidth=nng.constants.FIG_LINEWIDTH,
    # )

    ax.set_ylim(1e-5, 0.1)
    ax.set_xlim(0.01, 1.0)
    ax.grid(which="both", linewidth=0.5, alpha=0.5, linestyle="--")
    ax.set_ylabel("Annual Rate of Exceedance")
    ax.text(
        0.5,
        0.97,
        f"{site} - Distributed Seismicity Hazard",
        transform=ax.transAxes,
        horizontalalignment="center",
        verticalalignment="top",
        bbox=dict(boxstyle="round", fc="w", ec="0.8", alpha=0.5),
    )
    ax.set_xlabel(nng.utils.get_nice_im_name(im))

    # Legend
    legend_1 = ax.legend(
        handles=[
            (_make_line_proxy(emp_ds_line), _make_line_proxy(base_ds_line)),
            (
                _make_line_proxy(crustal_emp_ds_line),
                _make_line_proxy(crustal_base_ds_line),
            ),
            (_make_line_proxy(slab_emp_ds_line), _make_line_proxy(slab_base_ds_line)),
        ],
        labels=["Total", "Crustal", "Subduction Slab"],
        loc="lower left",
        handler_map={tuple: HandlerTuple(ndivide=None, pad=0.6)},
        handlelength=4.0,
        # bbox_to_anchor=(1.0, 0.875),
    )
    ax.add_artist(legend_1)

    ax.legend(
        handles=[_make_line_proxy(emp_ds_line), _make_line_proxy(base_ds_line)],
        labels=["Empirical", "Surrogate: Base"],
        loc="upper right",
    )

    fig.tight_layout()
    fig.savefig(
        out_dir
        / f"{site}_{nng.utils.get_im_filename(im)}_DS_hazard_plot.{nng.constants.FIG_FORMAT}"
    )
    plt.close(fig)


@app.command("single-site-hazard-plot")
def single_site_hazard_plot(
    site: str,
    im: str,
    base_nn_ds_dir: Path,
    loc_nn_ds_dir: Path,
    emp_ds_dir: Path,
    out_dir: Path,
):
    """
    Hazard plot (CS Fault, DS - Model, Total) for
    two NN-GMM models and empirical DS hazard
    for one site and one IM
    """
    nng.utils.setup_logging()
    _fig_settings()

    # Load Cybershake fault hazard
    cs_flt_hazard = pd.read_pickle(
        nng.constants.HAZARD_RESOURCES_DIR / "flt/Cybershake_hazard_data.pkl"
    )
    cs_flt_hazard = cs_flt_hazard[im].loc[site]

    # Load base NN-GMM DS hazard results
    base_ds_hazard = {
        cur_ffp.stem: pd.read_pickle(base_nn_ds_dir / f"{cur_ffp.stem}.pkl")
        for cur_ffp in base_nn_ds_dir.glob("*.pkl")
    }
    base_ds_hazard = base_ds_hazard[site]["total"][im]

    # Load location NN-GMM DS hazard results
    loc_ds_hazard = {
        cur_ffp.stem: pd.read_pickle(loc_nn_ds_dir / f"{cur_ffp.stem}.pkl")
        for cur_ffp in loc_nn_ds_dir.glob("*.pkl")
    }
    loc_ds_hazard = loc_ds_hazard[site]["total"][im]

    # Load empirical DS hazard results
    emp_ds_hazard = {
        cur_ffp.stem: pd.read_pickle(emp_ds_dir / f"{cur_ffp.stem}.pkl")
        for cur_ffp in emp_ds_dir.glob("*.pkl")
    }
    emp_ds_hazard = emp_ds_hazard[site]["total"][im]
    assert emp_ds_hazard.index.equals(
        base_ds_hazard.index
    ), "Base DS hazard and empirical DS hazard do not have the same intensity measure levels"
    assert emp_ds_hazard.index.equals(
        loc_ds_hazard.index
    ), "Location DS hazard and empirical DS hazard do not have the same intensity measure levels"

    # Interpolate CS fault hazard
    cs_flt_hazard = _interp_cs_hazard(cs_flt_hazard, base_ds_hazard)

    fig, ax = plt.subplots(figsize=nng.constants.FIG_SIZE, dpi=nng.constants.FIG_DPI)

    _plot_poe_lines(ax)

    # rp_475_prob = sha.utils.rp_to_prob(475)
    # ax.axhline(
    #     rp_475_prob,
    #     linestyle="dashed",
    #     color=nng.constants.DARKGRAY,
    #     linewidth=nng.constants.FIG_MINOR_LINEWIDTH,
    # )
    # ax.text(
    #     0.01,
    #     rp_475_prob * 1.05,
    #     "10% in 50 years",
    #     transform=ax.get_yaxis_transform(),
    #     va="bottom",
    #     ha="left",
    #     color=nng.constants.DARKGRAY,
    # )
    # rp_2475_prob = sha.utils.rp_to_prob(2475)
    # ax.axhline(
    #     rp_2475_prob,
    #     linestyle="dashed",
    #     color=nng.constants.DARKGRAY,
    #     linewidth=nng.constants.FIG_MINOR_LINEWIDTH,
    # )
    # ax.text(
    #     0.01,
    #     rp_2475_prob * 1.05,
    #     "2% in 50 years",
    #     transform=ax.get_yaxis_transform(),
    #     va="bottom",
    #     ha="left",
    #     color=nng.constants.DARKGRAY,
    # )

    (base_ds_line,) = ax.loglog(
        base_ds_hazard.index.values,
        base_ds_hazard.values,
        color="blue",
        linestyle="dashed",
        # linewidth=nng.constants.FIG_LINEWIDTH,
        linewidth=nng.constants.FIG_GROUP_LINEWIDTH,
    )
    # (loc_ds_line,) = ax.loglog(
    #     loc_ds_hazard.index.values,
    #     loc_ds_hazard.values,
    #     label="Location Model DS Hazard",
    #     color="red",
    #     linestyle="dashed",
    #     # linewidth=nng.constants.FIG_LINEWIDTH,
    #     linewidth=nng.constants.FIG_GROUP_LINEWIDTH,
    # )
    (emp_ds_line,) = ax.loglog(
        emp_ds_hazard.index.values,
        emp_ds_hazard.values,
        color="green",
        linewidth=nng.constants.FIG_GROUP_LINEWIDTH,
        # linewidth=nng.constants.FIG_LINEWIDTH,
        linestyle="dashed",
    )
    (cs_flt_line,) = ax.loglog(
        cs_flt_hazard.index.values,
        cs_flt_hazard.values,
        color="black",
        linestyle="dashdot",
        # linewidth=nng.constants.FIG_LINEWIDTH,
        linewidth=nng.constants.FIG_GROUP_LINEWIDTH,
    )
    (base_total_line,) = ax.loglog(
        base_ds_hazard.index.values,
        base_ds_hazard.values + cs_flt_hazard.values,
        color="blue",
        linestyle="solid",
        linewidth=nng.constants.FIG_LINEWIDTH,
    )
    # (loc_total_line,) = ax.loglog(
    #     loc_ds_hazard.index.values,
    #     loc_ds_hazard.values + cs_flt_hazard.values,
    #     color="red",
    #     linestyle="solid",
    #     label="Location Model Total Hazard",
    #     linewidth=nng.constants.FIG_LINEWIDTH,
    # )
    (emp_total_line,) = ax.loglog(
        emp_ds_hazard.index.values,
        emp_ds_hazard.values + cs_flt_hazard.values,
        color="green",
        linestyle="solid",
        linewidth=nng.constants.FIG_LINEWIDTH,
    )

    ax.set_ylim(1e-5, 0.1)
    ax.set_xlim(0.01, 1.0)
    ax.grid(which="both", linewidth=0.5, alpha=0.5, linestyle="--")
    ax.set_ylabel("Annual Rate of Exceedance")
    ax.text(
        0.5,
        0.97,
        site,
        transform=ax.transAxes,
        horizontalalignment="center",
        verticalalignment="top",
        bbox=dict(boxstyle="round", fc="w", ec="0.8", alpha=0.5),
    )
    ax.set_xlabel(nng.utils.get_nice_im_name(im))

    # legend = ax.legend(
    #     handles=[
    #         cs_flt_line,
    #         emp_ds_line,
    #         base_ds_line,
    #         loc_ds_line,
    #         emp_total_line,
    #         base_total_line,
    #         loc_total_line,
    #     ],
    #     loc="upper right",
    # )
    # for legline in legend.get_lines():
    # legline.set_linewidth(1.5)

    # Legend
    legend_1 = ax.legend(
        handles=[
            (
                _make_line_proxy(emp_ds_line),
                _make_line_proxy(emp_total_line),
            ),
            (_make_line_proxy(base_ds_line), _make_line_proxy(base_total_line)),
            # (_make_line_proxy(loc_ds_line), _make_line_proxy(loc_total_line)),
        ],
        labels=[
            "Empirical",
            "Surrogate",
            # "Surrogate: Location",
        ],
        loc="upper right",
        handler_map={tuple: HandlerTuple(ndivide=None, pad=0.6)},
        handlelength=4.0,
        # bbox_to_anchor=(1.0, 0.875),
    )
    ax.add_artist(legend_1)

    ax.legend(
        handles=[
            (
                _make_line_proxy(emp_total_line),
                _make_line_proxy(base_total_line),
                # _make_line_proxy(loc_total_line),
            ),
            _make_line_proxy(cs_flt_line),
            (
                _make_line_proxy(emp_ds_line),
                _make_line_proxy(base_ds_line),
                # _make_line_proxy(loc_ds_line),
            ),
        ],
        labels=["Total", "Cybershake: Fault", "Distributed Seismicity"],
        loc="lower left",
        handlelength=4.0,
    )

    fig.tight_layout()
    fig.savefig(
        out_dir
        / f"{site}_{nng.utils.get_im_filename(im)}_hazard_plot.{nng.constants.FIG_FORMAT}"
    )
    plt.close(fig)


def _plot_hazard(
    ax: plt.Axes,
    cur_cs_flt: pd.Series,
    cur_emp_ds: pd.Series,
    cur_base_ds: pd.Series,
    cur_loc_ds: pd.Series,
):
    (cs_flt_line,) = ax.loglog(
        cur_cs_flt.index.values,
        cur_cs_flt.values,
        color="black",
        linestyle="dashdot",
        # linewidth=nng.constants.FIG_LINEWIDTH,
        linewidth=nng.constants.FIG_GROUP_LINEWIDTH,
    )
    (base_ds_line,) = ax.loglog(
        cur_base_ds.index,
        cur_base_ds.values,
        color="blue",
        linestyle="dashed",
        # linewidth=nng.constants.FIG_LINEWIDTH,
        linewidth=nng.constants.FIG_GROUP_LINEWIDTH,
    )
    (emp_ds_line,) = ax.loglog(
        cur_emp_ds.index,
        cur_emp_ds.values,
        color="green",
        linestyle="dashed",
        # linewidth=nng.constants.FIG_LINEWIDTH,
        linewidth=nng.constants.FIG_GROUP_LINEWIDTH,
    )
    (loc_ds_line,) = ax.loglog(
        cur_loc_ds.index,
        cur_loc_ds.values,
        color="red",
        linestyle="dashed",
        # linewidth=nng.constants.FIG_LINEWIDTH,
        linewidth=nng.constants.FIG_GROUP_LINEWIDTH,
    )
    (emp_total_line,) = ax.loglog(
        cur_emp_ds.index,
        (cur_cs_flt + cur_emp_ds).values,
        color="green",
        linestyle="solid",
        linewidth=nng.constants.FIG_LINEWIDTH,
    )
    (base_total_line,) = ax.loglog(
        cur_base_ds.index,
        (cur_cs_flt + cur_base_ds).values,
        color="blue",
        linestyle="solid",
        linewidth=nng.constants.FIG_LINEWIDTH,
    )
    (loc_total_line,) = ax.loglog(
        cur_loc_ds.index,
        (cur_cs_flt + cur_loc_ds).values,
        color="red",
        linestyle="solid",
        linewidth=nng.constants.FIG_LINEWIDTH,
    )

    return (
        cs_flt_line,
        emp_ds_line,
        base_ds_line,
        loc_ds_line,
        emp_total_line,
        base_total_line,
        loc_total_line,
    )


@app.command("site-hazard-4-plot")
def site_hazard_4_plot(
    base_nn_ds_dir: Path, loc_nn_ds_dir: Path, emp_ds_dir: Path, out_dir: Path
):
    """
    Hazard plots for two NN-GMM models and
    empirical DS hazard for two sites and two IMs
    """
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
    cur_base_ds = base_ds_hazard[site1]["total"][im1]
    cur_cs_flt = _interp_cs_hazard(cs_flt_hazard[im1].loc[site1], cur_base_ds)
    cur_loc_ds = loc_ds_hazard[site1]["total"][im1]
    cur_emp_ds = emp_ds_hazard[site1]["total"][im1]

    (
        cur_cs_flt_line,
        cur_emp_ds_line,
        cur_base_ds_line,
        cur_loc_ds_line,
        cur_emp_total_line,
        cur_base_total_line,
        cur_loc_total_line,
    ) = _plot_hazard(ax1, cur_cs_flt, cur_emp_ds, cur_base_ds, cur_loc_ds)

    ax1.set_ylim(1e-4, 2.0)
    ax1.set_xlim(0.01, 3.0)
    ax1.grid(which="both", linewidth=0.5, alpha=0.5, linestyle="--")
    # ax1.tick_params(labelbottom=False)
    ax1.set_ylabel("Annual Rate of Exceedance")
    ax1.set_xlabel(nng.utils.get_nice_im_name(im1))
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

    # Legend
    legend_1 = ax1.legend(
        handles=[
            _make_line_proxy(cur_emp_total_line),
            _make_line_proxy(cur_base_total_line),
            _make_line_proxy(cur_loc_total_line),
        ],
        labels=["Empirical GM LT", "Base Model", "Location Model"],
        loc="upper right",
    )
    ax1.add_artist(legend_1)

    ax1.legend(
        handles=[
            _make_line_proxy(cur_cs_flt_line),
            (
                _make_line_proxy(cur_emp_ds_line),
                _make_line_proxy(cur_base_ds_line),
                _make_line_proxy(cur_loc_ds_line),
            ),
            (
                _make_line_proxy(cur_emp_total_line),
                _make_line_proxy(cur_base_total_line),
                _make_line_proxy(cur_loc_total_line),
            ),
        ],
        labels=["Fault", "Distributed", "Total"],
        loc="upper right",
        handler_map={tuple: HandlerTuple(ndivide=None, pad=0.6)},
        handlelength=6.0,
        bbox_to_anchor=(1.0, 0.775),
    )

    # Site 2 - IM 1
    ax2.sharey(ax1)
    ax2.sharex(ax1)

    cur_base_ds = base_ds_hazard[site2]["total"][im1]
    cur_cs_flt = _interp_cs_hazard(cs_flt_hazard[im1].loc[site2], cur_base_ds)
    cur_loc_ds = loc_ds_hazard[site2]["total"][im1]
    cur_emp_ds = emp_ds_hazard[site2]["total"][im1]

    _plot_hazard(ax2, cur_cs_flt, cur_emp_ds, cur_base_ds, cur_loc_ds)

    ax2.tick_params(labelleft=False)
    ax2.grid(which="both", linewidth=0.5, alpha=0.5, linestyle="--")
    ax2.set_xlabel(nng.utils.get_nice_im_name(im1))

    ax2.text(
        site_txt_offset[0],
        site_txt_offset[1],
        site2,
        transform=ax2.transAxes,
        horizontalalignment="left",
        verticalalignment="top",
        # fontweight="bold",
        bbox=bbox_dict,
    )

    # Site 1 - IM 2
    cur_base_ds = base_ds_hazard[site1]["total"][im2]
    cur_cs_flt = _interp_cs_hazard(cs_flt_hazard[im2].loc[site1], cur_base_ds)
    cur_loc_ds = loc_ds_hazard[site1]["total"][im2]
    cur_emp_ds = emp_ds_hazard[site1]["total"][im2]

    _plot_hazard(ax3, cur_cs_flt, cur_emp_ds, cur_base_ds, cur_loc_ds)

    ax3.set_xlabel(nng.utils.get_nice_im_name(im2))
    ax3.set_ylabel("Annual Rate of Exceedance")
    ax3.grid(which="both", linewidth=0.5, alpha=0.5, linestyle="--")
    ax3.set_ylim(1e-4, 0.08)
    ax3.set_xlim(0.01, 0.5)

    ax3.text(
        site_txt_offset[0],
        site_txt_offset[1],
        site1,
        transform=ax3.transAxes,
        horizontalalignment="left",
        verticalalignment="top",
        # fontweight="bold",
        bbox=bbox_dict,
    )

    # Site 2 - IM 2
    ax4.sharex(ax3)
    ax4.sharey(ax3)

    cur_base_ds = base_ds_hazard[site2]["total"][im2]
    cur_cs_flt = _interp_cs_hazard(cs_flt_hazard[im2].loc[site2], cur_base_ds)
    cur_loc_ds = loc_ds_hazard[site2]["total"][im2]
    cur_emp_ds = emp_ds_hazard[site2]["total"][im2]

    _plot_hazard(ax4, cur_cs_flt, cur_emp_ds, cur_base_ds, cur_loc_ds)

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
        # left=0.06, right=0.99, top=0.99, bottom=0.07, wspace=0.025, hspace=0.025
        left=0.0825,
        right=0.99,
        top=0.99,
        bottom=0.065,
        wspace=0.05,
        hspace=0.15,
    )
    fig.savefig(out_dir / f"site_hazard_4_plot.{nng.constants.FIG_FORMAT}")
    plt.close(fig)


def _plot_poe_lines(ax: plt.Axes, right: bool = False):
    rp_475_prob = nng.utils.rp_to_prob(475)
    ax.axhline(
        rp_475_prob,
        linestyle="dashed",
        color=nng.constants.DARKGRAY,
        linewidth=nng.constants.FIG_MINOR_LINEWIDTH,
    )
    ax.text(
        0.99 if right else 0.01,
        rp_475_prob * 1.05,
        "10% in 50 yrs",
        transform=ax.get_yaxis_transform(),
        va="bottom",
        ha="right" if right else "left",
        color=nng.constants.DARKGRAY,
    )
    rp_2475_prob = nng.utils.rp_to_prob(2475)
    ax.axhline(
        rp_2475_prob,
        linestyle="dashed",
        color=nng.constants.DARKGRAY,
        linewidth=nng.constants.FIG_MINOR_LINEWIDTH,
    )
    ax.text(
        0.99 if right else 0.01,
        rp_2475_prob * 1.05,
        "2% in 50 yrs",
        transform=ax.get_yaxis_transform(),
        va="bottom",
        ha="right" if right else "left",
        color=nng.constants.DARKGRAY,
    )


@app.command("site-hazard-2-plot")
def site_hazard_2_plot(
    base_nn_ds_dir: Path, loc_nn_ds_dir: Path, emp_ds_dir: Path, out_dir: Path
):
    """
    Hazard plots for two NN-GMM models and
    empirical DS hazard for two sites and two IMs
    """
    nng.utils.setup_logging()
    _fig_settings()

    site1, site2 = "HORC", "LHUS"
    im = "pSA_5.0"

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
    fig, (ax1, ax2) = mlt.plotting.get_fig_axes(
        2, 2, 1, ind_figsize=nng.constants.FIG_SIZE, dpi=nng.constants.FIG_DPI
    )

    # Site 1
    cur_base_ds = base_ds_hazard[site1]["total"][im]
    cur_cs_flt = _interp_cs_hazard(cs_flt_hazard[im].loc[site1], cur_base_ds)
    cur_loc_ds = loc_ds_hazard[site1]["total"][im]
    cur_emp_ds = emp_ds_hazard[site1]["total"][im]

    _plot_poe_lines(ax1, True)

    (
        cur_cs_flt_line,
        cur_emp_ds_line,
        cur_base_ds_line,
        cur_loc_ds_line,
        cur_emp_total_line,
        cur_base_total_line,
        cur_loc_total_line,
    ) = _plot_hazard(ax1, cur_cs_flt, cur_emp_ds, cur_base_ds, cur_loc_ds)

    ax1.set_ylim(1e-5, 0.1)
    ax1.set_xlim(0.01, 0.8)
    ax1.grid(which="both", linewidth=0.5, alpha=0.5, linestyle="--")
    # ax1.tick_params(labelbottom=False)
    ax1.set_ylabel("Annual Rate of Exceedance")
    ax1.set_xlabel(nng.utils.get_nice_im_name(im))
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

    # Legend
    legend_1 = ax1.legend(
        handles=[
            _make_line_proxy(cur_emp_total_line),
            _make_line_proxy(cur_base_total_line),
            _make_line_proxy(cur_loc_total_line),
        ],
        labels=["Empirical", "Surrogate: Base", "Surrogate: Location"],
        loc="upper right",
    )
    # ax2.add_artist(legend_1)

    ax2.legend(
        handles=[
            _make_line_proxy(cur_cs_flt_line),
            (
                _make_line_proxy(cur_emp_ds_line),
                _make_line_proxy(cur_base_ds_line),
                _make_line_proxy(cur_loc_ds_line),
            ),
            (
                _make_line_proxy(cur_emp_total_line),
                _make_line_proxy(cur_base_total_line),
                _make_line_proxy(cur_loc_total_line),
            ),
        ],
        labels=["Cybershake: Fault", "Distributed", "Total"],
        loc="upper right",
        handler_map={tuple: HandlerTuple(ndivide=None, pad=0.6)},
        handlelength=6.0,
        # bbox_to_anchor=(1.0, 0.775),
    )

    # Site 2
    ax2.sharey(ax1)
    ax2.sharex(ax1)

    cur_base_ds = base_ds_hazard[site2]["total"][im]
    cur_cs_flt = _interp_cs_hazard(cs_flt_hazard[im].loc[site2], cur_base_ds)
    cur_loc_ds = loc_ds_hazard[site2]["total"][im]
    cur_emp_ds = emp_ds_hazard[site2]["total"][im]

    _plot_poe_lines(ax2, True)

    _plot_hazard(ax2, cur_cs_flt, cur_emp_ds, cur_base_ds, cur_loc_ds)

    ax2.tick_params(labelleft=False)
    ax2.grid(which="both", linewidth=0.5, alpha=0.5, linestyle="--")
    ax2.set_xlabel(nng.utils.get_nice_im_name(im))

    ax2.text(
        site_txt_offset[0],
        site_txt_offset[1],
        site2,
        transform=ax2.transAxes,
        horizontalalignment="left",
        verticalalignment="top",
        # fontweight="bold",
        bbox=bbox_dict,
    )

    fig.subplots_adjust(
        # left=0.06, right=0.99, top=0.99, bottom=0.07, wspace=0.025, hspace=0.025
        left=0.08,
        right=0.99,
        top=0.98,
        # bottom=0.065,
        bottom=0.125,
        wspace=0.075,
        hspace=0.15,
    )
    fig.savefig(out_dir / f"site_hazard_2_plot.{nng.constants.FIG_FORMAT}")
    plt.close(fig)


@app.command("model-trends")
def model_trends(
    base_model_dir: Path,
    loc_model_dir: Path,
    config_ffp: Path,
    output_dir: Path,
    ims: list[str],
    locations: list[str] = None,
    cv: bool = True,
):
    MAG_IM_YAXIS_LIMITS = {
        "pSA_0.5": (1e-3, 2.0),
        "pSA_1.0": (1e-3, 2.0),
        "pSA_5.0": (1e-5, 1.0),
        "pSA_10.0": (1e-5, 1.0),
    }
    RRUP_IM_YAXIS_LIMITS = {
        "pSA_0.5": (1e-3, 5.0),
        "pSA_1.0": (1e-3, 5.0),
        "pSA_5.0": (1e-4, 1.0),
        "pSA_10.0": (1e-4, 1.0),
    }

    nng.utils.setup_logging()
    _fig_settings()

    base_run_config = nng.nn_gmm.load_config(base_model_dir / "run_config.yaml")
    loc_run_config = nng.nn_gmm.load_config(loc_model_dir / "run_config.yaml")
    assert loc_run_config.base_model_dir == base_model_dir

    val_record_ids = (
        pd.read_parquet(base_model_dir / "val_results.parquet").index.values
        if cv
        else np.load(base_model_dir / "train_record_ids.npy")
    )
    with nng.DuckIMDB(base_run_config.imdb_ffp, readonly=True) as imdb:
        site_df = imdb.get_site_df()
        site_df = nng.utils.add_basin_column(site_df)

        sim_df = imdb.get_im_data(base_run_config.ims, record_int_ids=val_record_ids)
        record_info_df = imdb.get_record_info_df(record_int_ids=val_record_ids)
        rel_df = imdb.get_rel_df(rel_int_ids=record_info_df.rel_int_id.unique())
        sim_df[["site_int_id", "rel_int_id", "event_int_id"]] = record_info_df.loc[
            sim_df.index, ["site_int_id", "rel_int_id", "event_int_id"]
        ]
        sim_df["magnitude"] = rel_df.loc[sim_df.rel_int_id.values, "magnitude"].values

        sim_df["site_event_int_id"] = nng.utils.get_site_event_int_id(
            sim_df.site_int_id.values, sim_df.event_int_id.values
        )
        site_event_df = imdb.get_site_event_df(
            site_event_int_ids=sim_df.site_event_int_id.unique()
        )
        sim_df["rrup"] = site_event_df.loc[
            sim_df.site_event_int_id.values, "rrup"
        ].values
        sim_df["basin"] = site_df.loc[sim_df.site_int_id.values, "basin"].values

    del record_info_df, rel_df, site_event_df, site_df

    legend_handles = []
    loc_colors = ["red", "orange", "yellow"]
    config = mlt.utils.load_yaml(config_ffp)
    if config["type"] == "mag":
        fig, axs = nng.plots.magnitude_trend_plot(
            base_model_dir,
            config["fixed_inputs"],
            config["records_range"],
            device,
            record_int_ids=val_record_ids,
            simulation_df=sim_df,
            cv=cv,
            ims=ims,
            plot_ind_cv=False,
            major_line_width=nng.constants.FIG_LINEWIDTH,
            minor_line_width=nng.constants.FIG_GROUP_LINEWIDTH,
            dpi=nng.constants.FIG_DPI,
            ind_fig_size=nng.constants.FIG_SIZE,
            legend=False,
            legend_labels=False,
            min_mag=config.get("min_mag", None),
            max_mag=config.get("max_mag", None),
        )

        legend_handles.append(mlines.Line2D([], [], color="blue", label="Base Model"))

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
                    cv=cv,
                    ims=ims,
                    plot_ind_cv=False,
                    major_line_width=nng.constants.FIG_LINEWIDTH,
                    minor_line_width=nng.constants.FIG_GROUP_LINEWIDTH,
                    plot_empirical=False,
                    nn_color=loc_colors[i],
                    axs=axs,
                    legend=False,
                    legend_labels=False,
                    fill_between=False,
                    min_mag=config.get("min_mag", None),
                    max_mag=config.get("max_mag", None),
                )

                legend_handles.append(
                    mlines.Line2D(
                        [], [], color=loc_colors[i], label=f"Location-Specific ({loc})"
                    )
                )
    elif config["type"] == "rrup":
        fig, axs = nng.plots.rrup_trend_plot(
            base_model_dir,
            config["fixed_inputs"],
            config["records_range"],
            device,
            record_int_ids=val_record_ids,
            simulation_df=sim_df,
            cv=cv,
            ims=ims,
            plot_ind_cv=False,
            major_line_width=nng.constants.FIG_LINEWIDTH,
            minor_line_width=nng.constants.FIG_GROUP_LINEWIDTH,
            dpi=nng.constants.FIG_DPI,
            ind_fig_size=nng.constants.FIG_SIZE,
            legend=False,
            legend_labels=False,
        )

        legend_handles.append(mlines.Line2D([], [], color="blue", label="Base Model"))

        if locations is not None:
            for i, loc in enumerate(locations):
                if loc not in nng.constants.NZ_SITE_LOCATIONS:
                    raise ValueError(f"Unknown NZ location: {loc}")
                nztm_coords = nng.constants.NZ_SITE_LOCATIONS[loc]["nztm"]

                nng.plots.rrup_trend_plot(
                    loc_model_dir,
                    config["fixed_inputs"]
                    | {"nztm_x": nztm_coords[0], "nztm_y": nztm_coords[1]},
                    config["records_range"],
                    device,
                    cv=cv,
                    ims=ims,
                    plot_ind_cv=False,
                    major_line_width=nng.constants.FIG_LINEWIDTH,
                    minor_line_width=nng.constants.FIG_GROUP_LINEWIDTH,
                    plot_empirical=False,
                    nn_color=loc_colors[i],
                    axs=axs,
                    legend=False,
                    legend_labels=False,
                    fill_between=False,
                )

                legend_handles.append(
                    mlines.Line2D(
                        [], [], color=loc_colors[i], label=f"Location-Specific ({loc})"
                    )
                )

    else:
        raise ValueError(f"Unknown trend type: {config['type']}")

    legend_handles.append(
        mlines.Line2D([], [], color="g", label="Empirical - Bradley (2013)")
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
        if config["type"] == "mag":
            ax.set_ylim(MAG_IM_YAXIS_LIMITS[im])
        elif config["type"] == "rrup":
            ax.set_ylim(RRUP_IM_YAXIS_LIMITS[im])

    fig.subplots_adjust(
        left=0.06, right=0.98, top=0.99, bottom=0.07, wspace=0.04, hspace=0.025
    )
    fig.savefig(
        output_dir
        / f"{config_ffp.stem}_{'cv' if cv else 'full'}.{nng.constants.FIG_FORMAT}"
    )


@app.command("sample-weights")
def sample_weights(imdb_ffp: Path, test_events_ffp: Path, output_dir: Path):
    """
    Generate sample weight figures for magnitude,
    rrup, vs30, tectonic type and hypocentre depth.
    """
    nng.utils.setup_logging()
    _fig_settings()

    bin_color = "#4363d8"

    with nng.DuckIMDB(imdb_ffp, readonly=True) as imdb:
        site_df = imdb.get_site_df(max_grid_level=0, min_grid_level=0)
        sites = site_df.site_id.values.astype(str)

        event_df = imdb.get_event_df()
        rel_df = imdb.get_rel_df()

        # Drop test events
        test_events = np.load(test_events_ffp).astype(str)
        event_df = event_df[~event_df.event_id.isin(test_events)]

        record_info_df = imdb.get_record_info_df(
            sites=sites, events=event_df.event_id.values.astype(str)
        )
        site_event_data = imdb.get_site_event_df(
            sites=site_df.site_id.values.astype(str)
        )

    # Add site-event int id
    record_info_df["site_event_int_id"] = nng.utils.get_site_event_int_id(
        record_info_df.site_int_id.values, record_info_df.event_int_id.values
    )

    # Add magnitude, rrup, vs30, and tectonic type
    record_info_df["vs30"] = site_df.loc[record_info_df.site_int_id, "vs30"].values
    record_info_df["magnitude"] = event_df.loc[
        record_info_df.event_int_id, "magnitude"
    ].values
    record_info_df["rrup"] = site_event_data.loc[
        record_info_df.site_event_int_id, "rrup"
    ].values
    record_info_df["tect_type"] = event_df.loc[
        record_info_df.event_int_id, "tect_type"
    ].values
    record_info_df["hypo_depth"] = rel_df.loc[
        record_info_df.rel_int_id, "hypo_depth"
    ].values

    # Spacing
    left, right = 0.07, 0.985
    top, bottom = 0.95, 0.125
    wspace = 0.175

    # Magnitude
    fig, (
        ax1,
        ax2,
    ) = mlt.plotting.get_fig_axes(
        2, 2, 1, ind_figsize=nng.constants.FIG_SIZE, dpi=nng.constants.FIG_DPI
    )
    record_info_df = nng.nn_gmm.get_mag_weights(record_info_df, 2.5)

    ax1.hist(
        record_info_df.magnitude,
        bins=nng.constants.MAG_WEIGHTING_BINS,
        edgecolor="black",
        color=bin_color,
    )
    ax1.grid(linewidth=0.5, alpha=0.5, linestyle="--")
    ax1.xaxis.set_minor_locator(plt.MultipleLocator(0.25))
    ax1.set_ylabel("Count")
    ax1.set_xlabel("Magnitude")
    ax1.set_xlim(
        nng.constants.MAG_WEIGHTING_BINS[0], nng.constants.MAG_WEIGHTING_BINS[-1]
    )

    ax2.hist(
        record_info_df.magnitude.values,
        bins=nng.constants.MAG_WEIGHTING_BINS,
        weights=1 + record_info_df["mag_weight"].values.astype(float),
        edgecolor="black",
        color=bin_color,
    )
    ax2.set_xlabel("Magnitude")
    ax2.grid(linewidth=0.5, alpha=0.5, linestyle="--")
    ax2.xaxis.set_minor_locator(plt.MultipleLocator(0.25))
    ax2.set_ylabel("Weighted Count")
    ax2.set_ylim(ax1.get_ylim())
    ax2.set_xlim(
        nng.constants.MAG_WEIGHTING_BINS[0], nng.constants.MAG_WEIGHTING_BINS[-1]
    )

    fig.subplots_adjust(left=left, right=right, top=top, bottom=bottom, wspace=wspace)
    fig.savefig(output_dir / f"mag_weights.{nng.constants.FIG_FORMAT}")
    plt.close(fig)

    # Source-to-site distance
    fig, (
        ax1,
        ax2,
    ) = mlt.plotting.get_fig_axes(
        2, 2, 1, ind_figsize=nng.constants.FIG_SIZE, dpi=nng.constants.FIG_DPI
    )
    record_info_df = nng.nn_gmm.get_rrup_weights(record_info_df, 2.5)

    ax1.hist(
        record_info_df.rrup,
        bins=nng.constants.RRUP_WEIGHTING_BINS,
        edgecolor="black",
        color=bin_color,
    )
    ax1.grid(linewidth=0.5, alpha=0.5, linestyle="--")
    ax1.xaxis.set_minor_locator(plt.MultipleLocator(25))
    ax1.set_ylabel("Count")
    ax1.set_xlabel("Source-to-site Distance, $R_{rup}$ (km)")
    ax1.set_xlim(
        nng.constants.RRUP_WEIGHTING_BINS[0], nng.constants.RRUP_WEIGHTING_BINS[-1]
    )

    ax2.hist(
        record_info_df.rrup.values,
        bins=nng.constants.RRUP_WEIGHTING_BINS,
        weights=1 + record_info_df["rrup_weight"].values.astype(float),
        edgecolor="black",
        color=bin_color,
    )
    ax2.set_xlabel("Source-to-site Distance, $R_{rup}$ (km)")
    ax2.grid(linewidth=0.5, alpha=0.5, linestyle="--")
    ax2.xaxis.set_minor_locator(plt.MultipleLocator(25))
    ax2.set_ylabel("Weighted Count")
    ax2.set_ylim(ax1.get_ylim())
    ax2.set_xlim(
        nng.constants.RRUP_WEIGHTING_BINS[0], nng.constants.RRUP_WEIGHTING_BINS[-1]
    )

    fig.subplots_adjust(left=left, right=right, top=top, bottom=bottom, wspace=wspace)
    fig.savefig(output_dir / f"rrup_weights.{nng.constants.FIG_FORMAT}")
    plt.close(fig)

    # Vs30
    fig, (
        ax1,
        ax2,
    ) = mlt.plotting.get_fig_axes(
        2, 2, 1, ind_figsize=nng.constants.FIG_SIZE, dpi=nng.constants.FIG_DPI
    )
    record_info_df = nng.nn_gmm.get_vs30_weights(record_info_df, 2.5)

    ax1.hist(
        record_info_df.vs30,
        bins=nng.constants.VS30_WEIGHTING_BINS,
        edgecolor="black",
        color=bin_color,
    )
    ax1.grid(linewidth=0.5, alpha=0.5, linestyle="--")
    ax1.xaxis.set_minor_locator(plt.MultipleLocator(100))
    ax1.xaxis.set_major_locator(plt.MultipleLocator(200))
    ax1.set_ylabel("Count")
    ax1.set_xlabel("$V_{S30}$ (m/s)")
    ax1.set_xlim(
        nng.constants.VS30_WEIGHTING_BINS[0], nng.constants.VS30_WEIGHTING_BINS[-1]
    )

    ax2.hist(
        record_info_df.vs30.values,
        bins=nng.constants.VS30_WEIGHTING_BINS,
        weights=1 + record_info_df["vs30_weight"].values.astype(float),
        edgecolor="black",
        color=bin_color,
    )
    ax2.set_xlabel("$V_{S30}$ (m/s)")
    ax2.grid(linewidth=0.5, alpha=0.5, linestyle="--")
    ax2.xaxis.set_minor_locator(plt.MultipleLocator(100))
    ax2.xaxis.set_major_locator(plt.MultipleLocator(200))
    ax2.set_ylabel("Weighted Count")
    ax2.set_ylim(ax1.get_ylim())
    ax2.set_xlim(
        nng.constants.VS30_WEIGHTING_BINS[0], nng.constants.VS30_WEIGHTING_BINS[-1]
    )

    fig.subplots_adjust(left=left, right=right, top=top, bottom=bottom, wspace=wspace)
    fig.savefig(output_dir / f"vs30_weights.{nng.constants.FIG_FORMAT}")
    plt.close(fig)

    # Tectonic type
    fig, (
        ax1,
        ax2,
    ) = mlt.plotting.get_fig_axes(
        2, 2, 1, ind_figsize=nng.constants.FIG_SIZE, dpi=nng.constants.FIG_DPI
    )

    record_info_df.tect_type = record_info_df.tect_type.map(
        {
            "ACTIVE_SHALLOW": "Crustal",
            "VOLCANIC": "Volcanic",
            "SUBDUCTION_INTERFACE": "Interface",
            "SUBDUCTION_SLAB": "Slab",
        }
    )
    tect_type_counts = record_info_df.tect_type.value_counts()
    record_info_df = nng.nn_gmm.get_tect_type_weights(record_info_df, 2.5)
    record_info_df["base_and_tect_weight"] = 1 + record_info_df["tect_type_weight"]
    weighted_counts = record_info_df.groupby("tect_type", observed=True)[
        "base_and_tect_weight"
    ].sum()

    ax1.bar(
        tect_type_counts.index,
        tect_type_counts.values,
        edgecolor="black",
        color=bin_color,
    )
    ax1.grid(linewidth=0.5, alpha=0.5, linestyle="--")
    ax1.set_ylabel("Count")

    ax2.bar(
        weighted_counts.index,
        weighted_counts.values,
        edgecolor="black",
        color=bin_color,
    )
    ax2.grid(linewidth=0.5, alpha=0.5, linestyle="--")
    ax2.set_ylabel("Weighted Count")
    ax2.set_ylim(ax1.get_ylim())

    fig.subplots_adjust(left=left, right=right, top=top, bottom=bottom, wspace=wspace)
    fig.savefig(output_dir / f"tect_type_weights.{nng.constants.FIG_FORMAT}")
    plt.close(fig)

    # Hypocentre depth
    fig, (
        ax1,
        ax2,
    ) = mlt.plotting.get_fig_axes(
        2, 2, 1, ind_figsize=nng.constants.FIG_SIZE, dpi=nng.constants.FIG_DPI
    )
    record_info_df = nng.nn_gmm.get_depth_weights(record_info_df, 2.5)

    ax1.hist(
        record_info_df.hypo_depth,
        bins=nng.constants.DEPTH_WEIGHTING_BINS,
        edgecolor="black",
        color=bin_color,
    )
    ax1.grid(linewidth=0.5, alpha=0.5, linestyle="--")
    # ax1.xaxis.set_minor_locator(plt.MultipleLocator(25))
    ax1.set_ylabel("Count")
    ax1.set_xlabel(r"Hypocentre Depth, $h_{Depth}$ (km)")
    ax1.set_xlim(
        nng.constants.DEPTH_WEIGHTING_BINS[0], nng.constants.DEPTH_WEIGHTING_BINS[-1]
    )

    ax2.hist(
        record_info_df.hypo_depth.values,
        bins=nng.constants.DEPTH_WEIGHTING_BINS,
        weights=1 + record_info_df["depth_weight"].values.astype(float),
        edgecolor="black",
        color=bin_color,
    )
    ax2.set_xlabel(r"Hypocentre Depth, $h_{Depth}$ (km)")
    ax2.grid(linewidth=0.5, alpha=0.5, linestyle="--")
    # ax2.xaxis.set_minor_locator(plt.MultipleLocator(25))
    ax2.set_ylabel("Weighted Count")
    ax2.set_ylim(ax1.get_ylim())
    ax2.set_xlim(
        nng.constants.DEPTH_WEIGHTING_BINS[0], nng.constants.DEPTH_WEIGHTING_BINS[-1]
    )

    fig.subplots_adjust(left=left, right=right, top=top, bottom=bottom, wspace=wspace)
    fig.savefig(output_dir / f"depth_weights.{nng.constants.FIG_FORMAT}")
    plt.close(fig)


@app.command("site-distribution-maps")
def site_distribution_maps(imdb_ffp: Path, output_dir: Path):
    """Generate site distribution maps for the NZ site locations."""
    nng.utils.setup_logging()
    _fig_settings()

    site_color, style, pen = "green", "t0.075c", "0.05p,black"
    basin_color = "purple"

    with nng.DuckIMDB(imdb_ffp, readonly=True) as imdb:
        site_df = imdb.get_site_df()

    # Level 0 sites
    site_map = nng.plots_spatial.SpatialPlot(
        plot_kwargs={
            "water_color": "white",
        }
    )
    site_map.plot_basins(fill=basin_color)
    site_map.plot_sites(
        site_df.loc[site_df.grid_level == 0], fill=site_color, style=style, pen=pen
    )
    site_map.fig.text(
        position="TL",
        text="Level 0 - 8km",
        offset="0.5c/-0.5c",
        font=nng.constants.GMT_FIG_BOLD_FONT_LABEL,
    )
    site_map.save(
        output_dir / f"level_0_site_distribution_map.{nng.constants.FIG_FORMAT}",
        dpi=nng.constants.FIG_DPI,
    )

    # Level 1 sites
    site_map = nng.plots_spatial.SpatialPlot(plot_kwargs={"water_color": "white"})
    site_map.plot_basins(fill=basin_color)
    site_map.plot_sites(
        site_df.loc[site_df.grid_level == 1], fill=site_color, style=style, pen=pen
    )
    site_map.fig.text(
        position="TL",
        text="Level 1 - 4km",
        offset="0.5c/-0.5c",
        font=nng.constants.GMT_FIG_BOLD_FONT_LABEL,
    )
    site_map.save(
        output_dir / f"level_1_site_distribution_map.{nng.constants.FIG_FORMAT}",
        dpi=nng.constants.FIG_DPI,
    )

    # Level 2 sites
    site_map = nng.plots_spatial.SpatialPlot(plot_kwargs={"water_color": "white"})
    site_map.plot_basins(fill=basin_color)
    site_map.plot_sites(
        site_df.loc[site_df.grid_level == 2], fill=site_color, style=style, pen=pen
    )
    site_map.fig.text(
        position="TL",
        text="Level 2 - 2km",
        offset="0.5c/-0.5c",
        font=nng.constants.GMT_FIG_BOLD_FONT_LABEL,
    )
    site_map.save(
        output_dir / f"level_2_site_distribution_map.{nng.constants.FIG_FORMAT}",
        dpi=nng.constants.FIG_DPI,
    )

    # Level 3 sites
    site_map = nng.plots_spatial.SpatialPlot(plot_kwargs={"water_color": "white"})
    site_map.plot_basins(fill=basin_color)
    site_map.plot_sites(
        site_df.loc[site_df.grid_level == 3], fill=site_color, style=style, pen=pen
    )
    site_map.fig.text(
        position="TL",
        text="Level 3 - 1km",
        offset="0.5c/-0.5c",
        font=nng.constants.GMT_FIG_BOLD_FONT_LABEL,
    )
    site_map.save(
        output_dir / f"level_3_site_distribution_map.{nng.constants.FIG_FORMAT}",
        dpi=nng.constants.FIG_DPI,
    )

    # Real sites
    site_map = nng.plots_spatial.SpatialPlot(plot_kwargs={"water_color": "white"})
    site_map.plot_basins(fill=basin_color)
    site_map.plot_sites(
        site_df.loc[site_df.grid_level == -1],
        fill="green",
        style="t0.1c",
        pen="0.05p,black",
    )
    site_map.fig.text(
        position="TL",
        text="Real Sites",
        offset="0.5c/-0.5c",
        font=nng.constants.GMT_FIG_BOLD_FONT_LABEL,
    )
    site_map.save(
        output_dir / f"real_site_distribution_map.{nng.constants.FIG_FORMAT}",
        dpi=nng.constants.FIG_DPI,
    )


@app.command("site-to-site-residual-hist")
def site_to_site_residual_hist(
    im: str,
    base_model_results_dir: Path,
    loc_adj_results_dir: Path,
    out_dir: Path,
    legend: bool = True,
    y_max_limit: float = None,
    empty_xaxis: bool = False,
    empty_yaxis: bool = False,
):
    """
    Generate histogram of site-to-site residuals
    for the base and location NN-GMM models.
    """
    from mera import MeraResults

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

    # Load site residuals
    base_site_res_df = MeraResults.load_from_parquet(base_mera_results_dir).site_res_df
    assert im in base_site_res_df.columns
    loc_site_res_df = MeraResults.load_from_parquet(
        loc_adj_mera_results_dir
    ).site_res_df
    assert im in loc_site_res_df.columns
    assert (
        base_site_res_df.shape[0] == loc_site_res_df.shape[0]
    ), "Different number of sites!"

    fig, ax = plt.subplots(figsize=nng.constants.FIG_SIZE, dpi=nng.constants.FIG_DPI)

    sns.kdeplot(
        base_site_res_df[im],
        fill=True,
        alpha=0.5,
        label=f"Base\n($\\mu$={np.round(base_site_res_df[im].mean(), 2):.2f}, $\\sigma$={np.round(base_site_res_df[im].std(), 2):.2f})",
        color="blue",
        ax=ax,
    )
    sns.kdeplot(
        loc_site_res_df[im],
        fill=True,
        alpha=0.5,
        label=f"Location\n($\\mu$={np.round(loc_site_res_df[im].mean(), 2):.2f}, $\\sigma$={np.round(loc_site_res_df[im].std(), 2):.2f})",
        color="red",
        ax=ax,
    )

    ax.grid(linewidth=0.5, alpha=0.5, linestyle="--")
    ax.xaxis.set_minor_locator(mticker.AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))
    ax.set_xlabel("Site-to-Site Residual")
    ax.set_ylabel("Density")
    ax.set_xlim(-0.5, 0.5)

    if y_max_limit is not None:
        ax.set_ylim(0, y_max_limit)

    if legend:
        ax.legend(loc="upper right")

    if empty_xaxis:
        ax.set_xlabel("")
        ax.set_xticklabels([])

    if empty_yaxis:
        ax.set_ylabel("")
        ax.set_yticklabels([])

    ax.text(
        0.02,
        0.975,
        nng.utils.get_nice_im_name(im),
        transform=ax.transAxes,
        horizontalalignment="left",
        verticalalignment="top",
        fontweight="bold",
    )

    bottom = 0.025 if empty_xaxis else 0.125
    left = 0.01 if empty_yaxis else 0.1
    fig.subplots_adjust(left=left, right=0.99, top=0.98, bottom=bottom)
    fig.savefig(
        out_dir
        / f"site_to_site_residual_hist_{nng.utils.get_im_filename(im)}.{nng.constants.FIG_FORMAT}",
    )


if __name__ == "__main__":
    app()
