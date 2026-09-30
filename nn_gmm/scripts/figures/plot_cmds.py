from pathlib import Path
from typing import Annotated

import matplotlib
import matplotlib.pyplot as plt
import ml_tools as mlt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import typer
from pygmt_helper import plots
from tqdm import tqdm

import nn_gmm as nng

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
if torch.mps.is_available():
    device = "mps"
print(f"Using device: {device.upper()}")

app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command("plot-fault-hazard-curves")
def plot_fault_hazard_curves(
    nn_flt_hazard_results_ffp: Path,
    output_dir: Path,
    cs_parametric_flt_hazard_ffp: Path | None = None,
    sites: list[str] | None = None,
    ims: list[str] = nng.constants.PLOT_IMS,
):
    """
    Generate fault hazard curves for specified sites and IMs.
    """
    nng.utils.setup_logging()
    output_dir.mkdir(exist_ok=True, parents=True)

    # Load CS & NN fault hazard
    cs_flt_hazard = pd.read_pickle(
        nng.constants.HAZARD_RESOURCES_DIR / "flt/Cybershake_hazard_data.pkl"
    )
    if cs_parametric_flt_hazard_ffp is not None:
        cs_parametric_flt_hazard = pd.read_pickle(cs_parametric_flt_hazard_ffp)
    nn_flt_hazard = pd.read_pickle(nn_flt_hazard_results_ffp)

    if sites is None:
        sites = list(nn_flt_hazard[ims[0]].index.values)

    for cur_site in tqdm(sites, desc="Processing sites"):
        for cur_im in ims:
            output_ffp = (
                output_dir
                / f"{cur_site}_{nng.utils.get_im_filename(cur_im)}_fault_hazard_curve.png"
            )
            nng.plots.fault_hazard(
                cs_flt_hazard,
                nn_flt_hazard,
                cur_site,
                cur_im,
                output_ffp,
                cs_parametric_flt_hazard=cs_parametric_flt_hazard,
            )


@app.command("nn-fault-hazard-bias-res-std")
def nn_fault_hazard_bias_res_std(
    base_nn_flt_hazard_results_ffp: Path,
    output_dir: Path,
    rps: list[int] | None = nng.constants.PLOT_RPS,
    loc_nn_flt_hazard_results_ffp: Path | None = None,
    emp_flt_hazard_results_ffp: Path | None = None,
    cs_parametric_flt_hazard_ffp: Path | None = None,
):
    nng.utils.setup_logging()

    nng.plots.nn_fault_hazard_bias_res_std(
        base_nn_flt_hazard_results_ffp,
        rps,
        output_dir=output_dir,
        loc_nn_flt_hazard_results_ffp=loc_nn_flt_hazard_results_ffp,
        emp_flt_hazard_results_ffp=emp_flt_hazard_results_ffp,
        cs_parametric_flt_hazard_ffp=cs_parametric_flt_hazard_ffp,
    )


@app.command("plot-disagg")
def plot_disagg(
    disagg_results_ffp: Path,
    output_dir: Path,
    site: str,
    im: str,
    rp: int,
):
    """
    Generate disaggregation plots for specified disagg results.
    """
    disagg_df = pd.read_parquet(disagg_results_ffp)
    disagg_df["contribution"] = disagg_df["contribution"] * 100

    # Maximum distance
    disagg_df = disagg_df[disagg_df["rrup"] < 200]

    mag_step_size = 0.25
    mag_bins = np.arange(5.0, 8.5 + 0.25, 0.25)
    dist_bins = np.arange(0, 200 + 10, 10)

    disagg_df["mag_bin"] = pd.cut(
        disagg_df["magnitude"],
        bins=mag_bins,
        labels=mag_bins[:-1] + (mag_step_size / 2),
    )
    disagg_df["rrup_bin"] = pd.cut(
        disagg_df["rrup"], bins=dist_bins, labels=dist_bins[:-1] + 5
    )

    disagg_df = (
        disagg_df.groupby(["mag_bin", "rrup_bin", "tect_type"], observed=True)[
            "contribution"
        ]
        .sum()
        .reset_index()
    )
    disagg_df = disagg_df.rename(columns={"mag_bin": "mag", "rrup_bin": "dist"})
    disagg_df["dist_bin_width"] = 10
    disagg_df["mag_bin_width"] = mag_step_size

    fig = plots.disagg_plot(
        disagg_df,
        (0, 200, 5.0, 8.5),
        plots.DisaggPlotType.TectonicType,
        "tect_type",
        category_specs={
            "ACTIVE_SHALLOW": (None, "blue"),
            "SUBDUCTION_INTERFACE": (None, "orange"),
            "SUBDUCTION_SLAB": (None, "red"),
        },
    )

    fig.text(
        position="TL",
        justify="TL",
        text=f"Site: {site}",
        offset="0.2c/-0.2c",
        font=nng.constants.GMT_FIG_BOLD_FONT_LABEL,
    )
    fig.text(
        position="TL",
        justify="TL",
        text=f"IM: {nng.utils.get_nice_im_name(im)}",
        offset="0.2c/-0.8c",
        font=nng.constants.GMT_FIG_BOLD_FONT_LABEL,
    )
    fig.text(
        position="TL",
        justify="TL",
        text=f"PoE: {nng.utils.rp_to_poe_string(rp)}",
        offset="0.2c/-1.4c",
        font=nng.constants.GMT_FIG_BOLD_FONT_LABEL,
    )

    fig.savefig(
        output_dir / f"{disagg_results_ffp.stem}_disagg_plot.png",
        dpi=900,
        anti_alias=True,
    )



@app.command("nn-site-bias-res-std")
def nn_site_bias_res_std(
    nn_dir: Path,
    ims: list[str],
    output_dir: Path,
    n_procs: int = 1,
    grid_spacing: str = "500e/500e",
):
    """
    Generate site bias and site residual standard deviation plots
    using NN-GMM CV results for specified IMs.
    """
    nng.utils.setup_logging()
    nng.plots_spatial.nn_site_bias_res_std(
        nn_dir, ims, output_dir, n_procs=n_procs, grid_spacing=grid_spacing
    )


@app.command("nn-site-term-map")
def nn_site_term_map(
    nn_dir: Path,
    ims: list[str],
    output_dir: Path,
    n_procs: int = 1,
    grid_spacing: str = "500e/500e",
):
    """
    Generate site term maps using NN-GMM CV results for specified IMs.
    I.e. map of delta_S2S
    """
    nng.utils.setup_logging()
    nng.plots_spatial.nn_site_term_maps(
        nn_dir, ims, output_dir, n_procs=n_procs, grid_spacing=grid_spacing
    )


@app.command("nn-rem-residual-map")
def nn_rem_residual_map(
    nn_dir: Path,
    ims: list[str],
    output_dir: Path,
    n_procs: int = 1,
    grid_spacing: str = "500e/500e",
):
    """
    Generate remaining residual term maps
      using NN-GMM CV results for specified IMs.
    """
    nng.utils.setup_logging()
    nng.plots_spatial.nn_rem_residual_maps(
        nn_dir, ims, output_dir, n_procs=n_procs, grid_spacing=grid_spacing
    )


@app.command("nn-gmm-full-ratio-map")
def nn_gmm_full_ratio_map(
    model_dir_1: Path,
    model_dir_2: Path,
    output_dir: Path,
    n_procs: int = 1,
    plot_model_predictions: bool = False,
):
    """
    Generates ratio plots of two full NN-GMM models for
    several different scenarios.

    Assumes model 1 does not use location as an input!!
    """
    nng.utils.setup_logging()
    nng.plots_spatial.nn_gmm_full_ratio_map(
        model_dir_1,
        model_dir_2,
        output_dir,
        device,
        n_procs=n_procs,
        plot_model_predictions=plot_model_predictions,
    )


@app.command("emp-gmm-bias-res-std")
def emp_gmm_bias_res_std(
    nn_model_dir: Path,
    empdb_ffp: Path,
    ims: list[str],
    output_dir: Path,
    n_procs: int = 1,
    grid_spacing: str = "500e/500e",
):
    """
    Generate site bias and site residual standard deviation plots
    using Empirical GMM results for specified IMs.
    """
    nng.utils.setup_logging()
    nng.plots_spatial.emp_gmm_bias_res_std(
        nn_model_dir,
        empdb_ffp,
        ims,
        output_dir,
        n_procs=n_procs,
        grid_spacing=grid_spacing,
    )


@app.command("bias-res-std-tectonic-type")
def bias_res_std_tectonic_type(cv_results_dir: Path):
    """
    Generate site bias and site residual standard deviation plots
    grouped by tectonic type using NN-GMM CV results.
    """
    nng.utils.setup_logging()

    (output_dir := cv_results_dir / "plots").mkdir(exist_ok=True, parents=False)
    nng.plots.bias_res_std_tect_type(
        cv_results_dir,
        output_dir,
    )


@app.command("basin-site-map")
def basin_site_map(
    imdb_ffp: Path,
    output_ffp: Path,
    site_levels: list[int],
    basin_site_levels: list[int] | None = None,
):
    """Generate map showing basin boundaries and site locations."""
    nng.utils.setup_logging()
    nng.plots_spatial.basin_site_map(
        imdb_ffp,
        output_ffp,
        site_levels=tuple(site_levels),
        basin_site_levels=tuple(basin_site_levels),
    )


@app.command("record-event-distribution-map")
def record_event_distribution_map(imdb_ffp: Path, output_dir: Path):
    """Creates two NZ wide maps showing spatial distribution of records and events"""
    nng.utils.setup_logging()
    nng.plots_spatial.record_event_distribution_map(imdb_ffp, output_dir)


@app.command("site-folds-map")
def site_folds_map(result_dir: Path, output_ffp: Path):
    """
    Generate map showing site locations and their site-fold they belong to
    for the given CV results.
    """
    logger = nng.utils.setup_logging()
    run_config = nng.nn_gmm.load_config(result_dir / "run_config.yaml")

    cv_dirs = [
        cur_dir
        for cur_dir in result_dir.iterdir()
        if cur_dir.is_dir() and cur_dir.name.startswith("cv_")
    ]

    with nng.DuckIMDB(run_config.imdb_ffp, readonly=True) as imdb:
        site_df = imdb.get_site_df(min_grid_level=0, add_nztm=True).set_index("site_id")

    site_df["site_fold"] = ""
    for cur_dir in cv_dirs:
        if not (cur_dir / "val_sites.npy").exists():
            logger.warning(f"No val_sites.npy found in {cur_dir}, skipping...")
            continue

        cur_val_sites = np.load(cur_dir / "val_sites.npy")
        site_df.loc[cur_val_sites, "site_fold"] += f"{cur_dir.name},"

    site_df["site_fold"] = site_df["site_fold"].str.rstrip(",")
    site_df = site_df.loc[site_df.site_fold != ""]

    site_groups = site_df.groupby("site_fold")
    n_site_groups = len(site_groups)

    symbols = ["t", "d", "i", "s"]
    site_colors = sns.color_palette("bright", n_site_groups)
    # Convert RGB tuples to hex colors for PyGMT
    site_colors_hex = [matplotlib.colors.to_hex(color) for color in site_colors]

    spatial_plot = nng.plots_spatial.SpatialPlot().plot_basin_boundaries()

    for i, (_, cur_df) in enumerate(site_groups):
        cur_color = site_colors_hex[i]
        cur_symbol = symbols[i % len(symbols)]
        spatial_plot.plot_sites(
            cur_df, style=f"{cur_symbol}0.05c", fill=cur_color, pen="0.05p,black"
        )

    spatial_plot.save(output_ffp)


@app.command("cv-mean-pred-std-map")
def cv_mean_pred_std_map(
    cv_model_results_dir: Path,
    output_dir: Path,
    ims: list[str],
    n_procs: int = 1,
    grid_spacing: str = "250e/250e",
):
    """
    Generate mean predicted standard deviation map
    using NN-GMM CV results.
    """
    nng.utils.setup_logging()
    nng.plots_spatial.cv_mean_pred_std_maps(
        cv_model_results_dir,
        output_dir,
        ims,
        n_procs=n_procs,
        grid_spacing=grid_spacing,
    )


@app.command("hazard-map")
def hazard_maps(
    imdb_ffp: Path,
    hazard_results_dir: Path,
    output_dir: Path,
    ims: Annotated[list[str], typer.Option(default=...)],
    rps: Annotated[list[int], typer.Option(default=...)],
    n_procs: int = 1,
    add_flt_hazard: bool = False,
):
    """
    Generate hazard maps using uniform grid DS hazard results for specified IMs
    at a given return period.
    """
    nng.utils.setup_logging()
    nng.plots_spatial.hazard_maps(
        imdb_ffp,
        hazard_results_dir,
        ims,
        rps,
        output_dir,
        n_procs=n_procs,
        add_flt_hazard=add_flt_hazard,
    )


@app.command("ds-hazard-ratio-map")
def ds_hazard_ratio_map(
    imdb_ffp: Path,
    hazard_results_dir_1: Path,
    hazard_results_dir_2: Path,
    output_dir: Path,
    cb_label_suffix: str,
    filename_prefix: str,
    ims: Annotated[list[str], typer.Option(default=...)],
    rps: Annotated[list[int], typer.Option(default=...)],
    n_procs: int = 1,
):
    nng.utils.setup_logging()
    nng.plots_spatial.ds_hazard_ratio_maps(
        imdb_ffp,
        hazard_results_dir_1,
        hazard_results_dir_2,
        output_dir,
        cb_label_suffix,
        filename_prefix,
        ims,
        rps,
        n_procs=n_procs,
    )


@app.command("nn-gmm-compare-site-bias-histogram")
def nn_gmm_compare_site_bias_histogram(
    model_dir_1: Path,
    model_dir_2: Path,
    output_dir: Path,
    ims: list[str] | None = None,
):
    """
    Generate site bias histograms comparing two NN-GMM models
    for specified IMs.

    Model 1 is plotted at the back.
    """
    nng.utils.setup_logging()
    nng.plots.site_bias_histogram_comparison(
        model_dir_1, model_dir_2, output_dir, ims=ims, dpi=300
    )


@app.command("nn-gmm-compare-site-bias-res-std")
def nn_gmm_compare_site_bias_res_std(
    model_dirs: list[Path],
    output_dir: Path,
):
    """
    Generates a mean site bias and std site bias plot vs pSA period.
    Each model is plotted as a separate line.
    """
    nng.utils.setup_logging()
    nng.plots.site_bias_res_std_comparison(model_dirs, output_dir, dpi=300)


@app.command("nn-mera-bias-res-std")
def nn_mera_bias_res_std(
    model_dir: Path,
):
    """
    Creates bias and residual std plots
    (wrt period) from MERA results.
    """
    nng.utils.setup_logging()

    nng.plots.nn_mera_bias_res_std(model_dir)


@app.command("mera-basin-site-term")
def mera_basin_site_term(
    model_dir_1: Path,
    model_dir_2: Path,
    output_dir: Path,
):
    """
    Generates periods vs basin site term comparison plot
    using MERA results for two models.
    """
    nng.utils.setup_logging()
    nng.plots.mera_basin_site_term_comparison(
        model_dir_1,
        model_dir_2,
        output_dir,
    )


@app.command("gen-site-hazard-plots")
def gen_site_hazard_plots(
    ds_results_dir: Path,
    output_dir: Path,
    emp_ds_results_dir: Path | None = None,
    lower_ds_results_dir: Path | None = None,
    upper_ds_results_dir: Path | None = None,
):
    """
    Generate site hazard plots for DS hazard results.
    Also adds empirical DS & simulation flt hazard curves,
    and the NN-GMM DS epistemic band if lower/upper results are given.
    """
    nng.utils.setup_logging()
    nng.plots.site_hazard(
        ds_results_dir,
        output_dir,
        emp_ds_results_dir=emp_ds_results_dir,
        lower_ds_results_dir=lower_ds_results_dir,
        upper_ds_results_dir=upper_ds_results_dir,
    )


@app.command("gen-site-uhs-plots")
def gen_site_uhs_plots(
    ds_results_dir: Path,
    output_dir: Path,
    rps: list[float],
    emp_ds_results_dir: Path | None = None,
):
    """
    Generate site UHS plots for DS hazard results.
    Also adds empirical DS & simulation flt UHS curves.
    """
    nng.utils.setup_logging()
    nng.plots.site_uhs(
        ds_results_dir, output_dir, rps=rps, emp_ds_results_dir=emp_ds_results_dir
    )


@app.command("pred-vs-res-std")
def pred_vs_res_std(
    model_dir: Path,
):
    """
    Generate predicted vs residual std scatter plot
    using CV results for all IMs.
    """
    nng.utils.setup_logging()
    nng.plots.pred_vs_res_std(model_dir)


@app.command("cv-feature-importance")
def cv_feature_importance(
    results_dir: Path,
):
    """
    Generate feature importance plot using SHAP values computed from CV results.
    """
    nng.utils.setup_logging()

    run_config = nng.nn_gmm.load_config(results_dir / "run_config.yaml")
    shap_values = pd.read_pickle(results_dir / "comb_shap_explanation.pkl")

    (output_dir := results_dir / "plots").mkdir(exist_ok=True, parents=False)
    nng.plots.feature_importance_plots(shap_values, run_config, output_dir)


if __name__ == "__main__":
    app()
