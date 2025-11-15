from pathlib import Path

import torch
import typer
import numpy as np
import pandas as pd
import matplotlib
import seaborn as sns


import nn_gmm as nng
import ml_tools as mlt

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
if torch.mps.is_available():
    device = "mps"
print(f"Using device: {device.upper()}")

app = typer.Typer(pretty_exceptions_show_locals=False)


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
    grid_spacing: str = "100e/100e",
):
    """
    Generate site term maps using NN-GMM CV results for specified IMs.
    I.e. map of delta_S2S
    """
    nng.utils.setup_logging()
    nng.plots_spatial.nn_site_term_map(
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
def gen_site_hazard_plots(ds_results_dir: Path, output_dir: Path, emp_ds_results_dir: Path | None = None):
    """
    Generate site hazard plots for DS hazard results.
    Also adds empirical DS & simulation flt hazard curves.
    """
    nng.utils.setup_logging()
    nng.plots.site_hazard(ds_results_dir, output_dir, emp_ds_results_dir=emp_ds_results_dir)
    
    
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



@app.command("ds-location-distribution-map")
def ds_location_distribution_map(output_dir: Path):
    logger = nng.utils.setup_logging()

    nng.plots_spatial.ds_location_distribution_map(output_dir)

if __name__ == "__main__":
    app()
