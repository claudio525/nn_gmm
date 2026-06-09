from pathlib import Path
from typing import Annotated

import torch
import typer
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
import seaborn as sns
import xarray as xr


import nn_gmm as nng
import ml_tools as mlt

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
if torch.mps.is_available():
    device = "mps"
print(f"Using device: {device.upper()}")

app = typer.Typer(pretty_exceptions_show_locals=False)

@app.command("plot-loss-curves")
def plot_loss_curves(model_dir: Path, output_dir: Path | None = None):
    """
    Generate training and validation loss curves from NN-GMM training results.
    """
    nng.utils.setup_logging()
    output_dir = output_dir if output_dir is not None else model_dir / "plots"
    output_dir.mkdir(exist_ok=True, parents=False)

    metric = "w_loss_hist"

    # Load metrics 
    da = xr.load_dataarray(model_dir / "metrics.nc")

    val_values = da.sel(metric=f"{metric}_val")
    val_values_mean, val_values_std = None, None
    if "cv_iter" in val_values.coords:
        val_values_mean = val_values.mean(dim="cv_iter").values
        val_values_std = val_values.std(dim="cv_iter").values
    else:
        val_values_mean = val_values

    train_values = da.sel(metric=f"{metric}_train")
    train_values_mean,train_values_std = None, None
    if "cv_iter" in train_values.coords:
        train_values_mean = train_values.mean(dim="cv_iter").values
        train_values_std = train_values.std(dim="cv_iter").values
    else:
        train_values_mean = train_values.values

    epochs = da.coords["epoch"].values + 1

    fig, ax = plt.subplots(figsize=(6,4))

    ax.plot(epochs, train_values_mean, label="Train Loss", color="blue")
    if train_values_std is not None:
        ax.fill_between(
            epochs,
            train_values_mean - train_values_std,
            train_values_mean + train_values_std,
            color="blue",
            alpha=0.3,
        )
    ax.plot(epochs, val_values_mean, label="Validation Loss", color="orange")
    if val_values_std is not None:
        ax.fill_between(
            epochs,
            val_values_mean - val_values_std,
            val_values_mean + val_values_std,
            color="orange",
            alpha=0.3,
        )
    
    if val_values_mean is not None:
        best_epoch = epochs[np.argmin(val_values_mean)]
        ax.axvline(best_epoch, color="red", linestyle="--", label=f"Best Epoch: {best_epoch}")

    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric)
    ax.grid(linewidth=0.5, alpha=0.5, linestyle="--")
    ax.set_xlim(1, epochs[-1])
    ax.legend()
    fig.tight_layout()

    out_ffp = output_dir / f"{metric}_curves.png"
    fig.savefig(out_ffp, dpi=300)
    mlt.utils.write_to_yaml(
            dict(
                type="loss",
                metric=metric,
            ),
            output_dir / out_ffp.with_suffix(".yaml"),
            clobber=True,
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
    test: bool = False,
):
    """
    Generate site term maps using NN-GMM CV results for specified IMs.
    I.e. map of delta_S2S
    """
    nng.utils.setup_logging()
    nng.plots_spatial.nn_site_term_maps(
        nn_dir, ims, output_dir, n_procs=n_procs, grid_spacing=grid_spacing, test=test
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
def bias_res_std_tectonic_type(
    cv_results_dir: Path
):
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
    ds_results_dir: Path, output_dir: Path, emp_ds_results_dir: Path | None = None
):
    """
    Generate site hazard plots for DS hazard results.
    Also adds empirical DS & simulation flt hazard curves.
    """
    nng.utils.setup_logging()
    nng.plots.site_hazard(
        ds_results_dir, output_dir, emp_ds_results_dir=emp_ds_results_dir
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
