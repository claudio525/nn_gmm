import os
import gc
import time

import psutil
from typing import Sequence, List, Optional
from pathlib import Path

import ml_tools.utils
import numpy as np
import pandas as pd
import typer
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt

from mera.mera_pymer4 import run_mera

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

app = typer.Typer()


@app.command("write-predictions")
def write_train_val_predictions(data_dir: Path, model_dir: Path):
    """Write the training & validation predictions to hdf5 databases"""
    nn_gmm.write_train_val_predictions(data_dir, model_dir)


@app.command("metrics")
def train_val_metrics(model_dir: Path, save: bool = False):
    """Computes training & validation metrics"""
    nn_gmm.comp_train_val_metrics(model_dir, save=save)


@app.command("basin-metrics")
def train_val_basin_metrics(
    model_dir: Path, basin_dir: Path, save: bool = True, print: bool = False
):
    """Computes training & validation metrics"""
    nn_gmm.comp_train_val_basin_metrics(
        model_dir, basin_dir, save=save, print_metrics=print
    )


@app.command("spatial-metrics")
def train_val_spatial_metrics(
    model_dir: Path,
    ims: List[str] = None,
    save: bool = True,
    use_sample_weights: bool = False,
):
    """Computes training & validation metrics"""
    nn_gmm.comp_train_val_spatial_metrics(
        model_dir,
        ims=None if len(ims) == 0 else ims,
        save=save,
        use_sample_weights=use_sample_weights,
    )


@app.command("rrup-bin")
def gen_rrup_bin_plot(model_dir: Path, ims: List[str] = None):
    """Creates a Rrup based bin (mag, vs30)
    plot for the specified IM"""
    ims = nn_gmm.GMM.load(model_dir).ims if len(ims) == 0 else ims

    nn_gmm.gen_rrup_bin_plots(model_dir, ims)


@app.command("trend-multi")
def trend_plot_rrup_multi(
    feature: str,
    model_dirs: List[Path],
    output_dir: Path,
    source_config_ffp: Path,
    site_config_ffp: Path,
    site_source_config_ffp: Path,
    ims: List[str] = None,
    model_names: List[str] = None,
):
    """Generates an trend plot for multiple models"""
    nn_gmm.gen_trend_plots(
        feature,
        model_dirs,
        output_dir,
        ims=ims if len(ims) > 0 else None,
        source_config=ml_tools.utils.load_yaml(source_config_ffp),
        site_config=ml_tools.utils.load_yaml(site_config_ffp),
        site_source_config=ml_tools.utils.load_yaml(site_source_config_ffp),
        model_names=None if len(model_names) == 0 else model_names,
    )


@app.command("trend-multi-combo")
def trend_plot_rrup_multi_combo(
    model_dirs: List[Path],
    output_dir: Path,
    source_config_ffps: List[Path] = typer.Option(..., help="The source configs"),
    site_config_ffps: List[Path] = typer.Option(..., help="The site configs"),
    site_source_config_ffps: List[Path] = typer.Option(
        ..., help="The site-source configs"
    ),
    ims: List[str] = None,
    model_names: List[str] = None,
    features: List[str] = None,
):
    """Generates an trend plot for multiple models
    for each of the possible combinations of specified
    site, site-source and source config

    For naming purposes, assumes that the configs have the
    following prefixes: site_, site_source_, source_
    """
    features = ["rrup", "mag", "vs30"] if len(features) == 0 else features

    for cur_source_config_ffp in source_config_ffps:
        cur_source_name = cur_source_config_ffp.stem.replace("source_", "")
        for cur_site_source_config_ffp in site_source_config_ffps:
            cur_site_source_name = cur_site_source_config_ffp.stem.replace(
                "site_source_", ""
            )
            for cur_site_config_ffp in site_config_ffps:
                cur_site_name = cur_site_config_ffp.stem.replace("site_", "")

                cur_out_dir = (
                    output_dir
                    / f"{cur_source_name}_{cur_site_source_name}_{cur_site_name}"
                )
                cur_out_dir.mkdir(exist_ok=True)

                print(f"Memory usage: {psutil.Process().memory_info().rss / 1e9}")
                if "rrup" in features:
                    nn_gmm.gen_trend_plots(
                        "rrup",
                        model_dirs,
                        cur_out_dir,
                        ims=list(ims) if len(ims) > 0 else None,
                        source_config=ml_tools.utils.load_yaml(cur_source_config_ffp),
                        site_config=ml_tools.utils.load_yaml(cur_site_config_ffp),
                        site_source_config=ml_tools.utils.load_yaml(
                            cur_site_source_config_ffp
                        ),
                        model_names=None if len(model_names) == 0 else model_names,
                    )

                if "mag" in features:
                    nn_gmm.gen_trend_plots(
                        "mag",
                        model_dirs,
                        cur_out_dir,
                        ims=list(ims) if len(ims) > 0 else None,
                        source_config=ml_tools.utils.load_yaml(cur_source_config_ffp),
                        site_config=ml_tools.utils.load_yaml(cur_site_config_ffp),
                        site_source_config=ml_tools.utils.load_yaml(
                            cur_site_source_config_ffp
                        ),
                        model_names=None if len(model_names) == 0 else model_names,
                    )

                if "vs30" in features:
                    nn_gmm.gen_trend_plots(
                        "vs30",
                        model_dirs,
                        cur_out_dir,
                        ims=list(ims) if len(ims) > 0 else None,
                        source_config=ml_tools.utils.load_yaml(cur_source_config_ffp),
                        site_config=ml_tools.utils.load_yaml(cur_site_config_ffp),
                        site_source_config=ml_tools.utils.load_yaml(
                            cur_site_source_config_ffp
                        ),
                        model_names=None if len(model_names) == 0 else model_names,
                    )
                tf.keras.backend.clear_session()
                gc.collect()


@app.command("residuals")
def residual_plots(
    model_dir: Path,
    ims: List[str] = None,
    sites: List[str] = None,
    site_names: List[str] = None,
):
    """Generates residual plots"""
    if len(sites) > 0:
        if len(site_names) > 0:
            sites = {
                cur_name: cur_site for cur_site, cur_name in zip(sites, site_names)
            }
        else:
            sites = {cur_site: cur_site for cur_site in sites}
    else:
        sites = None

    nn_gmm.gen_residual_plots(
        model_dir, None if len(ims) == 0 else list(ims), sites=sites,
    )


@app.command("spec-comp-bias-std")
def compare_spec_bias_std(
    output_dir: Path, model_dirs: List[Path], val_only: bool = False
):
    """Generates a spectral bias/std plot for all given models"""
    db_ffps = [cur_model_dir / "val_predictions.hdf5" for cur_model_dir in model_dirs]
    if not val_only:
        db_ffps += [
            cur_model_dir / "train_predictions.hdf5" for cur_model_dir in model_dirs
        ]

    ims = [
        im
        for im in nn_gmm.GMM.load(model_dirs[0]).ims
        if im.startswith("pSA") or im == "PGA"
    ]

    res_plot_gen = nn_gmm.ResPlotGen()
    res_plot_gen.gen_spectral_bias_std_plot(
        db_ffps, ims, output_dir / f"spec_bias_std.pdf"
    )
    # res_plot_gen.gen_binned_spectral_bias_std_plot(db_ffps, ims, output_dir)


@app.command("mixed-effects-analysis")
def comp_mera_data(model_dir: Path, source_rel_params_dir: Path, ims: List[str] = None):
    """
    Run mixed effects regression analysis for
    the given model for both training and validation dataset.

    Note: For training, only a subset of ruptures is used, due to
    data size limitations of the lmer algorithm
    """
    out_dir = Path(model_dir / "mera")
    out_dir.mkdir(exist_ok=True)

    ims = nn_gmm.GMM.load(model_dir).ims if len(ims) == 0 else list(ims)
    im_est = list(np.char.add(ims, "_est"))
    columns = ims + im_est + ["site", "rupture"]

    # Training data
    train_data_df = nn_gmm.ResultDB.get_data_static(
        model_dir / "train_predictions.hdf5", columns + ["fault"],
    )

    # For CS sources, only use 3 realisations (min, median, max magnitude)
    # as lmer can't handle a large number of records
    cs_ruptures = pd.read_csv(
        source_rel_params_dir / "cybershake_v20p4_200.csv", index_col=0
    )
    cs_ruptures["fault"] = np.stack(
        np.char.rsplit(cs_ruptures.index.values.astype(str), "_", maxsplit=1), axis=0
    )[:, 0]

    sel_ruptures = []

    for cur_fault in np.unique(cs_ruptures.fault):
        cur_mask = cs_ruptures.fault == cur_fault

        cur_mags = np.sort(np.unique(cs_ruptures.loc[cur_mask].mag))
        sel_mag = cur_mags[int(np.ceil(cur_mags.size / 2))]

        sel_ruptures.append((cs_ruptures.loc[cur_mask].mag == sel_mag).idxmax())
        sel_ruptures.append((cs_ruptures.loc[cur_mask].mag == cur_mags.min()).idxmax())
        sel_ruptures.append((cs_ruptures.loc[cur_mask].mag == cur_mags.max()).idxmax())

    cs_ruptures_mask = nn_gmm.pandas_isin(train_data_df.rupture, sel_ruptures)
    hist_ruptures_mask = ~nn_gmm.pandas_isin(
        train_data_df.rupture, cs_ruptures.index.values.astype(str)
    )

    train_data_df = train_data_df.loc[cs_ruptures_mask | hist_ruptures_mask]

    train_res_df = pd.DataFrame(
        columns=ims,
        data=train_data_df.loc[:, ims].values - train_data_df.loc[:, im_est].values,
        index=train_data_df.index.values,
    )
    train_res_df[["site", "rupture"]] = train_data_df[["site", "rupture"]]

    (
        train_event_res_df,
        train_site_res_df,
        train_rem_res_df,
        train_bias_std_df,
    ) = run_mera(train_res_df, ims, "rupture", "site")

    # Save
    train_event_res_df.to_csv(out_dir / "train_event_res.csv")
    train_site_res_df.to_csv(out_dir / "train_site_res.csv")
    train_rem_res_df.to_csv(out_dir / "train_rem_res.csv")
    train_bias_std_df.to_csv(out_dir / "train_bias_std.csv")

    # Validation data
    val_data_df = nn_gmm.ResultDB.get_data_static(
        model_dir / "val_predictions.hdf5", columns + ["fault"],
    )

    val_res_df = pd.DataFrame(
        columns=ims,
        data=val_data_df.loc[:, ims].values - val_data_df.loc[:, im_est].values,
        index=val_data_df.index.values,
    )
    val_res_df[["site", "rupture"]] = val_data_df[["site", "rupture"]]

    val_event_res_df, val_site_res_df, val_rem_res_df, val_bias_std_df = run_mera(
        val_res_df, ims, "rupture", "site"
    )

    # Save
    val_event_res_df.to_csv(out_dir / "val_event_res.csv")
    val_site_res_df.to_csv(out_dir / "val_site_res.csv")
    val_rem_res_df.to_csv(out_dir / "val_rem_res.csv")
    val_bias_std_df.to_csv(out_dir / "val_bias_std.csv")


@app.command("between-event-residual-dist")
def event_residual_dist(model_dir: Path, output_dir: Path, ims: List[str] = None):
    ims = nn_gmm.GMM.load(model_dir).ims if len(ims) == 0 else list(ims)

    # Load the data
    cur_train_df = pd.read_csv(model_dir / "mera" / f"train_event_res.csv", index_col=0)
    cur_val_df = pd.read_csv(model_dir / "mera" / f"val_event_res.csv", index_col=0)

    for cur_im in ims:
        fig = plt.figure(figsize=(16, 10))

        ax = fig.add_subplot(1, 2, 1)
        ax.hist(cur_train_df[cur_im], bins=25)
        ax.set_title("Training")
        ax.set_ylabel("Count")
        ax.set_xlabel(r"Between-event residual, $\delta B$")
        ax.set_xlim(-1.0, 1.0)
        ax.grid(which="both", linewidth=0.5, alpha=0.5, linestyle="--")

        ax = fig.add_subplot(1, 2, 2, sharey=ax)
        ax.hist(cur_val_df[cur_im], bins=25)
        ax.set_title("Validation")
        ax.set_ylabel("Count")
        ax.set_xlabel(r"Between-event residual, $\delta B$")
        ax.set_xlim(-1.0, 1.0)
        ax.grid(linewidth=0.5, alpha=0.5, linestyle="--")

        fig.tight_layout()
        fig.suptitle(cur_im)

        fig.savefig(output_dir / f"{cur_im.replace('.', 'p')}_event_residual_dist.png")


@app.command("comp-between-event-std")
def comp_event_std(output_ffp: Path, model_dirs: List[Path]):
    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot()

    # Setup run colours
    run_ids = np.unique([cur_model_dir.stem for cur_model_dir in model_dirs])
    run_colours = [cur_color for cur_color in sns.color_palette("tab10", n_colors=len(run_ids))]

    # Populate plot
    for cur_model_dir, cur_c in zip(model_dirs, run_colours):
        # Load the data
        cur_train_df = pd.read_csv(cur_model_dir / "mera" / f"train_event_res.csv", index_col=0)
        cur_val_df = pd.read_csv(cur_model_dir / "mera" / f"val_event_res.csv", index_col=0)

        cur_cols = np.asarray([cur_col for cur_col in cur_train_df.columns if cur_col.startswith("pSA")])
        cur_periods = np.asarray([float(cur_col.split("_")[-1].replace("p", ".")) for cur_col in cur_cols])

        # Sort by period
        sort_ind = np.argsort(cur_periods)
        cur_cols = cur_cols[sort_ind]
        cur_periods = cur_periods[sort_ind]

        # Compute and plot
        cur_train_std = cur_train_df.std()
        cur_val_df = cur_val_df.std()

        ax.plot(cur_periods, cur_train_std.loc[cur_cols], c=cur_c, linestyle="-", linewidth=0.75, label=cur_model_dir.stem)
        ax.plot(cur_periods, cur_val_df.loc[cur_cols], c=cur_c, linestyle="--", linewidth=0.75)

    ax.semilogx()
    ax.legend()
    ax.set_xlabel(f"Period, T (s)")
    ax.set_ylabel(r"Between-event residual, $\delta B$")
    ax.grid(which="both", linewidth=0.5, alpha=0.5, linestyle="--")
    ax.set_ylim(0.0, 0.4)

    fig.tight_layout()
    fig.savefig(output_ffp)

@app.command("comp-between-event-residual")
def comp_event_residual(output_ffp: Path, model_dirs: List[Path], val: bool = False):
    prefix = "val" if val else "train"

    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot()

    # Setup run colours
    run_ids = np.unique([cur_model_dir.stem for cur_model_dir in model_dirs])
    run_colours = [cur_color for cur_color in sns.color_palette("tab10", n_colors=len(run_ids))]

    # Populate plot
    for cur_model_dir, cur_c in zip(model_dirs, run_colours):
        # Load the data
        cur_df = pd.read_csv(cur_model_dir / "mera" / f"{prefix}_event_res.csv", index_col=0)

        cur_cols = np.asarray([cur_col for cur_col in cur_df.columns if cur_col.startswith("pSA")])
        cur_periods = np.asarray([float(cur_col.split("_")[-1].replace("p", ".")) for cur_col in cur_cols])

        # Sort by period
        sort_ind = np.argsort(cur_periods)
        cur_cols = cur_cols[sort_ind]
        cur_periods = cur_periods[sort_ind]

        # Compute & Plot
        cur_mean_values = cur_df.mean().loc[cur_cols]

        cur_std_values = cur_df.std().loc[cur_cols]
        cur_above = cur_mean_values + cur_std_values
        cur_below = cur_mean_values - cur_std_values

        ax.plot(cur_periods, cur_mean_values, c=cur_c, linestyle="-", label=cur_model_dir.stem, linewidth=0.75)
        ax.plot(cur_periods, cur_above, c=cur_c, linestyle="--", linewidth=0.75)
        ax.plot(cur_periods, cur_below, c=cur_c, linestyle="--" , linewidth=0.75)

    ax.semilogx()
    ax.legend()
    ax.set_xlabel(f"Period")
    ax.set_ylabel(r"Between-event residual, $\delta B$")
    ax.grid(which="both", linewidth=0.5, alpha=0.5, linestyle="--")

    fig.tight_layout()
    fig.savefig(output_ffp)


@app.command("spec-comp-basin-bias-std")
def compare_spec_basin_bias_std(
    output_ffp: Path,
    model_dirs: List[Path],
    use_train: bool = False,
    use_val: bool = False,
    basin_ids: List[str] = None,
):
    """Creates a basin bias/std plot for the given models"""
    # Get the dbs
    assert (use_val or use_train) and not (
        use_val and use_train
    ), "Either use_train or use_val has to be set (but not both!)"

    res_plot_gen = nn_gmm.ResPlotGen()
    res_plot_gen.gen_spectral_basin_bias_std_plot(
        output_ffp,
        model_dirs,
        use_train=use_train,
        use_val=use_val,
        basin_ids=basin_ids,
    )


@app.command("spec-comp-loss")
def compare_spec_loss(output_ffp: Path, model_dirs: List[Path], val_only: bool = False):
    """Generates a spectral loss plot for the specified models"""
    loss_dfs = [
        pd.read_csv(cur_model_dir / "loss.csv", index_col=0)
        for cur_model_dir in model_dirs
    ]
    run_ids = np.asarray([cur_model_dir.stem for cur_model_dir in model_dirs])
    ims = nn_gmm.GMM.load(model_dirs[0]).ims

    # Sort by run_id
    sort_ind = np.argsort(run_ids)
    run_ids = run_ids[sort_ind]
    loss_dfs = [loss_dfs[sort_ix] for sort_ix in sort_ind]

    nn_gmm.gen_spectral_loss_plot(loss_dfs, run_ids, ims, output_ffp, val_only=val_only)


@app.command("comp-basin-metrics-rrup-mag")
def compare_basin_metrics_mag_rrup(
    model_dirs: List[Path] = typer.Argument(
        ..., help="Directories of the models to compare"
    ),
    im: str = typer.Argument(...),
    output_dir: Path = typer.Argument(...),
    basin_dir: Path = typer.Argument(
        ..., help="Directorie that contains the basin definitions"
    ),
    metrics: List[str] = typer.Option(..., help="Metrics to compare"),
    regions: List[str] = typer.Option(..., help="Region/Basins to compare"),
    model_names: List[str] = typer.Option(
        None,
        help="Name of the models to use on the plot\n"
        "Has to be in the same order as model_dirs",
    ),
    val: bool = typer.Option(False, help="Generate for validation data"),
):
    """Creates a figure for each metric-region pair,
    showing the metrics trend (wrt. Magnitude and Rrup)
    for the region"""
    nn_gmm.gen_basin_comp_mag_rrup_plots(
        model_dirs,
        im,
        metrics,
        regions,
        basin_dir,
        output_dir,
        model_names=model_names,
        val=val,
    )


@app.command("comp-site-metrics-rrup-mag")
def compare_site_metrics_mag_rrup(
    model_dirs: List[Path] = typer.Argument(
        ..., help="Directories of the models to compare"
    ),
    im: str = typer.Argument(...),
    output_dir: Path = typer.Argument(...),
    metrics: List[str] = typer.Option(..., help="Metrics to compare"),
    sites: List[str] = typer.Option(..., help="Sites to compare"),
    model_names: List[str] = typer.Option(
        None,
        help="Name of the models to use on the plot\n"
        "Has to be in the same order as model_dirs",
    ),
    val: bool = typer.Option(False, help="Generate for validation data"),
    site_names: List[str] = typer.Option(
        None, help="Name of the sites\n " "Has to be in the same order as sites"
    ),
):
    """Creates a figure for each metric-site pair,
    showing the metrics trend (wrt. Magnitude and Rrup)
    for the sites"""
    nn_gmm.gen_site_comp_mag_rrup_plots(
        model_dirs,
        im,
        metrics,
        sites,
        output_dir,
        model_names=model_names if len(model_names) > 0 else None,
        site_names=site_names if len(site_names) > 0 else None,
        val=val,
    )


@app.command("comp-basin-matrix")
def compare_basin_metrics_matrix(
    model_dirs: List[Path] = typer.Argument(
        ..., help="The directories of the models to compare"
    ),
    output_dir: Path = typer.Argument(
        ..., help="Output directory path for the resulting plots"
    ),
    ims: List[str] = typer.Option(...),
    metrics: List[str] = typer.Option(..., help="The comparison metrics"),
    model_names: List[str] = typer.Option(
        None,
        help="The names of the models to use on the plot\n"
        "If not specified then the run_id (without tags) is used",
    ),
    val: bool = typer.Option(False, help="Generate for validation data"),
):
    """Creates a matrix plot the metric
    for the specified models & available basins
    """
    for cur_metric in metrics:
        for im in ims:
            nn_gmm.gen_basin_metric_comp_matrix(
                [
                    (cur_model_dir, model_names[ix] if len(model_names) > 0 else None)
                    for ix, cur_model_dir in enumerate(model_dirs)
                ],
                im,
                cur_metric,
                output_dir / f"val_{im.replace('.', 'p')}_{cur_metric}.png"
                if val
                else output_dir / f"train_{im.replace('.', 'p')}_{cur_metric}.png",
                val=val,
            )


@app.command("loss-comp")
def loss_compare(model_dirs: List[Path], output_ffp: Path):
    """
    Generates a plot comparing all the specified models
    for the superset of IMs
    """
    nn_gmm.loss_comp(model_dirs, output_ffp)


if __name__ == "__main__":
    app()
