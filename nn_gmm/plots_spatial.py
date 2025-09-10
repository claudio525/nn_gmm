import os
import multiprocessing as mp
import logging
from pathlib import Path
from importlib import reload

import pygmt
import pandas as pd
import numpy as np
from tqdm import tqdm

from pygmt_helper import plotting

from . import nn_gmm
from .imdb import IMDB
from . import constants

logger = logging.getLogger(__name__)


def basin_site_map(
    imdb_ffp: Path, output_ffp: Path, site_level: int = None, basin_dir: Path = None
):
    """Create a NZ wide map showing basin boundaries and site locations."""
    fig = plotting.gen_region_fig(
        plot_topo=True,
        plot_highways=True,
        config_options=dict(
            MAP_FRAME_TYPE="plain",
            FORMAT_GEO_MAP="ddd.xx",
            MAP_GRID_PEN="0.5p,gray",
            MAP_TICK_PEN_PRIMARY="1p,black",
            MAP_FRAME_PEN="1p,black",
            MAP_FRAME_AXES="wsne",
        ),
        plot_kwargs={
            "topo_cmap_min": 0,
            "topo_cmap_max": 3000,
            "topo_cmap_inc": 10,
            "highway_pen_width": 0.1,
        },
    )

    # Plot basin boundaries
    if basin_dir is not None:
        basin_files = list(basin_dir.glob("*.txt"))
        for cur_ffp in basin_files:
            cur_basin = np.loadtxt(cur_ffp)
            fig.plot(x=cur_basin[:, 0], y=cur_basin[:, 1], pen="0.15p,red")

    # Plot site locations
    if site_level is not None:
        with IMDB(imdb_ffp, readonly=True) as imdb:
            site_df = imdb.get_site_df(
                min_grid_level=site_level, max_grid_level=site_level
            )
            fig.plot(
                x=site_df["lon"].values,
                y=site_df["lat"].values,
                style="t0.03c",
                fill="blue",
            )

    fig.savefig(output_ffp, dpi=900, anti_alias=True)


def nn_site_bias_res_std(
    nn_dir: Path,
    ims: list[str],
    output_dir: Path = None,
    n_procs: int = 1,
    grid_spacing: str = "500e/500e",
):
    """
    Generate site bias and site residual standard deviation plots
    using NN-GMM results for specified IMs.
    """
    run_config = nn_gmm.RunConfig.from_yaml(nn_dir / "run_config.yaml")
    pred_df = pd.read_parquet(nn_dir / "val_results.parquet").sort_index()
    record_int_ids = pred_df.index.values.astype(int)

    logging.info(f"Loading IMDB data from {run_config.imdb_ffp}")
    with IMDB(run_config.imdb_ffp, readonly=True) as imdb:
        site_df = imdb.get_site_df()
        record_info_df = imdb.get_record_info_df(record_int_ids=pred_df.index.values)
        sim_df = imdb.get_im_data_tmp_table(run_config.ims, record_int_ids).sort_index()

    pred_df["site_int_id"] = record_info_df.loc[pred_df.index].site_int_id.values
    assert pred_df.index.equals(sim_df.index)

    logging.info(f"Generating site bias and residual std plots for IMs: {ims}")
    site_bias_res_std(
        pred_df,
        sim_df,
        site_df,
        ims,
        run_config,
        n_procs=n_procs,
        output_dir=output_dir,
        grid_spacing=grid_spacing,
    )


def site_bias_res_std(
    pred_df: pd.DataFrame,
    sim_df: pd.DataFrame,
    site_df: pd.DataFrame,
    ims: list[str],
    run_config: nn_gmm.RunConfig,
    n_procs: int = 1,
    output_dir: Path = None,
    grid_spacing: str = "500e/500e",
):
    """
    Generate site bias and site residual standard deviation plots
    using the given prediction and simulation data.
    """
    assert pred_df.index.equals(
        sim_df.index
    ), "Prediction DataFrame and Simulation DataFrame must have the same index"
    assert (
        "site_int_id" in pred_df.columns
    ), "Prediction DataFrame must contain 'site_int_id' column"

    # Compute residuals
    res_df = pd.DataFrame(
        data=np.log(sim_df[run_config.ims].values)
        - pred_df[run_config.pred_mean_keys].values,
        index=pred_df.index,
        columns=run_config.ims,
    )

    # Site bias
    res_df["site_int_id"] = pred_df["site_int_id"].values
    site_bias = res_df.groupby("site_int_id").mean()
    site_bias["lon"] = site_df.loc[site_bias.index, "lon"].values
    site_bias["lat"] = site_df.loc[site_bias.index, "lat"].values

    # Site residual standard deviation
    site_res_std = res_df.groupby("site_int_id").std()
    site_res_std["lon"] = site_df.loc[site_res_std.index, "lon"].values
    site_res_std["lat"] = site_df.loc[site_res_std.index, "lat"].values

    assert all([cur_im in run_config.ims for cur_im in ims]), "Unsupported IM"

    if n_procs == 1 or len(ims) == 1:
        for im in tqdm(ims):
            _gen_im_bias_res_std_plot(
                site_df,
                pred_df.site_int_id,
                site_bias,
                site_res_std,
                im,
                output_dir=output_dir,
                is_mp=False,
                grid_spacing=grid_spacing,
            )
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=n_procs) as pool:
            pool.starmap(
                _gen_im_bias_res_std_plot,
                [
                    (
                        site_df,
                        pred_df.site_int_id,
                        site_bias,
                        site_res_std,
                        im,
                        output_dir,
                        True,
                        grid_spacing,
                    )
                    for im in ims
                ],
            )


def _gen_im_bias_res_std_plot(
    site_df: pd.DataFrame,
    pred_site_int_ids: np.ndarray,
    site_bias: pd.DataFrame,
    site_res_std: pd.DataFrame,
    im: str,
    output_dir: Path = None,
    is_mp: bool = False,
    grid_spacing: str = "500e/500e",
):
    if is_mp:
        import pygmt

        reload(pygmt)

    basin_files = list(constants.BASIN_BOUNDARIES_DIR.glob("*.txt"))

    # Bias plot
    grid_bias = plotting.create_grid(site_bias, im, grid_spacing=grid_spacing)
    bias_fig = plotting.gen_region_fig(
        plot_highways=False,
        plot_topo=False,
        config_options=dict(
            MAP_FRAME_TYPE="plain",
            FORMAT_GEO_MAP="ddd.xx",
            MAP_GRID_PEN="0.5p,gray",
            MAP_TICK_PEN_PRIMARY="1p,black",
            MAP_FRAME_PEN="1p,black",
            MAP_FRAME_AXES="wsne",
        ),
    )

    plotting.plot_grid(
        bias_fig,
        grid_bias,
        "polar",
        (-0.5, 0.5, 1.0 / 16),
        ("darkred", "darkblue"),
        reverse_cmap=True,
        plot_contours=False,
    )

    for cur_ffp in basin_files:
        cur_basin = np.loadtxt(cur_ffp)
        bias_fig.plot(x=cur_basin[:, 0], y=cur_basin[:, 1], pen="0.15p,black")

    bias_fig.plot(
        x=site_df.loc[pred_site_int_ids, "lon"].values,
        y=site_df.loc[pred_site_int_ids, "lat"].values,
        style="p0.015c",
        fill="black",
    )

    # Create the inset
    with bias_fig.inset(
        position="jTL",  # +o0.2c",
        region=constants.WELLINGTON_REGION,
        projection="M8.5c",
        # projection="M4c",
        margin=0,
        box="+p1p,black",
    ):
        bias_fig = plotting.gen_region_fig(
            region=constants.WELLINGTON_REGION, plot_topo=False, fig=bias_fig,
            plot_highways=False
        )

        plotting.plot_grid(
            bias_fig,
            grid_bias,
            "polar",
            (-0.5, 0.5, 1.0 / 16),
            ("darkred", "darkblue"),
            reverse_cmap=True,
            plot_contours=False,
        )

        for cur_ffp in basin_files:
            cur_basin = np.loadtxt(cur_ffp)
            bias_fig.plot(x=cur_basin[:, 0], y=cur_basin[:, 1], pen="0.25p,black")

        bias_fig.plot(
            x=site_df.loc[pred_site_int_ids, "lon"].values,
            y=site_df.loc[pred_site_int_ids, "lat"].values,
            style="p0.015c",
            fill="black",
        )

    # Residual Standard Deviation
    grid_res_std = plotting.create_grid(site_res_std, im, grid_spacing=grid_spacing)
    res_std_fig = plotting.gen_region_fig(
        plot_topo=False,
        plot_highways=False,
        config_options=dict(
            MAP_FRAME_TYPE="plain",
            FORMAT_GEO_MAP="ddd.xx",
            MAP_GRID_PEN="0.5p,gray",
            MAP_TICK_PEN_PRIMARY="1p,black",
            MAP_FRAME_PEN="1p,black",
            MAP_FRAME_AXES="wsne",
        ),
    )
    plotting.plot_grid(
        res_std_fig,
        grid_res_std,
        "hot",
        (0, 1.0, 1.0 / 16),
        ("white", "black"),
        reverse_cmap=True,
        plot_contours=False,
    )
    res_std_fig.plot(
        x=site_df.loc[pred_site_int_ids, "lon"].values,
        y=site_df.loc[pred_site_int_ids, "lat"].values,
        style="p0.015c",
        fill="black",
    )

        # Create the inset
    with res_std_fig.inset(
        position="jTL",  # +o0.2c",
        region=constants.WELLINGTON_REGION,
        projection="M8.5c",
        # projection="M4c",
        margin=0,
        box="+p1p,black",
    ):
        res_std_fig = plotting.gen_region_fig(
            region=constants.WELLINGTON_REGION, plot_topo=False, fig=res_std_fig,
            plot_highways=False
        )

        plotting.plot_grid(
            res_std_fig,
            grid_res_std,
            "polar",
            (-0.5, 0.5, 1.0 / 16),
            ("darkred", "darkblue"),
            reverse_cmap=True,
            plot_contours=False,
        )

        for cur_ffp in basin_files:
            cur_basin = np.loadtxt(cur_ffp)
            res_std_fig.plot(x=cur_basin[:, 0], y=cur_basin[:, 1], pen="0.25p,black")

        res_std_fig.plot(
            x=site_df.loc[pred_site_int_ids, "lon"].values,
            y=site_df.loc[pred_site_int_ids, "lat"].values,
            style="p0.015c",
            fill="black",
        )


    if output_dir is not None:
        bias_fig.savefig(output_dir / f"{im}_site_bias.png", dpi=900, anti_alias=True)
        res_std_fig.savefig(
            output_dir / f"{im}_site_res_std.png", dpi=900, anti_alias=True
        )
    else:
        return bias_fig, res_std_fig
