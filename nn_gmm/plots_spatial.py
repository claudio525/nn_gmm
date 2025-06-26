import logging
from pathlib import Path

import pandas as pd
import numpy as np

from pygmt_helper import plotting

from . import nn_gmm
from .imdb import IMDB

logger = logging.getLogger(__name__)

def model_site_bias_res_std(model_dir: Path, results_ffp: Path, ims: list[str]):
    run_config = nn_gmm.RunConfig.from_yaml(model_dir / "run_config.yaml")
    pred_df = pd.read_parquet(results_ffp).sort_index()
    record_int_ids = pred_df.index.values.astype(int)

    with IMDB(run_config.imdb_ffp, readonly=True) as imdb:
        site_df = imdb.get_site_df()
        record_info_df = imdb.get_record_info_df(record_int_ids=pred_df.index.values)
        sim_df = imdb.get_im_data_tmp_table(
            run_config.ims, record_int_ids
        ).sort_index()

    pred_df["site_int_id"] = record_info_df.loc[pred_df.index].site_int_id.values
    assert pred_df.index.equals(sim_df.index)

    return site_bias_res_std(pred_df, sim_df, site_df, ims, run_config)

def site_bias_res_std(pred_df: pd.DataFrame, sim_df: pd.DataFrame, site_df: pd.DataFrame, ims: list[str], run_config: nn_gmm.RunConfig):
    assert pred_df.index.equals(sim_df.index), "Prediction DataFrame and Simulation DataFrame must have the same index"
    assert "site_int_id" in pred_df.columns, "Prediction DataFrame must contain 'site_int_id' column"

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

    result_figs = {}
    for im in ims:
        assert im in run_config.ims, f"Unsupported IM: {im}"

        # Bias plot
        grid_bias = plotting.create_grid(site_bias, im)
        bias_fig = plotting.gen_region_fig(
            region="NZ",
            map_data=None,
            # plot_kwargs={
            #     "highway_pen_width": 0.1,
            #     "coastline_pen_width": 0.01,
            #     "topo_cmap": "gray",  # etopo1 geo
            #     "topo_cmap_min": 0,
            #     "topo_cmap_max": 3700,
            #     "topo_cmap_inc": 250,
            #     "topo_cmap_reverse": True,
            # },
            config_options=dict(
                MAP_FRAME_TYPE="plain",
                FORMAT_GEO_MAP="ddd.xx",
                MAP_GRID_PEN="0.5p,gray",
                MAP_TICK_PEN_PRIMARY="1p,black",
                MAP_FRAME_PEN="1p,black",
                MAP_FRAME_AXES="WSne",
            ),
        )
        plotting.plot_grid(bias_fig, grid_bias, "polar", (-0.5, 0.5, 1.0 / 16), ("darkred", "darkblue"), reverse_cmap=True)
        bias_fig.plot(
            x=site_df.loc[pred_df.site_int_id, "lon"],
            y=site_df.loc[pred_df.site_int_id, "lat"],
            style="c0.025c",
            pen="black",
            fill="red",
        )

        # Residual Standard Deviation
        grid_res_std = plotting.create_grid(site_res_std, im)
        res_std_fig = plotting.gen_region_fig(
            region="NZ",
            map_data=None,
            # plot_kwargs={
            #     "highway_pen_width": 0.1,
            #     "coastline_pen_width": 0.01,
            #     "topo_cmap": "gray",  # etopo1 geo
            #     "topo_cmap_min": 0,
            #     "topo_cmap_max": 3700,
            #     "topo_cmap_inc": 250,
            #     "topo_cmap_reverse": True,
            # },
            config_options=dict(
                MAP_FRAME_TYPE="plain",
                FORMAT_GEO_MAP="ddd.xx",
                MAP_GRID_PEN="0.5p,gray",
                MAP_TICK_PEN_PRIMARY="1p,black",
                MAP_FRAME_PEN="1p,black",
                MAP_FRAME_AXES="WSne",
            ),
        )
        plotting.plot_grid(res_std_fig, grid_res_std, "hot", (0, 1.0, 1.0 / 16), ("white", "black"), reverse_cmap=True)
        res_std_fig.plot(
            x=site_df.loc[pred_df.site_int_id, "lon"],
            y=site_df.loc[pred_df.site_int_id, "lat"],
            style="c0.025c",
            pen="black",
            fill="red",
        )

        result_figs[im] = (bias_fig, res_std_fig)
        # out_prefix = f"{out_prefix}_" if out_prefix else ""
        # fig.savefig(output_dir / f"{out_prefix}{im}site_bias.png", dpi=900, anti_alias=True)

    return result_figs