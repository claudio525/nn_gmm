import multiprocessing as mp
import logging
from pathlib import Path
from importlib import reload

import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from pygmt_helper import plotting
from qcore import nhm, coordinates
import ml_tools as mlt

from . import nn_gmm
from .imdb import DuckIMDB
from .empdb import DuckEmpiricalDB
from . import constants
from . import utils
from . import loc_pre
from . import analysis


logger = logging.getLogger(__name__)


class SpatialPlot:

    IM_LIMITS_MAPPING = {
        "pSA_0.01": (0.0, 1.0, 0.05),
        "pSA_0.1": (0.0, 2.5, 0.125),
        "pSA_0.5": (0.0, 1.5, 0.075),
        "pSA_1.0": (0.0, 0.8, 0.04),
        "pSA_3.0": (0.0, 0.6, 0.03),
        "pSA_5.0": (0.0, 0.25, 0.0125),
        "pSA_10.0": (0.0, 0.025, 0.00125),
    }

    DEFAULT_PLT_KWARGS = {
        "topo_cmap_min": 0,
        "topo_cmap_max": 3000,
        "topo_cmap_inc": 10,
        "highway_pen_width": 0.1,
    }

    DEFAULT_CONFIG_OPTIONS = dict(
        MAP_FRAME_TYPE="plain",
        FORMAT_GEO_MAP="ddd.xx",
        MAP_GRID_PEN="1p,gray",
        MAP_TICK_PEN_PRIMARY="1p,black",
        MAP_FRAME_PEN="1p,black",
        MAP_FRAME_AXES="wsne",
    )

    def __init__(
        self, plot_kwargs: dict = None, config_options: dict = None, **fig_kwargs
    ):
        plot_kwargs = (
            self.DEFAULT_PLT_KWARGS
            if plot_kwargs is None
            else self.DEFAULT_PLT_KWARGS | plot_kwargs
        )
        config_options = (
            self.DEFAULT_CONFIG_OPTIONS
            if config_options is None
            else self.DEFAULT_CONFIG_OPTIONS | config_options
        )

        self.fig = plotting.gen_region_fig(
            **fig_kwargs,
            config_options=config_options,
            plot_kwargs=plot_kwargs,
        )

    def plot_sites(self, site_df: pd.DataFrame, **plot_kwargs):
        """Adds the specified sites to the existing figure."""
        plot_kwargs = {"style": "p0.035c", "fill": "black"} | plot_kwargs

        self.fig.plot(
            x=site_df["lon"].values,
            y=site_df["lat"].values,
            **plot_kwargs,
        )

        return self

    def plot_ratio(
        self,
        ratio_df: pd.DataFrame,
        data_key: str,
        grid_spacing: str = "500e/500e",
        cmap_limits: tuple[float, float, float] = None,
        **plot_grid_kwargs,
    ):
        """Adds a ratio grid to the existing figure."""
        plot_grid_kwargs = {
            "cb_label": data_key,
            "plot_contours": False,
            "reverse_cmap": True,
            "cmap": "polar",
            "cmap_limits": (
                (-0.5, 0.5, 1.0 / 16) if cmap_limits is None else cmap_limits
            ),
            "cmap_limit_colors": ("darkred", "darkblue"),
        } | plot_grid_kwargs

        grid = plotting.create_grid(ratio_df, data_key, grid_spacing=grid_spacing)

        plotting.plot_grid(self.fig, grid, **plot_grid_kwargs)

        return self

    def plot_im_values(
        self,
        im_df: pd.DataFrame,
        key: str,
        im: str | None = None,
        grid_spacing: str = "500e/500e",
        **plot_grid_kwargs,
    ):
        """
        Adds an IM grid to the existing figure.

        Parameters
        ----------
        im_df : pd.DataFrame
            DataFrame containing 'lon', 'lat', and IM column.
        key : str
            The IM column to plot.
        im : str
            The IM name (e.g., 'pSA_1.0') for setting color scale limits.
            If None, key is used
        """
        im = im or key
        assert key in im_df.columns, f"Invalid key: {key}"
        assert im in self.IM_LIMITS_MAPPING, f"Unsupported IM: {im}"
        assert "lon" in im_df.columns, "im_df must contain 'lon' column"
        assert "lat" in im_df.columns, "im_df must contain 'lat' column"

        plot_grid_kwargs = {
            "cb_label": im,
            "plot_contours": False,
            "reverse_cmap": True,
            "cmap": "hot",
            "cmap_limits": self.IM_LIMITS_MAPPING[im],
            "cmap_limit_colors": ("white", "black"),
        } | plot_grid_kwargs

        im_grid = plotting.create_grid(im_df, key, grid_spacing=grid_spacing)

        plotting.plot_grid(
            self.fig,
            im_grid,
            **plot_grid_kwargs,
        )

        return self

    def plot_basin_boundaries(
        self, basin_dir: Path = constants.BASIN_BOUNDARIES_DIR, **plot_kwargs
    ):
        """Adds basin boundaries to the existing figure."""
        plot_kwargs = {"pen": "0.15p,red"} | plot_kwargs

        basin_files = list(basin_dir.glob("*.txt"))
        for cur_ffp in basin_files:
            cur_basin = np.loadtxt(cur_ffp)
            self.fig.plot(x=cur_basin[:, 0], y=cur_basin[:, 1], **plot_kwargs)

        return self

    def plot_faults(self, faults: tuple[str] | list[str] | None = None, **plot_kwargs):
        """Adds fault traces to the existing figure."""
        plot_kwargs = {"pen": "0.5p,darkgray"} | plot_kwargs

        nhm_data = nhm.load_nhm(str(constants.NHM_FAULT_FFP))

        # Plot the fault traces
        for cur_name, cur_fault in nhm_data.items():
            if faults is None or cur_name in faults:
                cur_trace = cur_fault.trace
                self.fig.plot(
                    x=cur_trace[:, 0],
                    y=cur_trace[:, 1],
                    **plot_kwargs,
                )

        return self

    def plot_hypocentre(self, lon: float, lat: float, **plot_kwargs):
        """Adds a hypocentre to the existing figure."""
        plot_kwargs = {
            "style": "a0.25c",
            "fill": "purple",
            "pen": "0.5p,black",
        } | plot_kwargs

        self.fig.plot(
            x=lon,
            y=lat,
            **plot_kwargs,
        )

        return self

    def save(self, output_ffp: Path, dpi: int = 900):
        self.fig.savefig(output_ffp, dpi=dpi, anti_alias=True)


def basin_site_map(
    imdb_ffp: Path,
    output_ffp: Path,
    site_levels: tuple[int] = (0,),
    basin_site_levels: tuple[int] | None = None,
):
    """Create a NZ wide map showing basin boundaries and site locations."""
    with DuckIMDB(imdb_ffp, readonly=True) as imdb:
        site_df = imdb.get_site_df().set_index("site_id")

    site_df = utils.add_basin_column(site_df)

    mask = site_df.grid_level.isin(site_levels)
    if basin_site_levels is not None:
        mask |= (site_df["basin"] != "NiB") & (
            site_df.grid_level.isin(basin_site_levels)
        )
    site_df = site_df.loc[mask]

    plot = SpatialPlot().plot_basin_boundaries()

    logger.info(f"Number of level 0 sites: {len(site_df.loc[site_df.grid_level == 0])}")
    logger.info(f"Number of level 1 sites: {len(site_df.loc[site_df.grid_level == 1])}")
    logger.info(f"Number of level 2 sites: {len(site_df.loc[site_df.grid_level == 2])}")
    logger.info(f"Number of level 3 sites: {len(site_df.loc[site_df.grid_level == 3])}")

    logger.info(
        f"Number of level 0 basin sites: {len(site_df.loc[site_df.grid_level == 0])}"
    )
    logger.info(
        f"Number of level 1 basin sites: {len(site_df.loc[site_df.grid_level == 1])}"
    )
    logger.info(
        f"Number of level 2 basin sites: {len(site_df.loc[site_df.grid_level == 2])}"
    )
    logger.info(
        f"Number of level 3 basin sites: {len(site_df.loc[site_df.grid_level == 3])}"
    )
    logger.info(f"Total number of sites: {len(site_df)}")

    # Plot site locations
    if site_df.loc[site_df.grid_level == 0].shape[0] > 0:
        plot.plot_sites(
            site_df.loc[site_df.grid_level == 0], style="p0.035c", fill="black"
        )
    if site_df.loc[site_df.grid_level == 1].shape[0] > 0:
        plot.plot_sites(
            site_df.loc[site_df.grid_level == 1], style="d0.035c", fill="green"
        )
    if site_df.loc[site_df.grid_level == 2].shape[0] > 0:
        plot.plot_sites(
            site_df.loc[site_df.grid_level == 2], style="t0.035c", fill="red"
        )
    if site_df.loc[site_df.grid_level == 3].shape[0] > 0:
        plot.plot_sites(
            site_df.loc[site_df.grid_level == 3], style="i0.035c", fill="blue"
        )

    plot.save(output_ffp)


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
    res_df, *_ = analysis.get_nn_sim_residuals(nn_dir)

    run_config = nn_gmm.load_config(nn_dir / "run_config.yaml")
    logging.info(f"Loading IMDB data from {run_config.imdb_ffp}")
    with DuckIMDB(run_config.imdb_ffp, readonly=True) as imdb:
        site_df = imdb.get_site_df()

    logging.info(f"Generating site bias and residual std plots for IMs: {ims}")
    site_bias_res_std(
        res_df,
        site_df,
        ims,
        n_procs=n_procs,
        output_dir=output_dir,
        grid_spacing=grid_spacing,
    )


def emp_gmm_bias_res_std(
    nn_model_dir: Path,
    empdb_ffp: Path,
    ims: list[str],
    output_dir: Path = None,
    n_procs: int = 1,
    grid_spacing: str = "500e/500e",
):
    """
    Generate site bias and site residual standard deviation plots
    using Empirical GMM results for specified IMs.
    """
    run_config = nn_gmm.load_config(nn_model_dir / "run_config.yaml")
    nn_pred_df = pd.read_parquet(nn_model_dir / "val_results.parquet").sort_index()
    record_int_ids = nn_pred_df.index.values.astype(int)

    logging.info(f"Loading IMDB data from {run_config.imdb_ffp}")
    with DuckIMDB(run_config.imdb_ffp, readonly=True) as imdb:
        site_df = imdb.get_site_df()
        record_info_df = imdb.get_record_info_df(record_int_ids=record_int_ids)
        sim_df = imdb.get_im_data(run_config.ims, record_int_ids).sort_index()

    with DuckEmpiricalDB(empdb_ffp, readonly=True) as empdb:
        emp_pred_df = empdb.get_gm_params_tmp_table(
            record_int_ids, incl_std=False
        ).sort_index()

    emp_pred_df["site_int_id"] = record_info_df.loc[
        emp_pred_df.index
    ].site_int_id.values
    assert emp_pred_df.index.equals(sim_df.index)
    logging.info(f"Generating site bias and residual std plots for IMs: {ims}")
    site_bias_res_std(
        emp_pred_df,
        sim_df,
        site_df,
        ims,
        "_mean",
        n_procs=n_procs,
        output_dir=output_dir,
        grid_spacing=grid_spacing,
    )


def site_bias_res_std(
    res_df: pd.DataFrame,
    site_df: pd.DataFrame,
    ims: list[str],
    n_procs: int = 1,
    output_dir: Path = None,
    grid_spacing: str = "500e/500e",
):
    """
    Generate site bias and site residual standard deviation plots
    using the given prediction and simulation data.
    """
    assert (
        "site_int_id" in res_df.columns
    ), "Residual DataFrame must contain 'site_int_id' column"
    assert all([cur_im in res_df.columns for cur_im in ims]), "Unsupported IM"

    # res_df["site_int_id"] = pred_df["site_int_id"].values
    site_bias, site_res_std = analysis.get_site_bias_std(res_df, site_df)

    if n_procs == 1 or len(ims) == 1:
        for im in tqdm(ims):
            _gen_im_bias_res_std_plot(
                site_df,
                res_df.site_int_id,
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
                        res_df.site_int_id,
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


def nn_site_term_map(
    nn_dir: Path,
    ims: list[str],
    output_dir: Path = None,
    n_procs: int = 1,
    grid_spacing: str = "500e/500e",
):
    """
    Generate site term maps using NN-GMM results for specified IMs.
    I.e. map of delta_S2S
    """
    site_terms_ffp = nn_dir / "mera_site_term/site_res_df.parquet"
    if not site_terms_ffp.exists():
        raise FileNotFoundError(
            f"Site terms file not found: {site_terms_ffp}. "
            "Please run MERA analysis with site terms first."
        )

    run_config = nn_gmm.load_config(nn_dir / "run_config.yaml")
    logging.info(f"Loading IMDB data from {run_config.imdb_ffp}")
    with DuckIMDB(run_config.imdb_ffp, readonly=True) as imdb:
        site_df = imdb.get_site_df()

    site_res_df = pd.read_parquet(site_terms_ffp)
    site_res_df = site_res_df.join(site_df[["site_id", "lon", "lat"]].set_index("site_id"), how="left")

    if n_procs == 1:
        for cur_im in ims:
            _gen_im_site_term_map(site_res_df, cur_im, grid_spacing, output_dir)
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=n_procs) as pool:
            pool.starmap(
                _gen_im_site_term_map,
                [
                    (site_res_df, im, grid_spacing, output_dir)
                    for im in ims
                ],
            )

def _gen_im_site_term_map(site_res_df: pd.DataFrame, im: str, grid_spacing: str, output_dir: Path):
    spatial_plot = SpatialPlot()

    spatial_plot.plot_ratio(
        site_res_df,
        im,
        grid_spacing=grid_spacing,
        cmap_limits=(-0.5, 0.5, 1.0 / 16),
        cb_label=f"{utils.get_nice_im_name(im)} Site Term",
    )

    spatial_plot.plot_basin_boundaries().plot_sites(site_res_df, style="p0.015c")
    spatial_plot.save(output_dir / f"nn_site_term_map_{im}.png")

    mlt.utils.write_to_yaml(
        dict(type="nn-site-term-map", im=im, is_mera=True),
        output_dir / f"nn_site_term_map_{im}.yaml",
        clobber=True,
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
        cb_label=im,
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
            region=constants.WELLINGTON_REGION,
            plot_topo=False,
            fig=bias_fig,
            plot_highways=False,
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
            style="p0.075c",
            fill="black",
        )

    bias_metadata = dict(im=im, type="site-bias")

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
        cb_label=im,
        plot_contours=False,
    )

    for cur_ffp in basin_files:
        cur_basin = np.loadtxt(cur_ffp)
        bias_fig.plot(x=cur_basin[:, 0], y=cur_basin[:, 1], pen="0.15p,black")

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
            region=constants.WELLINGTON_REGION,
            plot_topo=False,
            fig=res_std_fig,
            plot_highways=False,
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

        for cur_ffp in basin_files:
            cur_basin = np.loadtxt(cur_ffp)
            res_std_fig.plot(x=cur_basin[:, 0], y=cur_basin[:, 1], pen="0.25p,black")

        res_std_fig.plot(
            x=site_df.loc[pred_site_int_ids, "lon"].values,
            y=site_df.loc[pred_site_int_ids, "lat"].values,
            style="p0.075c",
            fill="black",
        )

    res_std_metadata = dict(im=im, type="site-res-std")

    if output_dir is not None:
        bias_fig.savefig(output_dir / f"{im}_site_bias.png", dpi=900, anti_alias=True)
        res_std_fig.savefig(
            output_dir / f"{im}_site_res_std.png", dpi=900, anti_alias=True
        )
        mlt.utils.write_to_yaml(
            bias_metadata, output_dir / f"{im}_site_bias.yaml", clobber=True
        )
        mlt.utils.write_to_yaml(
            res_std_metadata, output_dir / f"{im}_site_res_std.yaml", clobber=True
        )
    else:
        return bias_fig, res_std_fig


def record_event_distribution_map(
    imdb_ffp: Path, output_dir: Path, grid_size: str = "500e/500e"
):
    """
    Creates two NZ wide maps showing spatial distribution of:
    1. Number of events
    2. Number of records
    """
    with DuckIMDB(imdb_ffp, readonly=True) as imdb:
        site_df = imdb.get_site_df(add_nztm=True, min_grid_level=0, max_grid_level=0)
        record_info_df = imdb.get_record_info_df(
            sites=site_df.site_id.values.astype(str)
        )

    site_groups = record_info_df.groupby("site_int_id")

    event_counts = site_groups["event_id"].nunique().to_frame("n_events")
    event_counts[["lon", "lat"]] = site_df.loc[event_counts.index, ["lon", "lat"]]

    event_grid = plotting.create_grid(event_counts, "n_events", grid_spacing=grid_size)
    event_plot = SpatialPlot(plot_topo=False)
    plotting.plot_grid(
        event_plot.fig,
        event_grid,
        cmap="hot",
        cmap_limits=(0, 130, 10),
        cmap_limit_colors=("white", "black"),
        reverse_cmap=True,
        plot_contours=False,
        cb_label="Number of events",
    )

    event_plot.plot_faults().plot_basin_boundaries(pen="0.25p,black").plot_sites(
        site_df, style="p0.02c"
    )
    event_plot.save(output_dir / "event_map.png")

    record_counts = site_groups.size().to_frame("n_rels")
    record_counts[["lon", "lat"]] = site_df.loc[record_counts.index, ["lon", "lat"]]
    record_grid = plotting.create_grid(record_counts, "n_rels", grid_spacing=grid_size)
    record_plot = SpatialPlot(plot_topo=False)
    plotting.plot_grid(
        record_plot.fig,
        record_grid,
        cmap="hot",
        cmap_limits=(0, 5000, 500),
        cmap_limit_colors=("white", "black"),
        reverse_cmap=True,
        plot_contours=False,
        cb_label="Number of records",
    )

    record_plot.plot_faults().plot_basin_boundaries(pen="0.25p,black").plot_sites(
        site_df, style="p0.02c"
    )
    record_plot.save(output_dir / "record_map.png")


def nn_gmm_full_ratio_map(
    model_dir_1: Path,
    model_dir_2: Path,
    output_dir: Path,
    device: str,
    n_procs: int = 1,
    plot_model_predictions: bool = False,
):
    """
    Generates ratio plots of two full NN-GMM models for
    several different scenarios.

    Assumes model 1 does not use location as an input!!
    """
    default_combs = [
        # Short distance
        [25, 6.5],
        [25, 7.0],
        [25, 7.5],
        [25, 8.0],
        # Moderate distance
        [75, 6.5],
        [75, 7.0],
        [75, 7.5],
        [75, 8.0],
        # Long distance
        [150, 6.5],
        [150, 7.0],
        [150, 7.5],
        [150, 8.0],
    ]

    run_config_1 = nn_gmm.load_config(model_dir_1 / "run_config.yaml")
    run_config_2 = nn_gmm.load_config(model_dir_2 / "run_config.yaml")

    # Load the models
    model_1 = torch.load(
        model_dir_1 / "model.pt", weights_only=False, map_location=device
    )
    model_2 = torch.load(
        model_dir_2 / "model.pt", weights_only=False, map_location=device
    )

    site_locs = loc_pre.get_random_sites(250_000)
    site_locs = site_locs[site_locs[:, 1] < -36.0]
    nztm_coords = coordinates.wgs_depth_to_nztm(site_locs[:, ::-1])

    base_inputs = {
        "tect_type": "crustal",
        "rake": 0,
        "dip": 90,
        "dtop": 0,
        "dbottom": 5,
        "vs30": 760,
        "z1p0": 0.1,
        "z2p5": 1.0,
    }

    for i, (cur_dist, cur_mag) in enumerate(default_combs):
        logger.info(
            f"Running scenario {i + 1}/{len(default_combs)}: {cur_dist} km, M{cur_mag}"
        )
        cur_inputs = base_inputs | {
            "magnitude": cur_mag,
            "rjb": cur_dist,
            "rx": cur_dist,
            "ry": cur_dist,
            "rrup": cur_dist,
        }

        # Get model 1 predictions
        model_1_input_df = pd.DataFrame([cur_inputs])
        model_1_preds = nn_gmm.run_predictions(
            model_1, run_config_1, model_1_input_df, device=device
        )

        loc_input_df = pd.DataFrame(data=site_locs, columns=["lon", "lat"])
        loc_input_df["nztm_y"] = nztm_coords[:, 0]
        loc_input_df["nztm_x"] = nztm_coords[:, 1]

        for cur_key in cur_inputs:
            loc_input_df[cur_key] = cur_inputs[cur_key]
        model_2_preds = nn_gmm.run_predictions(
            model_2, run_config_2, loc_input_df, device=device
        )

        prefix = f"dist_{cur_dist}_mag_{cur_mag}".replace(".", "p")
        if n_procs == 1:
            for im in constants.PLOT_IMS:
                _gen_nn_gmm_full_ratio_map(
                    model_1_preds,
                    model_2_preds,
                    output_dir=output_dir,
                    im=im,
                    prefix=prefix,
                    plot_model_predictions=plot_model_predictions,
                )
        else:
            ctx = mp.get_context("spawn")
            with ctx.Pool(processes=n_procs) as pool:
                pool.starmap(
                    _gen_nn_gmm_full_ratio_map,
                    [
                        (
                            model_1_preds,
                            model_2_preds,
                            output_dir,
                            im,
                            prefix,
                            plot_model_predictions,
                        )
                        for im in constants.PLOT_IMS
                    ],
                )


def _gen_nn_gmm_full_ratio_map(
    model_1_preds: pd.DataFrame,
    model_2_preds: pd.DataFrame,
    output_dir: Path,
    im: str,
    prefix: str,
    plot_model_predictions: bool,
):
    res_df = pd.DataFrame(
        data=model_1_preds[f"{im}_pred"].values - model_2_preds[f"{im}_pred"].values,
        columns=["residual"],
    )
    res_df[["lon", "lat"]] = model_2_preds[["lon", "lat"]]

    spatial_plot = (
        SpatialPlot(plot_topo=False, plot_roads=False, plot_highways=False)
        .plot_ratio(
            res_df,
            "residual",
            cmap_limits=(-1.5, 1.5, 3.0 / 15.0),
            continuous_cmap=True,
            cb_label=f"NoLocModel - LocModel, {utils.get_nice_im_name(im)} Ratio",
        )
        .plot_basin_boundaries(pen="0.15p,black")
    )
    spatial_plot.save(
        output_dir / f"{prefix}_{utils.get_im_filename(im)}_full_ratio.png"
    )
    mlt.utils.write_to_yaml(
        dict(type="full-scenario-model-ratio", im=im, prefix=prefix),
        output_dir / f"{prefix}_{utils.get_im_filename(im)}_full_ratio.yaml",
        clobber=True,
    )

    if plot_model_predictions:
        model_1_plot = (
            SpatialPlot(plot_topo=False, plot_roads=False, plot_highways=False)
            .plot_im_values(
                model_1_preds,
                im,
                grid_spacing="500e/500e",
                cb_label=f"{utils.get_nice_im_name(im)} (Model 1)",
            )
            .plot_basin_boundaries(pen="0.15p,black")
        )
        model_1_plot.save(
            output_dir / f"{prefix}_{utils.get_im_filename(im)}_model_1_pred.png"
        )
        mlt.utils.write_to_yaml(
            dict(type="full-scenario-noLocModel-prediction", im=im, prefix=prefix),
            output_dir / f"{prefix}_{utils.get_im_filename(im)}_model_1_pred.yaml",
            clobber=True,
        )

        model_2_plot = (
            SpatialPlot(plot_topo=False, plot_roads=False, plot_highways=False)
            .plot_im_values(
                model_2_preds,
                im,
                grid_spacing="500e/500e",
                cb_label=f"{utils.get_nice_im_name(im)} (Model 2)",
            )
            .plot_basin_boundaries(pen="0.15p,black")
        )
        model_2_plot.save(
            output_dir / f"{prefix}_{utils.get_im_filename(im)}_model_2_pred.png"
        )
        mlt.utils.write_to_yaml(
            dict(type="full-scenario-locModel-prediction", im=im, prefix=prefix),
            output_dir / f"{prefix}_{utils.get_im_filename(im)}_model_2_pred.yaml",
            clobber=True,
        )


def ds_location_distribution_map(output_dir: Path):
    print("wtf")