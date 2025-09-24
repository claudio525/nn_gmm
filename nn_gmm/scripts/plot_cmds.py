from pathlib import Path

import typer
import numpy as np
import matplotlib
import seaborn as sns


import nn_gmm as nng


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
    using NN-GMM results for specified IMs.
    """
    logger = nng.utils.setup_logging()
    nng.plots_spatial.nn_site_bias_res_std(
        nn_dir, ims, output_dir, n_procs=n_procs, grid_spacing=grid_spacing
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
    logger = nng.utils.setup_logging()
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
    imdb_ffp: Path, output_ffp: Path, site_levels: list[int], basin_site_levels: list[int] | None = None
):
    """Generate map showing basin boundaries and site locations."""
    logger = nng.utils.setup_logging()
    nng.plots_spatial.basin_site_map(
        imdb_ffp, output_ffp, site_levels=tuple(site_levels), basin_site_levels=tuple(basin_site_levels)
    )


@app.command("record-event-distribution-map")
def record_event_distribution_map(imdb_ffp: Path, output_ffp: Path):
    """Creates two NZ wide maps showing spatial distribution of records and events"""
    logger = nng.utils.setup_logging()
    nng.plots_spatial.record_event_distribution_map(imdb_ffp, output_ffp)


@app.command("site-folds-map")
def site_folds_map(result_dir: Path, output_ffp: Path):
    """
    Generate map showing site locations and their what site-fold they belong to
    for the given CV results.
    """
    logger = nng.utils.setup_logging()
    run_config = nng.RunConfig.from_yaml(result_dir / "run_config.yaml")

    cv_dirs = [
        cur_dir
        for cur_dir in result_dir.iterdir()
        if cur_dir.is_dir() and cur_dir.name.startswith("cv_")
    ]

    with nng.imdb.IMDB(run_config.imdb_ffp, readonly=True) as imdb:
        site_df = imdb.get_site_df(min_grid_level=0, add_nztm=True).set_index("site_id")

    site_df["site_fold"] = ""
    for cur_dir in cv_dirs:
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





if __name__ == "__main__":
    app()
