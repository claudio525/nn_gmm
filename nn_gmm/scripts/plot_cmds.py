from pathlib import Path
import multiprocessing as mp

import typer
import pandas as pd
import numpy as np


import ml_tools as mlt
import nn_gmm as nng


app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command("nn-site-bias-res-std")
def nn_site_bias_res_std(nn_dir: Path, ims: list[str], output_dir: Path, n_procs: int = 1, grid_spacing: str = "500e/500e"):
    logger = nng.utils.setup_logging()
    nng.plots_spatial.nn_site_bias_res_std(nn_dir, ims, output_dir, n_procs=n_procs, grid_spacing=grid_spacing)

@app.command("basin-site-map")
def basin_site_map(imdb_ffp: Path, output_ffp: Path, site_level: int = None, basin_dir: Path = None):
    logger = nng.utils.setup_logging()
    nng.plots_spatial.basin_site_map(imdb_ffp, output_ffp, site_level=site_level, basin_dir=basin_dir)


if __name__ == "__main__":
    app()