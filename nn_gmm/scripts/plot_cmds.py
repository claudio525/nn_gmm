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


@app.command("emp-gmm-bias-res-std")
def emp_gmm_bias_res_std(nn_model_dir: Path, empdb_ffp: Path, ims: list[str], output_dir: Path, n_procs: int = 1, grid_spacing: str = "500e/500e"):
    logger = nng.utils.setup_logging()
    nng.plots_spatial.emp_gmm_bias_res_std(nn_model_dir, empdb_ffp, ims, output_dir, n_procs=n_procs, grid_spacing=grid_spacing)


@app.command("basin-site-map")
def basin_site_map(imdb_ffp: Path, output_ffp: Path, site_level: int = None, basin_dir: Path = None):
    logger = nng.utils.setup_logging()
    nng.plots_spatial.basin_site_map(imdb_ffp, output_ffp, site_level=site_level, basin_dir=basin_dir)

@app.command("record-event-distribution-map")
def record_event_distribution_map(imdb_ffp: Path, output_ffp: Path):
    logger = nng.utils.setup_logging()
    nng.plots_spatial.record_event_distribution_map(imdb_ffp, output_ffp)
                                  



if __name__ == "__main__":
    app()