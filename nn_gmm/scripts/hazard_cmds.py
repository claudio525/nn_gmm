import pickle
import time
import logging
from pathlib import Path

import pandas as pd
import numpy as np
import torch
import typer
import xarray as xr

import ml_tools as mlt
import nn_gmm as nng

torch.multiprocessing.set_start_method("spawn", force=True)

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
if torch.mps.is_available():
    device = "mps"

print(f"Using device: {device.upper()}")

app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command("run-site-ds-hazard")
def run_site_ds_hazard(model_dir: Path, sites: list[str] | None = None, n_procs: int = 1):
    logger = nng.utils.setup_logging()

    if sites is not None:
        logger.info(f"Running DS hazard for specified sites: {sites}")
    else:
        logger.info("Running DS hazard for all reference sites.")
        sites = list(nng.constants.HAZARD_REF_SITES.keys())

    nng.hazard.run_sites_hazard(model_dir, sites, device, n_procs=n_procs)


@app.command("run-emp-site-ds-hazard")
def run_emp_site_ds_hazard(imdb_ffp: Path, output_dir: Path, sites: list[str] | None = None, n_procs: int = 1):
    logger = nng.utils.setup_logging()
    if sites is None:
        logger.info("Running empirical DS hazard for all reference sites.")
        sites = list(nng.constants.HAZARD_REF_SITES.keys())
    else:
        logger.info(f"Running empirical DS hazard for specified sites: {sites}")

    nng.hazard.run_emp_sites_hazard(imdb_ffp, sites, output_dir, n_procs=n_procs)

if __name__ == "__main__":
    app()