from pathlib import Path

import numpy as np
import torch
import typer

import nn_gmm as nng

torch.multiprocessing.set_start_method("spawn", force=True)

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
if torch.mps.is_available():
    device = "mps"

print(f"Using device: {device.upper()}")

app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command("compute-uniform-grid-ds-hazard")
def compute_uniform_grid_ds_hazard(imdb_ffp: Path,
                                   model_dir: Path,
                                   n_procs: int = 1,
):
    """
    Compute DS hazard on a uniform grid using a NN-GMM model.
    """
    nng.utils.setup_logging()

    nng.hazard.compute_uniform_grid_ds_hazard(
        imdb_ffp, model_dir, device, n_procs=n_procs
    )

@app.command("run-site-ds-hazard")
def run_site_ds_hazard(
    model_dir: Path, sites: list[str] | None = None, n_procs: int = 1
):
    """
    Compute DS hazard for specified sites
    or reference sites if not provided.
    """
    logger = nng.utils.setup_logging()

    if sites is not None:
        logger.info(f"Running DS hazard for specified sites: {sites}")
    else:
        logger.info("Running DS hazard for all reference sites.")
        sites = list(nng.constants.HAZARD_REF_SITES.keys())

    nng.hazard.run_sites_ds_hazard(model_dir, sites, device, n_procs=n_procs)


@app.command("run-emp-site-ds-hazard")
def run_emp_site_ds_hazard(
    imdb_ffp: Path, output_dir: Path, sites: list[str] | None = None, n_procs: int = 1
):
    """
    Compute empirical DS hazard for specified
    sites or reference sites if not provided.
    """
    logger = nng.utils.setup_logging()
    if sites is None:
        logger.info("Running empirical DS hazard for all reference sites.")
        sites = list(nng.constants.HAZARD_REF_SITES.keys())
    else:
        logger.info(f"Running empirical DS hazard for specified sites: {sites}")

    nng.hazard.run_emp_sites_ds_hazard(imdb_ffp, sites, output_dir, n_procs=n_procs)

@app.command("compute-uniform-grid-emp-ds-hazard")
def compute_uniform_grid_emp_ds_hazard(
    imdb_ffp: Path,
    output_dir: Path,
    n_procs: int = 1,
):
    """
    Compute empirical DS hazard on a uniform grid.
    """
    nng.utils.setup_logging()

    nng.hazard.compute_uniform_grid_emp_ds_hazard(
        imdb_ffp, output_dir, n_procs=n_procs
    )


@app.command("run-emp-ds-disagg")
def run_emp_ds_disagg(
    sites_ffp: Path,
    imdb_ffp: Path,
    output_dir: Path,
    rps: list[int],
    max_rrup: float = 500.0,
    n_procs: int = 1,
    tect_type: str | None = None,
    n_sites: int | None = None,
):
    """
    Compute empirical DS disaggregation for specified sites.

    tect_type has to be either None, SUBDUCTION_SLAB or ACTIVE_SHALLOW
    """
    logger = nng.utils.setup_logging()

    sites = np.load(sites_ffp)
    if n_sites is not None:
        sites = sites[:n_sites]
    logger.info(
        f"Running empirical DS disaggregation for {len(sites)} sites from {sites_ffp}."
    )
    nng.hazard.run_emp_ds_disagg(
        sites, imdb_ffp, output_dir, np.asarray(rps), max_rrup=max_rrup, n_procs=n_procs, tect_type=tect_type
    )


if __name__ == "__main__":
    app()
