import logging
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


@app.command("compute-fault-nn-hazard")
def compute_fault_nn_hazard(
    cv_model_dir: Path, site_grid_level: int
):
    """
    Compute fault hazard using the NN-GMM model.
    """
    nng.utils.setup_logging(console_level=logging.WARNING)

    output_ffp = (
        Path(cv_model_dir)
        / f"fault_hazard_site_grid_{site_grid_level if site_grid_level >= 0 else "real"}.pkl"
    )
    nng.hazard.compute_fault_nn_hazard(
        cv_model_dir, site_grid_level, output_ffp, device=device
    )


@app.command("compute-cs-parametric-hazard")
def compute_cs_parametric_hazard(imdb_ffp: Path, output_dir: Path, site_grid_level: int):
    """
    Compute Cybershake hazard using mean and sigma estimated
    from the realisations in the IMDB.
    """
    nng.utils.setup_logging(console_level=logging.WARNING)

    output_ffp = (
        Path(output_dir)
        / f"cs_parametric_hazard_site_grid_{site_grid_level if site_grid_level >= 0 else 'real'}.pkl"
    )
    nng.hazard.compute_cs_parametric_hazard(
        imdb_ffp, output_ffp, site_grid_level
    )

@app.command("compute-fault-emp-hazard")
def compute_fault_emp_hazard(
    imdb_ffp: Path, output_dir: Path, site_grid_level: int
):
    """
    Compute fault hazard using empirical distributions from the IMDB.
    """
    nng.utils.setup_logging()

    output_ffp = (
        output_dir
        / f"fault_emp_hazard_site_grid_{site_grid_level if site_grid_level >= 0 else 'real'}.pkl"
    )
    nng.hazard.compute_fault_emp_hazard(
        imdb_ffp, site_grid_level, output_ffp
    )

@app.command("compute-uniform-grid-ds-hazard")
def compute_uniform_grid_ds_hazard(
    imdb_ffp: Path,
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
    model_dir: Path,
    sites: list[str] | None = None,
    n_procs: int = 1,
    sigma_ept_ffp: Path | None = None,
    quantile: float | None = None,
):
    """
    Compute DS hazard for specified sites
    or reference sites if not provided.

    If sigma_ept_ffp (from compute-sigma-ept) and quantile are given,
    computes the hazard at that quantile of the surrogate epistemic uncertainty.
    """
    logger = nng.utils.setup_logging()

    if (sigma_ept_ffp is None) != (quantile is None):
        raise ValueError("sigma_ept_ffp and quantile have to be specified together.")
    if quantile is not None and not 0 < quantile < 1:
        raise ValueError(f"Quantile has to be in (0, 1), got {quantile}.")

    if sites is not None:
        logger.info(f"Running DS hazard for specified sites: {sites}")
    else:
        logger.info("Running DS hazard for all reference sites.")
        sites = list(nng.constants.HAZARD_REF_SITES.keys())

    nng.hazard.run_nn_sites_ds_hazard(
        model_dir,
        sites,
        device,
        n_procs=n_procs,
        sigma_ept_ffp=sigma_ept_ffp,
        quantile=quantile,
    )


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

    nng.hazard.compute_uniform_grid_emp_ds_hazard(imdb_ffp, output_dir, n_procs=n_procs)


@app.command("run-site-disagg")
def run_site_disagg(
    full_model_dir: Path, sites: list[str], im: str, rp: float, 
):
    """
    Compute disaggregation for specified sites using the
    NN-GMM model for DS hazard and simulations for fault sources
    """
    nng.utils.setup_logging()

    nng.hazard.compute_site_disagg(full_model_dir, sites, im, rp, device)
    

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

    Note: This was used to assist in selecting the DS ruptures for 
    simulation!
    """
    logger = nng.utils.setup_logging()

    sites = np.load(sites_ffp)
    if n_sites is not None:
        sites = sites[:n_sites]
    logger.info(
        f"Running empirical DS disaggregation for {len(sites)} sites from {sites_ffp}."
    )
    nng.hazard.run_emp_ds_disagg(
        sites,
        imdb_ffp,
        output_dir,
        np.asarray(rps),
        max_rrup=max_rrup,
        n_procs=n_procs,
        tect_type=tect_type,
    )


if __name__ == "__main__":
    app()
