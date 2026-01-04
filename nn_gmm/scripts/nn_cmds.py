import time
import logging
from pathlib import Path

import numpy as np
import torch
import typer

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


@app.command("train-gmm-cv")
def train_cv(
    run_config_ffp: Path,
    n_event_folds: int,
    n_site_folds: int,
    n_epochs: int | None = None,
    seed: int | None = None,
    batch_size: int | None = None,
    id_suffix: str | None = None,
    n_sites: int | None = None,
    n_procs: int = 1,
    run_notebook: bool = True,
    remove_cv_results: bool = False,
):
    """Train and evaluate the GMM using cross-validation."""
    run_config = nng.GMMRunConfig.from_config_kwargs(
        config_ffp=run_config_ffp,
        device=device,
        n_epochs=n_epochs,
        seed=seed,
        batch_size=batch_size,
    )

    id_suffix = f"_{id_suffix}" if id_suffix is not None else ""
    base_out_dir = (
        run_config.results_dir / f"{mlt.utils.create_run_id(False)}{id_suffix}"
    )
    base_out_dir.mkdir(exist_ok=False)

    log_ffp = base_out_dir / "nn_train_cv.log"
    logger = nng.utils.setup_logging(log_ffp, console_level=logging.DEBUG)
    print("Writing logs to:", log_ffp)

    start = time.time()
    nng.train_cv(
        run_config,
        n_event_folds,
        n_site_folds,
        base_out_dir,
        device,
        n_sites=n_sites,
        n_procs=n_procs,
        run_notebook=run_notebook,
        remove_cv_results=remove_cv_results,
    )
    logger.info(
        f"Took: {(time.time() - start) / 60} minutes to complete CV model training."
    )


@app.command("train-loc-adj-cv")
def train_loc_adj_cv(
    run_config_ffp: Path,
    batch_size: int | None = None,
    n_epochs: int | None = None,
    id_suffix: str | None = None,
    n_procs: int = 1,
    run_notebook: bool = True,
    remove_cv_results: bool = False,
    rel_base_model_dir: Path | None = None,
):
    """Train and evaluate the location adjustment model using cross-validation."""
    run_config = nng.LocAdjRunConfig.from_config_kwargs(
        config_ffp=run_config_ffp,
        device=device,
        n_epochs=n_epochs,
        batch_size=batch_size,
        rel_base_model_dir=rel_base_model_dir,
    )

    id_suffix = f"_{id_suffix}" if id_suffix is not None else ""
    base_out_dir = (
        run_config.results_dir / f"{mlt.utils.create_run_id(False)}{id_suffix}"
    )
    base_out_dir.mkdir(exist_ok=False)

    log_ffp = base_out_dir / "nn_train_cv.log"
    logger = nng.utils.setup_logging(log_ffp, console_level=logging.DEBUG)
    print("Writing logs to:", log_ffp)

    start = time.time()
    nng.train_loc_adj_cv(
        run_config,
        base_out_dir,
        n_procs=n_procs,
        run_notebook=run_notebook,
        remove_cv_results=remove_cv_results,
    )
    logger.info(
        f"Took: {(time.time() - start) / 60} minutes to complete CV model training."
    )


@app.command("train-full-gmm")
def train_full_gmm(
    run_config_ffp: Path,
    n_epochs: int | None = None,
    seed: int | None = None,
    batch_size: int | None = None,
    id_suffix: str | None = None,
    n_sites: int | None = None,
):
    """Train the GMM using all available data."""
    run_config = nng.GMMRunConfig.from_config_kwargs(
        config_ffp=run_config_ffp,
        device=device,
        n_epochs=n_epochs,
        seed=seed,
        batch_size=batch_size,
    )

    nng.nn_gmm.run_full_training(
        run_config=run_config,
        n_sites=n_sites,
        id_suffix=id_suffix,
    )


@app.command("train-full-loc-adj-model")
def train_full_loc_adj_model(
    run_config_ffp: Path,
    rel_base_model_dir: Path | None = None,
    n_epochs: int | None = None,
    id_suffix: str | None = None,
    n_sites: int | None = None,
):
    """
    Train location adjustment model
    using all available data.
    """
    run_config = nng.LocAdjRunConfig.from_config_kwargs(
        config_ffp=run_config_ffp,
        device=device,
        n_epochs=n_epochs,
        rel_base_model_dir=rel_base_model_dir,
    )

    nng.nn_gmm.run_full_training(run_config, id_suffix=id_suffix, n_sites=n_sites)


@app.command("run-mera")
def run_mera(
    result_dir: Path, site_term: bool = False, out_dir: Path = None, n_procs: int = 4, ims: list[str] = None
):
    """Run mixed effects residual analysis (MERA) on the CV validation residuals."""
    nng.utils.setup_logging()
    nng.analysis.run_nn_mera(
        result_dir, site_term=site_term, out_dir=out_dir, n_procs=n_procs, ims=ims
    )


@app.command("run-hp-opt")
def run_hp_opt(hp_config_ffp: Path, base_run_config_ffp: Path, n_trials: int, n_procs: int = 1, n_startup_trials: int = 25):
    """Run hyperparameter optimization using Optuna."""
    hp_config = nng.nn_hp_opt.HPOptConfig.from_config(
        hp_config_ffp, base_run_config_ffp, device
    )
    nng.nn_hp_opt.run_hp_opt(hp_config, n_trials, n_procs=n_procs, n_startup_trials=n_startup_trials)


@app.command("continue-hp-opt")
def continue_hp_opt(study_dir: Path, n_trials: int):
    """Continue a previously started hyperparameter optimization study."""
    nng.nn_hp_opt.continue_hp_opt(study_dir, n_trials)


if __name__ == "__main__":
    app()
