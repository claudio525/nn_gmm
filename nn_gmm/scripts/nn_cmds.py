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
    run_config = nng.RunConfig.from_config_kwargs(
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
    assert not base_out_dir.exists(), "Output directory already exists!"
    base_out_dir.mkdir()

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
    run_config = nng.RunConfig.from_config_kwargs(
        config_ffp=run_config_ffp,
        device=device,
        n_epochs=n_epochs,
        seed=seed,
        batch_size=batch_size,
    )

    id_suffix = f"_{id_suffix}" if id_suffix is not None else ""
    (
        out_dir := run_config.results_dir
        / f"{mlt.utils.create_run_id(False)}{id_suffix}"
    ).mkdir(parents=False, exist_ok=False)

    log_ffp = out_dir / "nn_train_cv.log"
    logger = nng.utils.setup_logging(log_ffp, console_level=logging.DEBUG)
    print("Writing logs to:", log_ffp)

    # Get event and site data
    with nng.DuckIMDB(run_config.imdb_ffp, readonly=True) as imdb:
        event_df = imdb.get_event_df()
        site_df = imdb.get_site_df(min_grid_level=0, add_nztm=True)

    if run_config.extra_basin_sites:
        # Take all level 0 sites and level 2 & 3 sites that are in a basin
        site_df = nng.utils.add_basin_column(site_df)
        site_df = site_df.loc[
            (site_df.grid_level == 0)
            | ((site_df["basin"] != "NiB") & (site_df.grid_level == 2))
        ]

    events, sites = event_df.event_id.values.astype(str), site_df.site_id.values.astype(
        str
    )

    # Drop test events
    events = events[~np.isin(events, run_config.test_events)]
    event_df = event_df.loc[event_df.event_id.isin(events)]

    np.random.seed(run_config.seed)

    # Only use a subset of sites for debugging
    if n_sites is not None:
        sites = np.random.choice(sites, size=n_sites, replace=False)

    start = time.time()
    nng.nn_gmm.run_model_training(
        out_dir,
        run_config,
        event_df,
        site_df,
        events,
        None,
        sites,
        None,
        save_train_results=False,
    )
    logger.info(f"Took: {(time.time() - start) / 60} minutes to complete model training.")


@app.command("run-mera")
def run_mera(
    result_dir: Path, site_term: bool = False, out_dir: Path = None, n_procs: int = 4
):
    """Run mixed effects residual analysis (MERA) on the CV validation residuals."""
    nng.utils.setup_logging()
    nng.analysis.run_nn_mera(
        result_dir, site_term=site_term, out_dir=out_dir, n_procs=n_procs
    )


@app.command("obs-fine-tune-cv-nn")
def obs_fine_tune_cv_nn(results_dir: Path, config_ffp: Path, suffix: str = None):
    """Fine-tune the CV-trained NN models using observed data."""
    nng.utils.setup_logging()

    # Create output directory
    output_dir = (
        results_dir / f"obs_fine_tune/{mlt.utils.create_run_name(suffix=suffix)}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)

    tune_config = nng.nn_gmm_obs.FineTuneConfig.from_yaml(config_ffp, device=device)
    nng.nn_gmm_obs.obs_fine_tune_cv(results_dir, tune_config, output_dir)


@app.command("run-hp-opt")
def run_hp_opt(hp_config_ffp: Path, base_run_config_ffp: Path, n_trials: int):
    """Run hyperparameter optimization using Optuna."""
    hp_config = nng.nn_hp_opt.HPOptConfig.from_config(
        hp_config_ffp, base_run_config_ffp, device
    )
    nng.nn_hp_opt.run_hp_opt(hp_config, n_trials)


@app.command("continue-hp-opt")
def continue_hp_opt(study_dir: Path, n_trials: int):
    """Continue a previously started hyperparameter optimization study."""
    nng.nn_hp_opt.continue_hp_opt(study_dir, n_trials)


if __name__ == "__main__":
    app()
