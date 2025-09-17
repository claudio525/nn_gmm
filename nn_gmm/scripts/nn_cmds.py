import logging
import shutil
from pathlib import Path

import numpy as np
import torch
import typer
from sklearn.model_selection import train_test_split

import ml_tools as mlt
import nn_gmm as nng

torch.multiprocessing.set_start_method("spawn", force=True)

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"

print(f"Using device: {device.upper()}")

app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command("train-gmm-holdout")
def train_gmm(
    run_config_ffp: Path,
    n_epochs: int = None,
    batch_size: int = None,
    id_suffix: str = None,
    n_sites: int = None,
):
    log_ffp = Path(__file__).parent / "nn_cmds.log"
    logger = nng.utils.setup_logging(log_ffp)
    print("Writing logs to:", log_ffp)

    run_config = nng.RunConfig.from_config_kwargs(
        config_ffp=run_config_ffp,
        device=device,
        n_epochs=n_epochs,
        batch_size=batch_size,
    )

    logger.info(f"Using device: {device.upper()}")

    # Get the data
    with nng.IMDB(run_config.imdb_ffp, readonly=True) as imdb:
        event_df = imdb.get_event_df()
        site_df = imdb.get_site_df(max_grid_level=0)

    events, sites = event_df.event_id.values.astype(str), site_df.site_id.values.astype(
        str
    )

    # Drop test events
    events = events[~np.isin(events, run_config.test_events)]

    # Only use a subset of sites for testing/debugging
    if n_sites is not None:
        sites = sites[:n_sites]

    logger.info(f"Number of available events: {len(events)}")
    logger.info(f"Number of available sites: {len(sites)}")

    # Split into training and validation sets
    train_events, val_events = train_test_split(
        events, test_size=0.2, random_state=run_config.seed
    )
    train_sites, val_sites = train_test_split(
        sites, test_size=0.2, random_state=run_config.seed
    )
    logger.info(
        f"Number of events - Training: {len(train_events)}, Validation: {len(val_events)}"
    )
    logger.info(
        f"Number of sites - Training: {len(train_sites)}, Validation: {len(val_sites)}"
    )

    id_suffix = f"_{id_suffix}" if id_suffix is not None else ""
    out_dir = run_config.results_dir / f"{mlt.utils.create_run_id(False)}{id_suffix}"
    assert not out_dir.exists(), "Output directory already exists!"

    # Run model training
    nng.nn_gmm.run_model_training(
        out_dir,
        run_config,
        event_df,
        site_df,
        train_events,
        val_events,
        train_sites,
        val_sites,
    )

    # Move log file to output directory
    shutil.move(log_ffp, out_dir / log_ffp.name)


@app.command("train-gmm-cv")
def train_cv(
    run_config_ffp: Path,
    n_event_folds: int,
    n_site_folds: int,
    n_epochs: int = None,
    batch_size: int = None,
    id_suffix: str = None,
    n_sites: int = None,
    n_procs: int = 1,
    run_notebook: bool = True,
    remove_cv_results: bool = False,
):
    """Train and evaluate the GMM using cross-validation."""
    run_config = nng.RunConfig.from_config_kwargs(
        config_ffp=run_config_ffp,
        device=device,
        n_epochs=n_epochs,
        batch_size=batch_size,
    )

    id_suffix = f"_{id_suffix}" if id_suffix is not None else ""
    base_out_dir = (
        run_config.results_dir / f"{mlt.utils.create_run_id(False)}{id_suffix}"
    )
    assert not base_out_dir.exists(), "Output directory already exists!"
    base_out_dir.mkdir()

    log_ffp = base_out_dir / "nn_train_cv.log"
    nng.utils.setup_logging(log_ffp, console_level=logging.DEBUG)
    print("Writing logs to:", log_ffp)

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
