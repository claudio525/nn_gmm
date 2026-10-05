import logging
import multiprocessing as mp
import shutil
from pathlib import Path

import ml_tools as mlt
import optuna as opt
import torch
import typer

import nn_gmm as nng

torch.multiprocessing.set_start_method("spawn", force=True)

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"


print(f"Using device: {device.upper()}")

app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command("train-loc-model")
def train_loc_model(
    model_type: str,
    n_train_sites: int,
    n_val_sites: int,
    n_epochs: int,
    units: list[int],
    l2_reg: float,
    batch_size: int,
    activation_fn: str,
    embedding_dim: int,
    dropout_rate: float,
    use_batch_norm: bool,
    base_out_dir: Path,
    suffix: str = "",
    imdb_ffp: Path | None = None,
):
    """Train a location model."""
    log_ffp = Path(__file__).parent / "nn_cmds.log"
    logger = nng.utils.setup_logging(log_ffp)
    print("Writing logs to:", log_ffp)

    logger.info(f"Using device: {device.upper()}")

    # Log all function arguments
    logger.info("Training parameters:")
    logger.info(f"  n_train_sites: {n_train_sites}")
    logger.info(f"  n_val_sites: {n_val_sites}")
    logger.info(f"  n_epochs: {n_epochs}")
    logger.info(f"  units: {units}")
    logger.info(f"  l2_reg: {l2_reg}")
    logger.info(f"  batch_size: {batch_size}")
    logger.info(f"  activation_fn: {activation_fn}")
    logger.info(f"  embedding_dim: {embedding_dim}")
    logger.info(f"  dropout_rate: {dropout_rate}")
    logger.info(f"  use_batch_norm: {use_batch_norm}")

    if model_type == "region":
        out_dir = nng.loc_pre.run_region_model_training(
            n_train_sites=n_train_sites,
            n_val_sites=n_val_sites,
            n_epochs=n_epochs,
            units=units,
            l2_reg=l2_reg,
            batch_size=batch_size,
            activation_fn=activation_fn,
            dropout_rate=dropout_rate,
            embedding_dim=embedding_dim,
            use_batch_norm=use_batch_norm,
            device=device,
            base_out_dir=base_out_dir,
            suffix=suffix,
        )
    elif model_type == "site_cond":
        if imdb_ffp is None:
            raise ValueError("imdb_ffp must be provided for site_cond model training")

        out_dir = nng.loc_pre.run_site_cond_model_training(
            imdb_ffp=imdb_ffp,
            n_train_sites=n_train_sites,
            n_val_sites=n_val_sites,
            n_epochs=n_epochs,
            units=units,
            l2_reg=l2_reg,
            batch_size=batch_size,
            activation_fn=activation_fn,
            dropout_rate=dropout_rate,
            embedding_dim=embedding_dim,
            use_batch_norm=use_batch_norm,
            device=device,
            base_out_dir=base_out_dir,
            suffix=suffix,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    shutil.move(log_ffp, out_dir / log_ffp.name)


@app.command("opt-loc-model")
def opt_loc_model(
    model_type: str,
    base_out_dir: Path,
    n_train_sites: int,
    n_val_sites: int,
    n_epochs: int,
    n_trials: int,
    suffix: str = "",
    n_procs: int = 1,
    imdb_ffp: Path | None = None,
    study_dir: Path | None  = None
):
    """Run hyperparameter optimization for location model."""
    if study_dir is None:
        (study_dir := base_out_dir / mlt.utils.create_run_name(suffix=suffix)).mkdir(
            parents=False, exist_ok=False
        )
        # Create the study first
        study = opt.create_study(
            direction="minimize",
            study_name=study_dir.name,
            storage=f"sqlite:///{study_dir / study_dir.name}.db",
            load_if_exists=True,
        )

    log_ffp = study_dir / "study.log"
    logger = nng.utils.setup_logging(
        log_ffp,
        file_level=logging.WARNING if n_procs > 1 else logging.INFO,
        console_level=logging.WARNING if n_procs > 1 else logging.INFO,
        file_append=True
    )
    print("Writing logs to:", log_ffp)

    if n_procs == 1:
        nng.loc_pre.run_hp_study(
            study_dir,
            model_type,
            n_train_sites,
            n_val_sites,
            n_epochs,
            n_trials,
            device,
            imdb_ffp=imdb_ffp,
        )
    else:
        with mp.Pool(n_procs) as pool:
            pool.starmap(
                nng.loc_pre.run_hp_study,
                [
                    (
                        study_dir,
                        model_type,
                        n_train_sites,
                        n_val_sites,
                        n_epochs,
                        n_trials,
                        device,
                        imdb_ffp,
                        True
                    )
                for _ in range(n_procs)],
            )



if __name__ == "__main__":
    app()
