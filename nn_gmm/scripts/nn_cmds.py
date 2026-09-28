import time
import logging
from pathlib import Path

import pandas as pd
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
    result_dir: Path,
    site_term: bool = False,
    out_dir: Path = None,
    n_procs: int = 4,
    ims: list[str] = None,
):
    """Run mixed effects residual analysis (MERA) on the CV validation realisation residuals."""
    logger = nng.utils.setup_logging()

    logger.info("Getting NN residuals")
    res_df, _, __, record_info_df = nng.analysis.get_nn_sim_residuals(result_dir)

    nng.analysis.run_nn_mera(
        res_df,
        record_info_df,
        out_dir if out_dir else result_dir / f"mera{'_site_term' if site_term else ''}",
        site_term=site_term,
        n_procs=n_procs,
        ims=ims,
    )


@app.command("run-event-mera")
def run_event_mera(
    result_dir: Path,
    site_term: bool = False,
    out_dir: Path = None,
    n_procs: int = 4,
    ims: list[str] = None,
):
    """
    Run mixed effects residual analysis (MERA) on the CV validation event residuals."""
    nng.utils.setup_logging()
    nng.analysis.run_event_mera(
        result_dir, site_term=site_term, out_dir=out_dir, n_procs=n_procs, ims=ims
    )


@app.command("run-base-hp-opt")
def run_base_hp_opt(
    hp_config_ffp: Path,
    base_run_config_ffp: Path,
    n_trials: int,
    n_procs: int = 1,
    n_startup_trials: int = 25,
):
    """Run hyperparameter optimization using Optuna."""
    hp_config = nng.nn_hp_opt.BaseModelHPOptConfig.from_config(
        hp_config_ffp, base_run_config_ffp, device
    )
    nng.nn_hp_opt.run_base_hp_opt(
        hp_config, n_trials, n_procs=n_procs, n_startup_trials=n_startup_trials
    )


@app.command("run-loc-adj-hp-opt")
def run_loc_adj_hp_opt(
    hp_config_ffp: Path,
    loc_run_config_ffp: Path,
    rel_base_model_dir: str,
    n_trials: int,
    n_procs: int = 1,
    n_startup_trials: int = 25,
):
    """Run hyperparameter optimization for location adjustment model using Optuna."""
    hp_config = nng.nn_hp_opt.LocAdjModelHPOptConfig.from_config(
        hp_config_ffp, loc_run_config_ffp, rel_base_model_dir, device
    )
    nng.nn_hp_opt.run_loc_adj_hp_opt(
        hp_config,
        n_trials,
        suffix="loc_adj",
        n_procs=n_procs,
        n_startup_trials=n_startup_trials,
    )


@app.command("continue-hp-opt")
def continue_hp_opt(study_dir: Path, n_trials: int, n_procs: int = 1):
    """Continue a previously started hyperparameter optimization study."""
    nng.nn_hp_opt.continue_base_hp_opt(study_dir, n_trials, n_procs=n_procs)


@app.command("compute-cv-shap-values")
def compute_cv_shap_values(result_dirs: Path, n_procs: int = 8):
    """Compute SHAP values for the specified CV results."""
    nng.utils.setup_logging(console_level=logging.DEBUG)
    nng.analysis.compute_cv_shape_values(result_dirs, device, n_procs=n_procs)


@app.command("compute-cv-event-val-results")
def compute_cv_event_val_results(result_dirs: Path):
    """
    Compute event-level validation results for the specified CV results,
    by averaging across the realisations.
    """
    nng.utils.setup_logging(console_level=logging.DEBUG)
    nng.nn_gmm_cv.compute_event_val_results(result_dirs)


@app.command("compute-sigma-ept")
def compute_sigma_ept(
    cv_model_dir: Path,
    mag_bins: list[float] = [3.0, 6.0, 6.5, 7.0, 7.5, 9.0],
):
    """
    Compute the surrogate epistemic uncertainty per 
    tectonic type and magnitude bin, based on the CV residuals.
    """
    run_config = nng.nn_gmm.load_config(cv_model_dir / "run_config.yaml")

    with nng.DuckIMDB(run_config.imdb_ffp, readonly=True) as imdb:
        event_df = imdb.get_event_df()

    event_df["tect_type"] = event_df["tect_type"].astype(str)
    mag_bin = pd.cut(event_df["magnitude"], mag_bins, right=False)
    event_df["mag_bin"] = mag_bin.astype(str)
    event_df["mag_bin_min"] = mag_bin.map(lambda b: b.left).astype(float)
    event_df["mag_bin_max"] = mag_bin.map(lambda b: b.right).astype(float)

    res_df = (
        pd.read_parquet(cv_model_dir / "event_val_results.parquet")[
            run_config.ln_residual_keys
        ]
        .rename(columns=dict(zip(run_config.ln_residual_keys, run_config.ims)))
        .reset_index()
        .join(
            event_df[["tect_type", "mag_bin", "mag_bin_min", "mag_bin_max"]],
            on="event_int_id",
        )
    )
    res_groups = res_df.groupby(["tect_type", "mag_bin"])
    sigma_ept_df = (
        pd.concat(
            {
                "sigma_ept": res_groups[run_config.ims].std(),
                "bias": res_groups[run_config.ims].mean(),
            },
            names=["stat"],
        )
        .reset_index("stat")
        .join(
            res_groups.agg(
                mag_bin_min=("mag_bin_min", "first"),
                mag_bin_max=("mag_bin_max", "first"),
                n_events=("event_int_id", "nunique"),
                n_pairs=("event_int_id", "size"),
            )
        )
        .set_index("stat", append=True)
        .sort_index()
    )

    out_ffp = cv_model_dir / "sigma_ept.csv"
    sigma_ept_df.to_csv(out_ffp)


@app.command("compute-full-test-results")
def compute_full_test_results(model_dir: Path):
    """Compute test results for the specified full model."""
    nng.utils.setup_logging(console_level=logging.DEBUG)
    nng.nn_gmm.compute_full_test_results(model_dir, device)


@app.command("run-test-mera")
def run_test_mera(
    result_dir: Path,
    out_dir: Path = None,
    n_procs: int = 4,
    ims: list[str] = None,
):
    """Run mixed effects residual analysis (MERA) on the full model test realisation residuals."""
    logger = nng.utils.setup_logging()

    if not (result_dir / "test_results.parquet").exists():
        raise FileNotFoundError(
            f"Test results not found in {result_dir}. Please run 'compute-full-test-results' command first."
        )

    logger.info("Getting NN residuals & running MERA")
    res_df, _, __, record_info_df = nng.analysis.get_nn_sim_residuals(
        result_dir, test_results=True
    )

    nng.analysis.run_nn_mera(
        res_df,
        record_info_df,
        (out_dir if out_dir else result_dir / "test_mera"),
        n_procs=n_procs,
        ims=ims,
    )


if __name__ == "__main__":
    app()
