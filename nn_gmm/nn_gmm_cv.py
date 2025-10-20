"""Module for running custom CV for NN-GMM models."""
import gc
import time
import copy
import logging
from pathlib import Path
import multiprocessing as mp

import torch
import numpy as np
import pandas as pd
import xarray as xr
import ml_tools as mlt
from sklearn.model_selection import StratifiedKFold

from . import utils
from . import nn_gmm
from .imdb import DuckIMDB

logger = logging.getLogger(__name__)


def train_cv(
    run_config: nn_gmm.RunConfig,
    n_event_folds: int,
    n_site_folds: int,
    base_out_dir: Path,
    device: str = "cpu",
    n_sites: int = None,
    n_procs: int = 1,
    run_notebook: bool = True,
    remove_cv_results: bool = False,
):
    logger.info(f"Using device: {device.upper()}")

    # Get the data
    with DuckIMDB(run_config.imdb_ffp, readonly=True) as imdb:
        event_df = imdb.get_event_df()
        site_df = imdb.get_site_df(min_grid_level=0, add_nztm=True)

    if run_config.extra_basin_sites:
        # Take all level 0 sites and level 2 & 3 sites that are in a basin
        site_df = utils.add_basin_column(site_df)
        site_df = site_df.loc[
            (site_df.grid_level == 0)
            | ((site_df["basin"] != "NiB") & (site_df.grid_level == 2))
        ]
    else:
        site_df = site_df.loc[site_df.grid_level == 0]

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

    # Create bins for stratified magnitude and vs30 sampling
    site_df["vs30_bin"] = pd.cut(
        site_df["vs30"],
        bins=[0, 180, 360, 760, 1500],
        labels=["vs30_0_180", "vs30_180_360", "vs30_360_760", "vs30_760_1500"],
    )
    event_df["mag_bin"] = pd.cut(
        event_df["magnitude"],
        bins=[5.5, 6.25, 7.25, 7.75, 8.5],
        labels=["mag_5p5_6p25", "mag_6p25_7p25", "mag_7p25_7p75", "mag_7p75_8p5"],
    )

    # Ensure even sampling of events wrt. magnitude
    mag_split = StratifiedKFold(
        n_splits=n_event_folds, shuffle=True, random_state=run_config.seed
    )
    event_folds = [
        events[ind[1]]
        for ind in mag_split.split(
            event_df,
            event_df["mag_bin"],
        )
    ]

    vs30_split = StratifiedKFold(
        n_splits=n_site_folds, shuffle=True, random_state=run_config.seed
    )
    site_folds = [
        sites[ind[1]]
        for ind in vs30_split.split(
            site_df.loc[site_df.site_id.isin(sites)],
            site_df.loc[site_df.site_id.isin(sites), "vs30_bin"],
        )
    ]

    fold_combs = [(i, j) for i in range(n_event_folds) for j in range(n_site_folds)]

    # Run CV
    if n_procs == 1:
        out_dirs = []
        for cv_iter, (train_folds_ind, val_fold_ind) in enumerate(
            get_cv_iterator(fold_combs)
        ):
            logging.info(f"Running CV iteration {cv_iter + 1}/{len(fold_combs)}")
            cur_out_dir = _run_helper(
                copy.deepcopy(run_config),
                event_df,
                site_df,
                event_folds,
                site_folds,
                cv_iter,
                train_folds_ind,
                val_fold_ind,
                base_out_dir,
            )
            out_dirs.append(cur_out_dir)
    else:
        logging.info(f"Running CV with {n_procs} processes.")
        with mp.Pool(n_procs, maxtasksperchild=1) as pool:
            out_dirs = pool.starmap(
                _run_helper,
                [
                    (
                        run_config,
                        event_df,
                        site_df,
                        event_folds,
                        site_folds,
                        cv_iter,
                        train_folds_ind,
                        val_fold_ind,
                        base_out_dir,
                        cv_iter,
                    )
                    for cv_iter, (train_folds_ind, val_fold_ind) in enumerate(
                        get_cv_iterator(fold_combs)
                    )
                ],
            )

    logger.info("Cross-validation run successfully.")

    # Post-processing
    run_config.to_yaml(base_out_dir / "run_config.yaml")

    # Combine validation results & metrics
    logging.info("Combining validation results and metrics.")
    val_results, metrics = [], {}
    for cur_out_dir in out_dirs:
        cur_val_result = pd.read_parquet(cur_out_dir / "val_results.parquet")
        cur_val_result["cv_iter"] = int(cur_out_dir.stem.split("_")[-1])
        val_results.append(cur_val_result)

        metrics[cur_out_dir.stem] = pd.read_parquet(cur_out_dir / "metrics.parquet")

    val_results = pd.concat(val_results, axis=0)
    val_results.to_parquet(base_out_dir / "val_results.parquet")

    # Ensure metrics columns are consistent
    assert np.all(
        [
            np.all(cur_df.columns == metrics["cv_00"].columns)
            for cur_df in metrics.values()
        ]
    )

    # Create xarray DataArray from metrics
    logging.info("Saving metrics")
    cv_iters = list(metrics.keys())
    metrics_array = np.stack([metrics[cv_iter].values for cv_iter in cv_iters], axis=0)
    metrics_da = xr.DataArray(
        metrics_array,
        dims=["cv_iter", "epoch", "metric"],
        coords={
            "cv_iter": cv_iters,
            "epoch": np.arange(run_config.n_epochs),
            "metric": list(metrics["cv_00"].columns),
        },
    )
    metrics_da.to_netcdf(base_out_dir / "metrics.nc")
    
    # Remove the train/validation result files
    # for each CV directory
    if remove_cv_results:
        for cur_cv_dir in out_dirs:
            (cur_cv_dir / "val_results.parquet").unlink()
            (cur_cv_dir / "train_results.parquet").unlink(missing_ok=True)

    if run_notebook:
        results_report_notebook_ffp = (
            Path(__file__).parent / "result_notebooks/cv_result_analysis.ipynb"
        )
        mlt.quarto.render_quarto(
            "mamba activate nn-gmm-gmt",
            results_report_notebook_ffp,
            base_out_dir / "results_report.html",
            result_dir=str(base_out_dir),
        )



def _run_helper(
    run_config: nn_gmm.RunConfig,
    event_df: pd.DataFrame,
    site_df: pd.DataFrame,
    event_folds: list[np.ndarray],
    site_folds: list[np.ndarray],
    cv_iter: int,
    train_folds_ind: list[tuple[int, int]],
    val_fold_ind: tuple[int, int],
    base_out_dir: Path,
    p_ix: int = None,
):
    (out_dir := base_out_dir / f"cv_{cv_iter:02d}").mkdir(parents=True)

    # Set up logging
    log_ffp = out_dir / f"nn_cv_iter_{cv_iter:02d}.log"
    if p_ix is None:
        root_logger = logging.getLogger()
        file_handler = logging.FileHandler(log_ffp)
        file_handler.setLevel(logging.DEBUG)
        root_logger.addHandler(file_handler)
    else:
        logger = utils.setup_logging(log_ffp, enable_console=False)
        logger.info(
            f"Running CV iteration {cv_iter + 1}/{len(event_folds) * len(site_folds)} on process {p_ix}."
        )
        logger.info(
            f"Sleeping for {10 * p_ix} seconds to stagger process start times."
        )
        time.sleep(10 * p_ix)

    val_events = event_folds[val_fold_ind[0]]
    val_sites = site_folds[val_fold_ind[1]]

    train_events = np.unique(
        np.concatenate([event_folds[i] for i, _ in train_folds_ind])
    )
    train_sites = np.unique(np.concatenate([site_folds[j] for _, j in train_folds_ind]))

    nn_gmm.run_model_training(
        out_dir,
        run_config,
        event_df,
        site_df,
        train_events,
        val_events,
        train_sites,
        val_sites,
        save_train_results=False,
        verbose=False,
        # verbose=p_ix is None,
    )

    # Explicit GPU cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # Force garbage collection
    gc.collect()

    if p_ix is None:
        root_logger.removeHandler(file_handler)
    return out_dir


def get_cv_iterator(fold_combs: list[tuple[int, int]]):
    for val_fold_ind in fold_combs:
        train_folds_ind = [
            cur_fold
            for cur_fold in fold_combs
            if (cur_fold[0] != val_fold_ind[0]) and (cur_fold[1] != val_fold_ind[1])
        ]
        yield train_folds_ind, val_fold_ind
