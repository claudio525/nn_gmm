import os
import copy
import functools
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import xarray as xr
import optuna as opt

import ml_tools as mlt

from .nn_gmm import GMMRunConfig
from .nn_gmm_cv import train_cv


@dataclass
class HPOptConfig:

    rel_base_output_dir: str
    base_run_config: GMMRunConfig

    n_event_folds: int
    n_site_folds: int
    n_epochs: int

    batch_sizes: list[int]
    learning_rates: list[float]

    n_layers_min: int
    n_layers_max: int
    unit_sizes: list[int]
    last_hlayer_sizes: list[int]
    activation_fns: list[str]
    l2_regs: list[float]
    dropout_rates: list[float]

    def __post_init__(self):
        self._study_dir = None

    @property
    def base_output_dir(self) -> Path:
        return Path(os.environ["wdata"]) / self.rel_base_output_dir

    @property
    def study_dir(self) -> Path:
        return self._study_dir

    @study_dir.setter
    def study_dir(self, value: Path):
        if self._study_dir is not None:
            raise ValueError("study_dir has already been set and cannot be modified.")
        self._study_dir = value

    @classmethod
    def from_config(cls, config_ffp: Path, run_config_ffp: Path, device: str):
        run_config = GMMRunConfig.from_config_kwargs(run_config_ffp, device=device)
        config_dict = mlt.utils.load_yaml(config_ffp)

        return cls(
            rel_base_output_dir=config_dict["rel_base_output_dir"],
            base_run_config=run_config,
            n_event_folds=config_dict["n_event_folds"],
            n_site_folds=config_dict["n_site_folds"],
            n_epochs=config_dict["n_epochs"],
            batch_sizes=config_dict["batch_sizes"],
            learning_rates=config_dict["learning_rates"],
            n_layers_min=config_dict["n_layers_min"],
            n_layers_max=config_dict["n_layers_max"],
            unit_sizes=config_dict["unit_sizes"],
            last_hlayer_sizes=config_dict["last_hlayer_sizes"],
            activation_fns=config_dict["activation_fns"],
            l2_regs=config_dict["l2_regs"],
            dropout_rates=config_dict["dropout_rates"],
        )

    def to_dict(self) -> dict:
        return {
            "rel_base_output_dir": self.rel_base_output_dir,
            "base_run_config": self.base_run_config.to_dict(),
            "n_event_folds": self.n_event_folds,
            "n_site_folds": self.n_site_folds,
            "n_epochs": self.n_epochs,
            "batch_sizes": self.batch_sizes,
            "learning_rates": self.learning_rates,
            "n_layers_min": self.n_layers_min,
            "n_layers_max": self.n_layers_max,
            "unit_sizes": self.unit_sizes,
            "last_hlayer_sizes": self.last_hlayer_sizes,
            "activation_fns": self.activation_fns,
            "l2_regs": self.l2_regs,
            "dropout_rates": self.dropout_rates,
        }

    def to_yaml(self, ffp: Path):
        """Save the RunConfig to a YAML file."""
        mlt.utils.write_to_yaml(self.to_dict(), ffp)

    @classmethod
    def from_dict(cls, config_dict: dict):
        return cls(
            rel_base_output_dir=config_dict["rel_base_output_dir"],
            base_run_config=GMMRunConfig.from_dict(config_dict["base_run_config"]),
            n_event_folds=config_dict["n_event_folds"],
            n_site_folds=config_dict["n_site_folds"],
            n_epochs=config_dict["n_epochs"],
            batch_sizes=config_dict["batch_sizes"],
            learning_rates=config_dict["learning_rates"],
            n_layers_min=config_dict["n_layers_min"],
            n_layers_max=config_dict["n_layers_max"],
            unit_sizes=config_dict["unit_sizes"],
            activation_fns=config_dict["activation_fns"],
            l2_regs=config_dict["l2_regs"],
            dropout_rates=config_dict["dropout_rates"],
        )

    @classmethod
    def from_yaml(cls, ffp: Path):
        return cls.from_dict(mlt.utils.load_yaml(ffp))


def continue_hp_opt(study_dir: Path, n_trials: int):
    """Continue a previously started hyperparameter optimization study."""
    hp_config = HPOptConfig.from_yaml(study_dir / "hp_config.yaml")
    hp_config.study_dir = study_dir

    study = opt.create_study(
        study_name=study_dir.name,
        storage="sqlite:///{}.db".format(study_dir / study_dir.name),
        load_if_exists=True,
    )
    study.optimize(functools.partial(objective, hp_config=hp_config), n_trials=n_trials)

def run_hp_opt(hp_config: HPOptConfig, n_trials: int, suffix: str = "", n_procs: int = 1, n_startup_trials: int = 25):
    """Run hyperparameter optimization using Optuna."""
    objective_fn_call = functools.partial(objective, hp_config=hp_config, n_procs=n_procs)

    study_id = mlt.utils.create_run_id()
    study_name = f"{study_id}{f'_{suffix}' if suffix else ''}"
    hp_config.study_dir = hp_config.base_output_dir / study_name
    hp_config.study_dir.mkdir(parents=False, exist_ok=False)
    hp_config.to_yaml(hp_config.study_dir / "hp_config.yaml")

    # Create the study & start optimizing
    study = opt.create_study(
        study_name=study_name,
        direction="minimize",
        sampler=opt.samplers.TPESampler(n_startup_trials=n_startup_trials, n_ei_candidates=1000),
        storage="sqlite:///{}.db".format(hp_config.study_dir / study_name),
    )
    study.optimize(objective_fn_call, n_trials=n_trials)


def objective(trial: opt.Trial, hp_config: HPOptConfig, n_procs: int) -> float:
    """Objective function for hyperparameter optimization."""
    run_config = _get_run_config(trial, hp_config)
    run_config.n_epochs = hp_config.n_epochs

    output_dir = hp_config.study_dir / f"trial_{trial.number:03d}"
    output_dir.mkdir(parents=False, exist_ok=False)

    train_cv(
        run_config,
        hp_config.n_event_folds,
        hp_config.n_site_folds,
        output_dir,
        device=run_config.device,
        n_procs=n_procs,
        run_notebook=False,
        remove_cv_results=True,

    )

    metrics = xr.open_dataarray(output_dir / "metrics.nc")
    w_val_loss_data = metrics.sel(metric="w_loss_hist_val")
    median_w_val_loss = float(w_val_loss_data.min(dim="epoch").median())

    trial.set_user_attr("median_w_val_loss", )
    trial.set_user_attr("percentile_16_84_w_val_loss", tuple(np.percentile(w_val_loss_data.min(dim="epoch").values, [16, 84])))
    trial.set_user_attr("median_best_epoch", float(w_val_loss_data.argmin(dim="epoch").median()))

    (output_dir / "val_results.parquet").unlink()

    return median_w_val_loss


def _get_run_config(trial: opt.Trial, hp_config: HPOptConfig) -> GMMRunConfig:
    """Get a RunConfig object with hyperparameters set from the trial."""
    run_config = copy.deepcopy(hp_config.base_run_config)
    run_config.rel_results_dir = "nn_gmm/hp_opt"

    run_config.batch_size = trial.suggest_categorical(
        "batch_size", hp_config.batch_sizes
    )
    run_config.learning_rate = trial.suggest_categorical(
        "learning_rate", hp_config.learning_rates
    )

    n_layers = trial.suggest_int(
        "n_layers", hp_config.n_layers_min, hp_config.n_layers_max
    )
    unit_size = trial.suggest_categorical("unit_size", hp_config.unit_sizes)
    last_layer_size = trial.suggest_categorical("last_hlayer_size", hp_config.last_hlayer_sizes)
    run_config.model_config.units = [unit_size] * (n_layers - 1) + [last_layer_size]

    run_config.model_config.activation = trial.suggest_categorical(
        "activation_fn", hp_config.activation_fns
    )
    run_config.model_config.l2_reg = trial.suggest_categorical(
        "l2_reg", hp_config.l2_regs
    )
    run_config.model_config.dropout_rate = trial.suggest_categorical(
        "dropout_rate", hp_config.dropout_rates
    )
    run_config.model_config.use_batch_norm = trial.suggest_categorical(
        "use_batch_norm", [True, False]
    )

    return run_config
