import copy
import time
import os
import logging
from pathlib import Path
from dataclasses import dataclass, field, fields
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
import numpy as np
import pandas as pd
from tqdm import tqdm

import ml_tools as mlt

from . import DuckIMDB
from . import preprocessing
from . import data
from . import obs_data as obsd
from . import constants
from . import utils
from . import nn_gmm_modules as modules

logger = logging.getLogger(__name__)


@dataclass
class BaseRunConfig:
    """
    Base run configuration for all models.

    Defines common parameters used
    across different model types.
    """

    seed: int

    device: str
    """Device to use"""

    rel_imdb_ffp: str
    """Relative path to the IMDB file."""

    rel_nzgmdb_ffp: str
    """Relative path to the NZGMDB file."""

    rel_test_events_ffp: str
    """Relative path to the test events file."""

    max_rrup: float
    """Maximum Rrup distance to consider."""

    ignore_events: list[str]
    """List of events to ignore."""

    extra_basin_sites: bool
    """Whether to include extra basin sites."""

    im_set: str
    """IM set to use"""

    scale_ims: bool
    """Whether to scale the IMs or not"""

    apply_mag_sample_weighting: bool
    """Whether to apply magnitude-based sample weighting"""

    max_mag_weight: float
    """Maximum weight for magnitude-based sample weighting"""

    apply_rrup_sample_weighting: bool
    """Whether to apply Rrup-based sample weighting"""

    max_rrup_weight: float
    """Maximum additional weight for Rrup-based sample weighting"""

    apply_vs30_sample_weighting: bool
    """Whether to apply Vs30-based sample weighting"""

    max_vs30_weight: float
    """Maximum additional weight for Vs30-based sample weighting"""

    apply_tect_type_sample_weighting: bool
    """Whether to apply tectonic type-based sample weighting"""

    max_tect_type_weight: float
    """Maximum additional weight for tectonic type-based sample weighting"""

    apply_depth_sample_weighting: bool
    """Whether to apply depth-based sample weighting"""

    max_depth_weight: float
    """Maximum additional weight for depth-based sample weighting"""

    total_max_weight: float
    """Maximum additional weight allowed for any sample"""

    apply_im_weighting: bool
    """Whether to apply IM-based sample weighting"""

    im_weights: dict[str, float] | None
    """Weights for each IM"""

    _im_scale_params: dict[str, pd.Series] | None = None

    def __post_init__(self):
        self._test_events = np.load(self.test_events_ffp)

        # Handle loading IM scale parameters from a dict
        if self._im_scale_params is not None:
            tmp = {
                cur_key: pd.Series(cur_dict)
                for cur_key, cur_dict in self._im_scale_params.items()
            }
            self._im_scale_params = tmp

    @property
    def test_events_ffp(self) -> Path:
        """Absolute path to the test events file."""
        return Path(os.environ["wdata"]) / self.rel_test_events_ffp

    @property
    def imdb_ffp(self) -> Path:
        """Absolute path to the IMDB file."""
        return Path(os.environ["wdata"]) / self.rel_imdb_ffp

    @property
    def nzgmdb_ffp(self) -> Path:
        """Absolute path to the NZGMDB file."""
        return Path(os.environ["wdata"]) / self.rel_nzgmdb_ffp

    @property
    def results_dir(self) -> Path:
        """Absolute path to the results directory."""
        return Path(os.environ["wdata"]) / self.rel_results_dir

    @property
    def test_events(self) -> np.ndarray:
        """Get the test events."""
        return self._test_events

    @property
    def ims(self) -> np.ndarray:
        """IMs to use for the model."""
        return np.array(constants.IM_SET_MAPPING[self.im_set])

    @property
    def pSA_ims(self) -> np.ndarray:
        return np.array([cur_im for cur_im in self.ims if cur_im.startswith("pSA")])

    @property
    def pSA_periods(self) -> np.ndarray:
        return np.array([float(cur_im.split("_")[-1]) for cur_im in self.pSA_ims])

    @property
    def ind_loss_keys(self) -> np.ndarray:
        """Individual loss keys."""
        return np.array([f"{cur_im}_loss" for cur_im in self.ims])

    @property
    def pred_mean_keys(self) -> np.ndarray:
        """Predicted mean IM keys."""
        return np.array([f"{cur_im}_pred" for cur_im in self.ims])

    @property
    def ln_residual_keys(self) -> np.ndarray:
        """Residual keys."""
        return np.array([f"{cur_im}_ln_residual" for cur_im in self.ims])

    @property
    def pred_pSA_mean_keys(self) -> np.ndarray:
        """Predicted mean pSA keys."""
        return np.array([f"{cur_im}_pred" for cur_im in self.pSA_ims])

    @property
    def pred_std_keys(self) -> np.ndarray:
        """Predicted IM std keys."""
        return np.array([f"{cur_im}_pred_std" for cur_im in self.ims])

    @property
    def n_ims(self) -> int:
        """Number of IMs."""
        return len(self.ims)

    @property
    def use_sample_weights(self) -> bool:
        """Whether to use sample weights."""
        return (
            self.apply_mag_sample_weighting
            or self.apply_rrup_sample_weighting
            or self.apply_vs30_sample_weighting
        )

    @property
    def im_scale_params(self):
        if self.scale_ims:
            return self._im_scale_params

        raise ValueError("IM standardization is not enabled")

    @im_scale_params.setter
    def im_scale_params(self, value):
        if self.scale_ims:
            if self._im_scale_params is None:
                self._im_scale_params = value
            else:
                raise ValueError("IM standardization is already set")
        else:
            raise ValueError("IM standardization is not enabled")

    def to_dict(self) -> dict:
        """
        Convert the RunConfig object to a dictionary.

        Returns
        -------
        dict
            Dictionary representation of the RunConfig object.
        """
        config_dict = {
            "seed": int(self.seed),
            "rel_imdb_ffp": str(self.rel_imdb_ffp),
            "rel_nzgmdb_ffp": str(self.rel_nzgmdb_ffp),
            "rel_test_events_ffp": str(self.rel_test_events_ffp),
            "max_rrup": float(self.max_rrup),
            "ignore_events": self.ignore_events,
            "extra_basin_sites": bool(self.extra_basin_sites),
            "device": self.device,
            "im_set": str(self.im_set),
            "scale_ims": self.scale_ims,
            "apply_mag_sample_weighting": self.apply_mag_sample_weighting,
            "max_mag_weight": float(self.max_mag_weight),
            "apply_rrup_sample_weighting": self.apply_rrup_sample_weighting,
            "max_rrup_weight": float(self.max_rrup_weight),
            "apply_vs30_sample_weighting": self.apply_vs30_sample_weighting,
            "max_vs30_weight": float(self.max_vs30_weight),
            "apply_tect_type_sample_weighting": self.apply_tect_type_sample_weighting,
            "max_tect_type_weight": float(self.max_tect_type_weight),
            "apply_depth_sample_weighting": self.apply_depth_sample_weighting,
            "max_depth_weight": float(self.max_depth_weight),
            "total_max_weight": float(self.total_max_weight),
            "apply_im_weighting": bool(self.apply_im_weighting),
            "im_weights": self.im_weights,
            "rel_results_dir": str(self.rel_results_dir),
        }
        if self.scale_ims and self.im_scale_params is not None:
            config_dict["_im_scale_params"] = {
                cur_key: cur_df.to_dict()
                for cur_key, cur_df in self.im_scale_params.items()
            }
        return config_dict

    def to_yaml(self, ffp: Path):
        """Save the RunConfig to a YAML file."""
        mlt.utils.write_to_yaml(self.to_dict(), ffp)

    @classmethod
    def from_yaml(cls, ffp: Path):
        return cls.from_dict(mlt.utils.load_yaml(ffp))

    def _repr_html_(self):
        """
        Returns a nice HTML representation of the RunConfig class for Jupyter notebooks.
        Excludes im_scale_params from the display.
        """
        html = "<div style='font-family: monospace; border: 1px solid #ddd; padding: 10px; margin: 5px;'>"
        html += "<h3 style='margin-top: 0; color: #333;'>RunConfig</h3>"
        html += "<table style='border-collapse: collapse; width: 100%;'>"

        # Get all attributes except im_scale_params
        config_dict = self.to_dict()
        config_dict.pop("_im_scale_params", None)  # Remove if present

        for key, value in config_dict.items():
            # Format different types of values
            if isinstance(value, (list, tuple)):
                if len(value) > 5:
                    display_value = (
                        f"[{', '.join(map(str, value[:3]))}, ... ({len(value)} items)]"
                    )
                else:
                    display_value = str(value)
            elif isinstance(value, dict):
                if key == "model":
                    # Special formatting for model config
                    display_value = "<br>".join(
                        [f"&nbsp;&nbsp;{k}: {v}" for k, v in value.items()]
                    )
                else:
                    display_value = f"dict with {len(value)} keys"
            elif isinstance(value, (int, float)):
                display_value = (
                    f"{value:,}"
                    if isinstance(value, int) and value > 1000
                    else str(value)
                )
            else:
                if isinstance(value, BaseRunConfig):
                    display_value = "BaseRunConfig"
                else:
                    display_value = str(value)

            # Add table row with fixed width for variable names
            html += "<tr style='border-bottom: 1px solid #eee;'>"
            html += f"<td style='padding: 5px; font-weight: bold; width: 400px; min-width: 400px; max-width: 400px; vertical-align: top; text-align: left;'>{key}</td>"
            html += f"<td style='padding: 5px; word-break: break-word; text-align: left;'>{display_value}</td>"
            html += "</tr>"

        html += "</table></div>"
        return html


@dataclass
class ModelConfig:

    units: list[int]
    """List of units for each layer in the model."""
    use_batch_norm: bool
    """Whether to use batch normalization in the model."""
    activation: str
    """Activation function to use in the model."""

    dropout_rate: float
    """Dropout rate for the model."""
    l2_reg: float
    """L2 regularization strength."""

    @classmethod
    def from_dict(cls, d: dict):
        """
        Create a ModelConfig instance from a dictionary.

        Parameters
        ----------
        d : dict
            Dictionary containing model configuration parameters.

        Returns
        -------
        ModelConfig
            An instance of ModelConfig.
        """
        return cls(**d)

    @classmethod
    def to_dict(cls, model_config: "ModelConfig") -> dict:
        """
        Convert a ModelConfig instance to a dictionary.

        Parameters
        ----------
        model_config : ModelConfig
            The ModelConfig instance to convert.

        Returns
        -------
        dict
            Dictionary representation of the ModelConfig instance.
        """
        return {
            "units": model_config.units,
            "use_batch_norm": model_config.use_batch_norm,
            "activation": str(model_config.activation),
            "l2_reg": float(model_config.l2_reg),
            "dropout_rate": float(model_config.dropout_rate),
        }


@dataclass
class GMMRunConfig(BaseRunConfig):
    """
    Run configuration for a standard
    ML-surrogate GMM model.

    Defines model inputs, training parameters,
    and model architecture.
    """

    site_inputs: list[str] = field(kw_only=True)
    """Model site inputs"""

    source_inputs: list[str] = field(kw_only=True)
    """Model source inputs"""

    source_to_site_inputs: list[str] = field(kw_only=True)
    """Model source to site inputs"""

    n_epochs: int = field(kw_only=True)
    """Number of epochs to train the model"""

    batch_size: int = field(kw_only=True)
    """Batch size for training"""

    learning_rate: float = field(kw_only=True)
    """Learning rate for training"""

    lr_factor: float | None = field(kw_only=True)
    """Learning rate reduction factor"""

    lr_patience: int | None = field(kw_only=True)
    """Learning rate reduction patience"""

    max_grad_norm: float | None = field(kw_only=True)
    """Maximum gradient norm for core model"""

    model_config: ModelConfig = field(kw_only=True)
    """Model configuration"""

    rel_results_dir: str = field(kw_only=True)
    """Relative path to the results directory."""

    @property
    def n_inputs(self) -> int:
        """Get the number of inputs for the model."""
        n_inputs = (
            len((self.site_inputs))
            + len((self.source_inputs))
            + len(self.source_to_site_inputs)
        )
        if "tect_type" in self.source_inputs:
            n_inputs += len(constants.NN_TECT_TYPES) - 1  # One-hot encoding

        return n_inputs

    @property
    def model_inputs(self) -> list[str]:
        return self.site_inputs + self.source_inputs + self.source_to_site_inputs

    def to_dict(self) -> dict:
        """
        Convert the GMMRunConfig object to a dictionary.

        Returns
        -------
        dict
            Dictionary representation of the GMMRunConfig object.
        """
        base_dict = super().to_dict()
        config_dict = {
            "site_inputs": list(self.site_inputs),
            "source_inputs": list(self.source_inputs),
            "source_to_site_inputs": list(self.source_to_site_inputs),
            "batch_size": int(self.batch_size),
            "learning_rate": float(self.learning_rate),
            "lr_factor": (
                float(self.lr_factor) if self.lr_factor is not None else None
            ),
            "lr_patience": (
                int(self.lr_patience) if self.lr_patience is not None else None
            ),
            "max_grad_norm": (
                float(self.max_grad_norm) if self.max_grad_norm is not None else None
            ),
            "model": ModelConfig.to_dict(self.model_config),
            "rel_results_dir": str(self.rel_results_dir),
            "n_epochs": int(self.n_epochs),
        }
        return base_dict | config_dict

    @classmethod
    def from_config_kwargs(cls, config_ffp: Path, **kwargs):
        """
        Creates an instance from the given config.
        If kwargs are set then they overwrite the values
        specified in the config.
        """
        config_dict = mlt.utils.load_yaml(config_ffp)

        for cur_key, cur_val in kwargs.items():
            if cur_val is not None:
                config_dict[cur_key] = cur_val

        return cls.from_dict(config_dict)

    @classmethod
    def from_dict(cls, d: dict):
        model_config = ModelConfig.from_dict(d.pop("model"))
        d["model_config"] = model_config

        return cls(**d)


@dataclass
class LocAdjRunConfig(BaseRunConfig):
    """
    Run configuration for a location adjustment model.

    Defines base model, location model inputs,
    training parameters, and model architectures.
    """

    rel_base_model_dir: str = field(kw_only=True)
    """Directory of the base model."""

    loc_inputs: list[str] = field(kw_only=True)
    """Location model inputs."""

    loc_emb_dim: int = field(kw_only=True)
    """Dimensionality of the location embedding."""

    loc_emb_model_config: ModelConfig = field(kw_only=True)
    """Model configuration for the location adjustment model."""

    rel_pre_loc_emb_model_dir: str = field(kw_only=True)
    """
    Directory of pre-trained location embedding model.
    If given, loc_emb_model_config and loc_emb_dim are ignored.
    """

    adj_model_config: ModelConfig = field(kw_only=True)
    """Model configuration for the adjustment model."""

    n_epochs: int = field(kw_only=True)
    """Number of training epochs."""

    batch_size: int = field(kw_only=True)
    """Batch size for training."""

    loc_learning_rate: float = field(kw_only=True)
    """Learning rate for location embedding model."""

    adj_learning_rate: float = field(kw_only=True)
    """Learning rate for adjustment model."""

    loc_emb_max_grad_norm: float | None = field(kw_only=True)
    """Maximum gradient norm for training of the location embedding model."""

    adj_max_grad_norm: float | None = field(kw_only=True)
    """Maximum gradient norm for training of the adjustment model."""

    rel_results_dir: str = field(kw_only=True)
    """Relative path to the results directory."""

    base_gmm_run_config: GMMRunConfig | None = field(default=None, init=False)
    """GMMRunConfig of the base model."""

    def __post_init__(self):
        super().__post_init__()
        self.base_gmm_run_config = GMMRunConfig.from_yaml(
            self.base_model_dir / "run_config.yaml"
        )

    @property
    def base_model_dir(self) -> Path:
        """Absolute path to the base model directory."""
        return Path(os.environ["wdata"]) / self.rel_base_model_dir

    @property
    def pre_loc_emb_model_dir(self) -> Path | None:
        """Absolute path to the pre-trained location embedding model directory."""
        if self.rel_pre_loc_emb_model_dir is not None:
            return Path(os.environ["wdata"]) / self.rel_pre_loc_emb_model_dir
        return None

    @property
    def results_dir(self) -> Path:
        """Absolute path to the results directory."""
        return Path(os.environ["wdata"]) / self.rel_results_dir

    #### Pass through properties from base GMM model config ####

    @property
    def site_inputs(self) -> list[str]:
        """Site inputs for the base GMM model."""
        return self.base_gmm_run_config.site_inputs

    @property
    def source_inputs(self) -> list[str]:
        """Source inputs from the base GMM model."""
        return self.base_gmm_run_config.source_inputs

    @property
    def source_to_site_inputs(self) -> list[str]:
        """Source to site inputs from the base GMM model."""
        return self.base_gmm_run_config.source_to_site_inputs

    ### End pass through properties ###

    @property
    def model_inputs(self) -> list[str]:
        """Model inputs for the location adjustment model."""
        return self.base_gmm_run_config.model_inputs + self.loc_inputs

    def to_yaml(self, ffp: Path):
        """Save the RunConfig to a YAML file."""
        mlt.utils.write_to_yaml(self.to_dict(), ffp)

    def to_dict(self) -> dict:
        """
        Convert the LocAdjustmentRunConfig object to a dictionary.

        Returns
        -------
        dict
            Dictionary representation of the LocAdjustmentRunConfig object.
        """
        base_dict = super().to_dict()
        config_dict = {
            "rel_base_model_dir": (
                str(self.rel_base_model_dir) if self.rel_base_model_dir else None
            ),
            "loc_inputs": list(self.loc_inputs),
            "rel_pre_loc_emb_model_dir": (
                str(self.rel_pre_loc_emb_model_dir)
                if self.rel_pre_loc_emb_model_dir
                else None
            ),
            "loc_emb_dim": (
                int(self.loc_emb_dim) if self.loc_emb_dim is not None else None
            ),
            "loc_emb_model_config": (
                ModelConfig.to_dict(self.loc_emb_model_config)
                if self.loc_emb_model_config is not None
                else None
            ),
            "adj_model_config": ModelConfig.to_dict(self.adj_model_config),
            "n_epochs": int(self.n_epochs),
            "batch_size": int(self.batch_size),
            "loc_learning_rate": float(self.loc_learning_rate),
            "adj_learning_rate": float(self.adj_learning_rate),
            "loc_emb_max_grad_norm": (
                float(self.loc_emb_max_grad_norm)
                if self.loc_emb_max_grad_norm is not None
                else None
            ),
            "adj_max_grad_norm": (
                float(self.adj_max_grad_norm)
                if self.adj_max_grad_norm is not None
                else None
            ),
            "rel_results_dir": str(self.rel_results_dir),
        }
        return base_dict | config_dict

    @classmethod
    def from_config_kwargs(cls, config_ffp: Path, **kwargs):
        """
        Creates an instance from the given config.
        If kwargs are set then they overwrite the values
        specified in the config.
        """
        config_dict = mlt.utils.load_yaml(config_ffp)

        for cur_key, cur_val in kwargs.items():
            if cur_val is not None:
                config_dict[cur_key] = cur_val

        return cls.from_dict(config_dict)

    @classmethod
    def from_dict(cls, d: dict):
        if d.get("rel_pre_loc_emb_model_dir"):
            assert (
                d["loc_emb_model_config"] is None and d["loc_emb_dim"] is None
            ), "If rel_pre_loc_emb_model_dir is set, loc_emb_model_config and loc_emb_dim must not be set."
        else:
            loc_emb_model_config = ModelConfig.from_dict(d.pop("loc_emb_model_config"))
            d["loc_emb_model_config"] = loc_emb_model_config

        adj_model_config = ModelConfig.from_dict(d.pop("adj_model_config"))
        d["adj_model_config"] = adj_model_config

        # Add common fields from base GMM config
        missing_base_config_fields = [
            f.name
            for f in fields(BaseRunConfig)
            if f.name not in ["_im_scale_params"] and f.name not in d
        ]
        if len(missing_base_config_fields) > 0:
            gmm_config_ffp = (
                Path(os.environ["wdata"]) / d["rel_base_model_dir"] / "run_config.yaml"
            )
            gmm_config = GMMRunConfig.from_yaml(gmm_config_ffp)
            for f in fields(gmm_config):
                if f.name in missing_base_config_fields and f.name not in d:
                    d[f.name] = getattr(gmm_config, f.name)

        return cls(**d)


class BatchResult(NamedTuple):
    batch: data.SimBatchData
    """The batch data"""

    pred_mean: torch.Tensor
    """The predicted mean values"""
    pred_ln_std: torch.Tensor
    """The predicted standard deviation values in logspace"""
    pred_std: torch.Tensor
    """The predicted standard deviation values"""

    loss: torch.Tensor
    """The batch loss"""
    ind_loss: torch.Tensor
    """The individual losses, this includes nan-values"""
    ind_w_loss: torch.Tensor
    """The individual weighted losses, this includes nan-values"""

    nan_mask: torch.Tensor
    """Mask for the nan values"""


def get_model(
    run_config: GMMRunConfig | LocAdjRunConfig,
) -> tuple[
    modules.BaseNNModel,
    torch.optim.Optimizer | None,
    torch.optim.lr_scheduler.LRScheduler | None,
]:
    """
    Get the model and optimizer for the given run configuration.
    """
    if isinstance(run_config, GMMRunConfig):
        model = modules.BaseNNModel(
            model=modules.create_multi_mlp(
                run_config.n_inputs,
                run_config.model_config.units,
                run_config.n_ims * 2,
                run_config.model_config.activation,
                use_batch_norm=run_config.model_config.use_batch_norm,
                dropout_rate=run_config.model_config.dropout_rate,
            ),
            max_norm=run_config.max_grad_norm,
        )
        optimizer = torch.optim.Adam(
            model.parameters(),
            weight_decay=run_config.model_config.l2_reg,
            lr=run_config.learning_rate,
        )
        if run_config.lr_factor is not None and run_config.lr_patience is not None:
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                factor=run_config.lr_factor,
                patience=run_config.lr_patience,
            )
            return model, optimizer, lr_scheduler

        return model, optimizer, None
    elif isinstance(run_config, LocAdjRunConfig):
        # Load the base model
        base_model = torch.load(
            run_config.base_model_dir / "model.pt",
            weights_only=False,
            map_location=run_config.device,
        )
        assert isinstance(base_model, modules.BaseNNModel) and isinstance(
            base_model.model, nn.Sequential
        )
        # Remove the output layer and clip layer if present
        base_model = base_model.model
        if isinstance(base_model[-1], modules.ClipLayer):
            base_model = nn.Sequential(*list(base_model.children())[:-2])
        else:
            base_model = nn.Sequential(*list(base_model.children())[:-1])

        loc_opt_configs = []
        if run_config.pre_loc_emb_model_dir is not None:
            # Load the pre-trained location embedding model
            loc_emb_model = torch.load(
                run_config.pre_loc_emb_model_dir / "model.pt",
                weights_only=False,
                map_location=run_config.device,
            ).embedding_model
            loc_emb_dim = loc_emb_model.embedding_dim

            lr = run_config.loc_learning_rate
            for cur_layer in loc_emb_model.nn[::-1]:
                if isinstance(cur_layer, nn.Linear):
                    loc_opt_configs.append({"params": cur_layer.parameters(), "lr": lr})
                    lr *= 0.9
        else:
            # Create the location embedding model
            loc_emb_model = modules.create_multi_mlp(
                2,
                run_config.loc_emb_model_config.units,
                run_config.loc_emb_dim,
                run_config.loc_emb_model_config.activation,
                use_batch_norm=run_config.loc_emb_model_config.use_batch_norm,
                dropout_rate=run_config.loc_emb_model_config.dropout_rate,
                add_clip_layer=False,
            )
            loc_emb_dim = run_config.loc_emb_dim
            loc_opt_configs.append(
                {
                    "params": model.loc_emb_model.parameters(),
                    "weight_decay": run_config.loc_emb_model_config.l2_reg,
                    "lr": run_config.loc_learning_rate,
                },
            )

        # Create the adjustment model
        adj_model = modules.create_multi_mlp(
            loc_emb_dim + run_config.base_gmm_run_config.model_config.units[-1],
            run_config.adj_model_config.units,
            run_config.base_gmm_run_config.n_ims * 2,
            run_config.adj_model_config.activation,
            use_batch_norm=run_config.adj_model_config.use_batch_norm,
            dropout_rate=run_config.adj_model_config.dropout_rate,
        )

        model = modules.LocAdjModel(
            loc_emb_model,
            adj_model,
            base_model,
            max_loc_emb_grad_norm=run_config.loc_emb_max_grad_norm,
            max_adj_grad_norm=run_config.adj_max_grad_norm,
        )
        model = model.to(run_config.device)

        # Create the optimizer
        loc_opt_configs.append(
            {
                "params": model.adj_model.parameters(),
                "weight_decay": run_config.adj_model_config.l2_reg,
                "lr": run_config.adj_learning_rate,
            }
        )
        optimizer = torch.optim.Adam(
            loc_opt_configs,
        )
        return model, optimizer, None
    else:
        raise ValueError("Invalid run_config type")


def run_model_training(
    output_dir: Path,
    run_config: GMMRunConfig | LocAdjRunConfig,
    event_df: pd.DataFrame,
    site_df: pd.DataFrame,
    train_record_ids: np.ndarray | None = None,
    val_record_ids: np.ndarray | None = None,
    train_events: list[str] | None = None,
    val_events: list[str] | None = None,
    train_sites: list[str] | None = None,
    val_sites: list[str] | None = None,
    save_train_results: bool = True,
    verbose: bool = True,
):
    """
    Runs the model training process for the given run configuration,
      training and validation events and sites.

    Parameters
    ----------
    ouput_dir : Path
        Directory to save the results.
    run_config : RunConfig
    event_df : pd.DataFrame
        DataFrame containing event information.
    site_df : pd.DataFrame
        DataFrame containing site information.
    train_events : list[str]
        List of training event IDs.
    val_events : list[str] | None
        List of validation event IDs.
        Set to None to skip validation.
    train_sites : list[str]
        List of training site IDs.
    val_sites : list[str] | None
        List of validation site IDs.
        Set to None to skip validation.
    """
    run_config = copy.deepcopy(run_config)
    if train_record_ids is not None:
        assert (
            train_events is None
            and val_events is None
            and train_sites is None
            and val_sites is None
        ), "If train_record_ids is provided, train_events, val_events, "
        "train_sites, and val_sites must be None."
        logger.info("Using record_ids for training/validation data selection.")

        record_ids = np.concatenate(
            [train_record_ids, val_record_ids]
            if val_record_ids is not None
            else [train_record_ids]
        )

        with DuckIMDB(run_config.base_gmm_run_config.imdb_ffp, readonly=True) as imdb:
            record_info_df = imdb.get_record_info_df(record_int_ids=record_ids)
            events = record_info_df.event_id.unique().astype(str)
            source_df = imdb.get_rel_df(events=events)

            # Add site-event-int-id, and update types to reduce memory usage
            record_info_df["site_event_int_id"] = utils.get_site_event_int_id(
                record_info_df.site_int_id.values, record_info_df.event_int_id.values
            )
            site_event_df = imdb.get_site_event_df(
                site_event_int_ids=record_info_df.site_event_int_id.values
            )

        # Sanity check that there is no overlap between train and val sets
        if val_record_ids is not None:
            assert (
                np.isin(train_record_ids, val_record_ids).sum() == 0
            ), "There is overlap between training and validation record IDs."
    else:
        logger.info("Using event and site IDs for training/validation data selection.")
        assert (val_events is not None and val_sites is not None) or (
            val_events is None and val_sites is None
        ), "If validation events are provided, validation sites must be provided and vice versa."
        events = (
            np.concatenate([train_events, val_events])
            if val_events is not None
            else train_events
        )
        sites = (
            np.concatenate([train_sites, val_sites])
            if val_sites is not None
            else train_sites
        )

        # Sanity check
        assert (
            np.isin(run_config.test_events, events).sum() == 0
        ), "Test events are not allowed in the training or validation sets. "

        with DuckIMDB(run_config.imdb_ffp, readonly=True) as db:
            source_df = db.get_rel_df(events=events)
            site_event_df = db.get_site_event_df(
                sites=sites, max_rrup=run_config.max_rrup
            )
            record_info_df = db.get_record_info_df(events=events, sites=sites)

        # Add site-event-int-id, and update types to reduce memory usage
        record_info_df["site_event_int_id"] = utils.get_site_event_int_id(
            record_info_df.site_int_id.values, record_info_df.event_int_id.values
        )

        assert record_info_df.site_event_int_id.dtype == np.int64

        # Drop records that are not in site_event_df (due to max_rrup)
        drop_mask = ~record_info_df.site_event_int_id.isin(site_event_df.index.values)
        if np.any(drop_mask):
            record_info_df = record_info_df.loc[~drop_mask]
            logger.info(
                f"Dropping {drop_mask.sum()} records that are not in the site_event_df due to max_rrup"
            )

        # Get record ids
        train_record_ids = record_info_df.loc[
            record_info_df.event_id.isin(train_events)
            & record_info_df.site_id.isin(train_sites)
        ].index.values.astype(int)
        if val_events is not None:
            val_record_ids = record_info_df.loc[
                record_info_df.event_id.isin(val_events)
                & record_info_df.site_id.isin(val_sites)
            ].index.values.astype(int)

            # Sanity check that there is no overlap between train and val sets
            assert (
                record_info_df.loc[train_record_ids].event_id.isin(val_events).sum()
                == 0
            )
            assert (
                record_info_df.loc[train_record_ids].site_id.isin(val_sites).sum() == 0
            )
            assert (
                record_info_df.loc[val_record_ids].event_id.isin(train_events).sum()
                == 0
            )
            assert (
                record_info_df.loc[val_record_ids].site_id.isin(train_sites).sum() == 0
            )

    record_info_df = record_info_df.astype(
        {"event_int_id": np.uint32, "site_int_id": np.uint32, "rel_int_id": np.uint32}
    )

    # Add event level source data
    source_df["tect_type"] = event_df.loc[source_df.event_int_id].tect_type.values
    source_df["dip"] = event_df.loc[source_df.event_int_id].dip.values
    source_df["dtop"] = event_df.loc[source_df.event_int_id].dtop.values
    source_df["dbottom"] = event_df.loc[source_df.event_int_id].dbottom.values
    source_df["is_point_source"] = (
        event_df.loc[source_df.event_int_id].fault_type.values == "DS_POINT_SOURCE"
    )

    logger.info(
        f"Number of active shallow sources: {(source_df.tect_type == 'ACTIVE_SHALLOW').sum()}"
    )
    logger.info(
        f"Number of volcanic sources: {(source_df.tect_type == 'VOLCANIC').sum()}"
    )
    logger.info(
        f"Number of subduction interface sources: {(source_df.tect_type == 'SUBDUCTION_INTERFACE').sum()}"
    )
    logger.info(
        f"Number of subduction slab sources: {(source_df.tect_type == 'SUBDUCTION_SLAB').sum()}"
    )

    # Add sample weights
    record_info_df = _add_sample_weights(
        run_config,
        record_info_df,
        source_df,
        site_df,
        site_event_df,
    )

    # Run preprocessing
    pre_site_df = preprocessing.preprocess_site_features(
        site_df, run_config.site_inputs
    )
    pre_source_df = preprocessing.preprocess_source_features(
        source_df, run_config.source_inputs
    )
    pre_site_event_df = preprocessing.preprocess_event_site_features(
        site_event_df, run_config.source_to_site_inputs, run_config.max_rrup
    )

    pre_loc_df = None
    if hasattr(run_config, "loc_inputs"):
        pre_loc_df = preprocessing.preprocess_site_features(
            site_df, run_config.loc_inputs
        )

    # Prepare training data
    train_dataset = data.OptimizedIMDBDataset(
        run_config.imdb_ffp,
        train_record_ids,
        run_config.ims,
        pre_site_df,
        pre_source_df,
        pre_site_event_df,
        record_info_df,
        run_config.device,
        run_config.scale_ims,
        loc_df=pre_loc_df,
    )
    run_config.im_scale_params = train_dataset.im_scale_params
    train_dataloader = data.CustomDataLoader(
        train_dataset, batch_size=run_config.batch_size, shuffle=True
    )

    # Prepare validation data
    val_dataset, val_dataloader = None, None
    if val_record_ids is not None:
        val_dataset = data.OptimizedIMDBDataset(
            run_config.imdb_ffp,
            val_record_ids,
            run_config.ims,
            pre_site_df,
            pre_source_df,
            pre_site_event_df,
            record_info_df,
            run_config.device,
            run_config.scale_ims,
            im_scale_params=run_config.im_scale_params,
            loc_df=pre_loc_df,
        )
        val_dataloader = data.CustomDataLoader(
            val_dataset, batch_size=run_config.batch_size, shuffle=False
        )

    # Create the model
    model, optimizer, lr_scheduler = get_model(run_config)
    model.to(run_config.device)

    logger.info(f"Model has {modules.get_n_params(model)} trainable parameters")
    logger.info(f"Training model with {len(train_dataset)} training records ")
    if val_record_ids is not None:
        logger.info(f"Validating model on {len(val_dataset)} records")

    metrics, best_model_state, best_model_epoch = train(
        model,
        optimizer,
        train_dataloader,
        val_dataloader,
        run_config.n_epochs,
        use_sample_weights=run_config.use_sample_weights,
        lr_scheduler=lr_scheduler,
        verbose=verbose,
    )
    metrics_df = pd.DataFrame(metrics)

    # Load the best model
    model.load_state_dict(best_model_state)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get simulation predictions
    if val_record_ids is not None:
        logger.info("Getting validation dataset predictions")
        val_results_df = get_dataset_predictions(
            model, val_dataset, run_config, verbose=verbose
        )
        val_results_df.to_parquet(output_dir / "val_results.parquet")

    if save_train_results:
        logger.info("Getting training dataset predictions")
        train_results_df = get_dataset_predictions(
            model, train_dataset, run_config, verbose=verbose
        )
        train_results_df.to_parquet(output_dir / "train_results.parquet")

    # Get observation predictions
    logging.info("Getting observation predictions")
    obs_data = obsd.load_obs_nzgmdb(run_config.nzgmdb_ffp)
    obs_results_df = run_predictions(
        model, run_config, obsd.get_input_df(obs_data, run_config), run_config.device
    )
    obs_results_df.to_parquet(output_dir / "obs_results.parquet")

    logging.info("Saving results")
    run_config.to_yaml(output_dir / "run_config.yaml")
    metrics_df.to_parquet(output_dir / "metrics.parquet")

    np.save(output_dir / "train_record_ids.npy", train_record_ids)
    np.save(output_dir / "train_events.npy", train_events)
    np.save(output_dir / "train_sites.npy", train_sites)
    if val_record_ids is not None:
        np.save(output_dir / "val_record_ids.npy", val_record_ids)
        if val_events is not None:
            np.save(output_dir / "val_events.npy", val_events)
        if val_sites is not None:
            np.save(output_dir / "val_sites.npy", val_sites)

    torch.save(model, output_dir / "model.pt")

    metadata = {
        "best_model_epoch": int(best_model_epoch),
        "best_model_val_loss": float(metrics["loss_hist_val"][best_model_epoch]),
        "n_train_samples": int(train_record_ids.shape[0]),
        "n_val_samples": (
            int(val_record_ids.shape[0]) if val_record_ids is not None else 0
        ),
        "n_model_params": int(modules.get_n_params(model)),
    }
    mlt.utils.write_to_yaml(metadata, output_dir / "metadata.yaml")


def get_dataset_predictions(
    model: nn.Module,
    dataset: data.IMDBDataset,
    run_config: GMMRunConfig,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Get predictions for the given model and dataset.

    Parameters
    ----------
    model : nn.Module
        The trained model to use for predictions.
    dataset : data.IMDBDataset
        The dataset to get predictions for.
    run_config : RunConfig
        The run configuration containing the necessary parameters.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the predicted mean and standard deviation values.
    """
    dataloader = data.CustomDataLoader(
        dataset, batch_size=run_config.batch_size, shuffle=False
    )

    result_dfs = []
    for cur_batch in tqdm(dataloader, desc="Predicting", disable=not verbose):
        model.eval()
        with torch.no_grad():
            batch_result = _get_batch_result(
                cur_batch, model, run_config.use_sample_weights, has_nan=False
            )

            ind_loss = (
                batch_result.ind_w_loss.numpy(force=True).astype(np.float16)
                if run_config.use_sample_weights
                else batch_result.ind_loss.numpy(force=True).astype(np.float16)
            )

            if run_config.scale_ims:
                pred_mean, pred_std = revert_im_scaling(
                    batch_result.pred_mean.numpy(force=True),
                    run_config,
                    batch_result.pred_std.numpy(force=True),
                )

            cur_result_df = pd.DataFrame(
                data=np.concatenate(
                    [
                        pred_mean,
                        pred_std,
                        ind_loss,
                    ],
                    axis=1,
                ),
                columns=np.concatenate(
                    (
                        run_config.pred_mean_keys,
                        run_config.pred_std_keys,
                        run_config.ind_loss_keys,
                    )
                ),
                index=(
                    cur_batch.record_int_ids
                    if isinstance(cur_batch, data.SimBatchData)
                    else cur_batch.record_ids
                ),
            ).astype({col: np.float16 for col in run_config.ind_loss_keys})

            if run_config.use_sample_weights:
                if len(cur_batch.sample_weights.shape) == 1:
                    cur_result_df["sample_weight"] = cur_batch.sample_weights.numpy(
                        force=True
                    ).astype(np.float32)
                else:
                    logger.info("Sample weights have multiple columns, skipping")

            result_dfs.append(cur_result_df)

    result_df = pd.concat(result_dfs, axis=0)
    return result_df


def train(
    model: modules.BaseNNModel,
    optimizer: torch.optim.Optimizer | None,
    train_dataloader: data.CustomDataLoader,
    val_dataloader: data.CustomDataLoader | None,
    n_epochs: int,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    use_sample_weights: bool = False,
    verbose: bool = True,
):
    """Function for training a model"""
    # Setup metrics to log
    metrics = {
        "w_loss_hist_train": np.zeros(n_epochs),
        "w_loss_hist_val": np.zeros(n_epochs),
        "loss_hist_train": np.zeros(n_epochs),
        "loss_hist_val": np.zeros(n_epochs),
        "mse_hist_train": np.zeros(n_epochs),
        "mse_hist_val": np.zeros(n_epochs),
        "mean_sigma_hist_train": np.zeros(n_epochs),
        "mean_sigma_hist_val": np.zeros(n_epochs),
    }

    best_val_loss = np.inf
    best_model_state, best_model_epoch = None, None

    for cur_epoch_ix in range(n_epochs):
        if verbose:
            logger.debug(f"Epoch: {cur_epoch_ix + 1}/{n_epochs}")

        ### Training
        n_samples = 0
        model.train()
        for cur_batch in tqdm(
            train_dataloader,
            disable=not verbose,
            desc=f"Epoch {cur_epoch_ix + 1}/{n_epochs}",
        ):
            optimizer.zero_grad()

            cur_bresult = _get_batch_result(
                cur_batch, model, use_sample_weights, has_nan=False
            )

            cur_bresult.loss.backward()
            optimizer.step()

            model.apply_grad_clipping()

            model.update_logged_grad_norms()

            metrics = _save_metrics(
                cur_bresult, metrics, use_sample_weights, cur_epoch_ix, "train"
            )
            n_samples += cur_batch.n_samples

        logger.debug(f"\n{model.grad_norm_log_msg()}")

        metrics["w_loss_hist_train"][cur_epoch_ix] /= n_samples
        metrics["loss_hist_train"][cur_epoch_ix] /= n_samples
        metrics["mse_hist_train"][cur_epoch_ix] /= n_samples
        metrics["mean_sigma_hist_train"][cur_epoch_ix] /= n_samples

        ### Validation
        if val_dataloader is not None:
            model.eval()
            n_samples = 0
            with torch.no_grad():
                for cur_batch in val_dataloader:
                    cur_bresult = _get_batch_result(
                        cur_batch, model, use_sample_weights, has_nan=False
                    )

                    metrics = _save_metrics(
                        cur_bresult, metrics, use_sample_weights, cur_epoch_ix, "val"
                    )
                    n_samples += cur_batch.n_samples

            metrics["w_loss_hist_val"][cur_epoch_ix] /= n_samples
            metrics["loss_hist_val"][cur_epoch_ix] /= n_samples
            metrics["mse_hist_val"][cur_epoch_ix] /= n_samples
            metrics["mean_sigma_hist_val"][cur_epoch_ix] /= n_samples

            if lr_scheduler is not None:
                lr_scheduler.step(metrics["w_loss_hist_val"][cur_epoch_ix])

            # Keep track of the best model
            if metrics["w_loss_hist_val"][cur_epoch_ix] < best_val_loss:
                best_val_loss = metrics["w_loss_hist_val"][cur_epoch_ix]
                best_model_state = model.state_dict()
                best_model_epoch = cur_epoch_ix

        logger.info(f"Epoch {cur_epoch_ix + 1}/{n_epochs} completed.")
        logger.info(
            f"Training\t"
            f"Weighted Loss: {metrics['w_loss_hist_train'][cur_epoch_ix]:.4f}, "
            f"MSE: {metrics['mse_hist_train'][cur_epoch_ix]:.5f}"
        )
        if val_dataloader is not None:
            logger.info(
                f"Validation\t"
                f"Weighted Loss: {metrics['w_loss_hist_val'][cur_epoch_ix] :.4f}, "
                f"MSE: {metrics['mse_hist_val'][cur_epoch_ix]:.5f}"
            )

    if val_dataloader is not None:
        logger.info(
            f"Training completed. Best model at epoch "
            f"{best_model_epoch + 1} with val loss {best_val_loss:.4f}"
        )
    else:
        best_model_state = model.state_dict()
        best_model_epoch = n_epochs - 1
        logger.info("Training completed.")

    return metrics, best_model_state, best_model_epoch


def _get_batch_result(
    batch: data.BaseBatchData,
    model: modules.BaseNNModel,
    use_sample_weights: bool,
    has_nan: bool = False,
) -> BatchResult:
    """
    Get the batch result for the given batch and model

    Parameters
    ----------
    batch : data.BaseBatchData
        The batch data
    model : nn.Module
        The model to use
    use_sample_weights : bool
        Whether to use sample weights or not
    has_nan : bool, optional
        Whether the batch has NaN values in the target variable
    """
    if model.uses_loc_inputs:
        pred_mean, pred_ln_std = model(batch.X, batch.X_loc).chunk(2, dim=-1)
    else:
        pred_mean, pred_ln_std = model(batch.X).chunk(2, dim=-1)
    pred_std = torch.exp(pred_ln_std)

    sample_weights = (
        batch.sample_weights
        if len(batch.sample_weights.shape) == 2
        else einops.repeat(batch.sample_weights, "b -> b im", im=batch.y.shape[1])
    )
    if has_nan:
        nan_mask = torch.isnan(batch.y)

        ind_loss = torch.full_like(batch.y, torch.nan)
        ind_loss_ravel = F.gaussian_nll_loss(
            pred_mean[~nan_mask],
            batch.y[~nan_mask],
            pred_std[~nan_mask] ** 2,
            reduction="none",
        )
        ind_loss[~nan_mask] = ind_loss_ravel

        if (nan_count := ind_loss_ravel.isnan().sum()) > 0:
            logger.warning(f"Loss has {nan_count} NaN values!!")

        if use_sample_weights:
            ind_w_loss = sample_weights * ind_loss
            loss = (sample_weights[~nan_mask] * ind_loss_ravel).mean()
        else:
            ind_w_loss = None
            loss = ind_loss_ravel.mean()
    else:
        ind_loss = F.gaussian_nll_loss(
            pred_mean,
            batch.y,
            pred_std**2,
            reduction="none",
        )
        if (nan_count := ind_loss.isnan().sum()) > 0:
            logger.warning(f"Loss has {nan_count} NaN values!!")

        if use_sample_weights:
            ind_w_loss = ind_loss * sample_weights
            loss = ind_w_loss.mean()
        else:
            ind_w_loss = None
            loss = ind_loss.mean()

    return BatchResult(
        batch,
        pred_mean,
        pred_ln_std,
        pred_std,
        loss,
        ind_loss,
        ind_w_loss,
        nan_mask if has_nan else None,
    )


def _save_metrics(
    batch_result: BatchResult,
    metrics: dict[str, np.ndarray[float]],
    use_sample_weights: bool,
    epoch_ix: int,
    result_type: str,
):
    """Computes and saves the metrics for a single batch"""
    loss_hist_key = f"loss_hist_{result_type}"
    w_loss_hist_key = f"w_loss_hist_{result_type}"
    mse_hist_key = f"mse_hist_{result_type}"
    mean_sigma_hist_key = f"mean_sigma_hist_{result_type}"

    # Save metrics
    # Note: This is saved per record, so need to divide by the number of records
    # at the end of the epoch.
    metrics[loss_hist_key][epoch_ix] += (
        batch_result.ind_loss.nanmean(dim=1).sum().item()
    )
    metrics[mse_hist_key][epoch_ix] += (
        F.mse_loss(
            batch_result.pred_mean,
            batch_result.batch.y,
            reduction="none",
        )
        .nanmean(dim=1)
        .sum()
        .item()
    )
    metrics[mean_sigma_hist_key][epoch_ix] += (
        batch_result.pred_std.nanmean(dim=1).sum().item()
    )

    if use_sample_weights:
        metrics[w_loss_hist_key][epoch_ix] += (
            batch_result.ind_w_loss.nanmean(dim=1).sum().item()
        )

    return metrics


def revert_im_scaling(
    scaled_ln_im_mean: np.ndarray[float],
    run_config: GMMRunConfig,
    scaled_ln_im_std: np.ndarray[float] = None,
):
    """
    Reverts the IM scaling

    Parameters
    ----------
    scaled_ln_im_mean: np.ndarray[float]
        The scaled IM (mean) values
    run_config: RunConfig
    scaled_ln_im_std: np.ndarray[float], optional
        The scaled IM standard deviation values

    Returns
    -------
    ln_im_mean: np.ndarray[float]
        The unscaled IM (mean) values
    ln_im_std: np.ndarray[float]
        The unscaled IM standard deviation values.
        Only returned if scaled_ln_im_std is not None.
    """
    ln_im_mean = (
        scaled_ln_im_mean
        * run_config.im_scale_params["std"][run_config.ims].values[None, :]
        + run_config.im_scale_params["mean"][run_config.ims].values[None, :]
    )

    if scaled_ln_im_std is not None:
        ln_im_std = (
            scaled_ln_im_std
            * run_config.im_scale_params["std"][run_config.ims].values[None, :]
        )
        return ln_im_mean, ln_im_std

    return ln_im_mean, None


def get_mag_weights(record_info_df: pd.DataFrame, max_weight: int) -> pd.DataFrame:
    """
    Computes the additional sample weight due to magnitude,
    to be added to the base weight of one.
    """
    record_info_df["mag_bin"] = pd.cut(
        record_info_df.magnitude,
        constants.MAG_WEIGHTING_BINS,
        labels=constants.MAG_WEIGHTING_BIN_NAMES,
    )

    mag_bin_counts = record_info_df.mag_bin.value_counts().sort_index()

    mag_bin_weights = np.clip(
        (mag_bin_counts.max() / mag_bin_counts) - 1, 0.0, max_weight
    )
    record_info_df["mag_weight"] = record_info_df.mag_bin.map(mag_bin_weights).astype(
        np.float16
    )

    return record_info_df


def get_rrup_weights(record_info_df: pd.DataFrame, max_weight: int) -> pd.DataFrame:
    """
    Computes the additional sample weight due to rrup,
    to be added to the base weight of one.
    """
    record_info_df["rrup_bin"] = pd.cut(
        record_info_df.rrup,
        constants.RRUP_WEIGHTING_BINS,
        labels=constants.RRUP_WEIGHTING_BIN_NAMES,
    )

    rrup_bin_counts = record_info_df.rrup_bin.value_counts().sort_index()

    rrup_bin_weights = np.clip(
        (rrup_bin_counts.max() / rrup_bin_counts) - 1, 0.0, max_weight
    )
    record_info_df["rrup_weight"] = record_info_df.rrup_bin.map(
        rrup_bin_weights
    ).astype(np.float16)

    return record_info_df


def get_vs30_weights(record_info_df: pd.DataFrame, max_weight: int) -> pd.DataFrame:
    """
    Computes the additional sample weight due to Vs30,
    to be added to the base weight of one.
    """
    record_info_df["vs30_bin"] = pd.cut(
        record_info_df.vs30,
        constants.VS30_WEIGHTING_BINS,
        labels=constants.VS30_WEIGHTING_BIN_NAMES,
    )

    vs30_bin_counts = record_info_df.vs30_bin.value_counts().sort_index()

    vs30_bin_weights = np.clip(
        (vs30_bin_counts.max() / vs30_bin_counts) - 1, 0.0, max_weight
    )
    record_info_df["vs30_weight"] = record_info_df.vs30_bin.map(
        vs30_bin_weights
    ).astype(np.float16)

    return record_info_df


def get_depth_weights(record_info_df: pd.DataFrame, max_weight: int) -> pd.DataFrame:
    """
    Computes the additional sample weight due to depth,
    to be added to the base weight of one.
    """
    record_info_df["depth_bin"] = pd.cut(
        record_info_df["hypo_depth"],
        constants.DEPTH_WEIGHTING_BINS,
        labels=constants.DEPTH_WEIGHTING_BIN_NAMES,
    )

    depth_bin_counts = record_info_df.depth_bin.value_counts().sort_index()

    depth_bin_weights = np.clip(
        (depth_bin_counts.max() / depth_bin_counts) - 1, 0.0, max_weight
    )
    record_info_df["depth_weight"] = record_info_df.depth_bin.map(
        depth_bin_weights
    ).astype(np.float16)

    return record_info_df


def get_tect_type_weights(
    record_info_df: pd.DataFrame, max_weight: int
) -> pd.DataFrame:
    """
    Computes the additional sample weight due to tectonic type,
    to be added to the base weight of one.
    """
    tect_type_counts = record_info_df["tect_type"].value_counts().sort_index()

    tect_type_weights = np.clip(
        (tect_type_counts.max() / tect_type_counts) - 1, 0.0, max_weight
    )
    record_info_df["tect_type_weight"] = (
        record_info_df["tect_type"].map(tect_type_weights).astype(np.float16)
    )

    return record_info_df


def run_predictions_dir(
    model_dir: Path, input_df: pd.DataFrame, device: str
) -> pd.DataFrame:
    """
    Run predictions using the model stored in the given directory.

    Parameters
    ----------
    model_dir : Path
        Directory containing the model files.
    input_df : pd.DataFrame
        DataFrame containing the input features for the model.
    device : str
        Device to run the model on, e.g., 'cpu' or 'cuda'.
    """
    # Load the model and run config
    run_config = load_config(model_dir / "run_config.yaml")
    model = torch.load(
        model_dir / "model.pt", weights_only=False, map_location=torch.device(device)
    )
    return run_predictions(model, run_config, input_df, device)


def run_predictions(
    model: modules.BaseNNModel,
    run_config: BaseRunConfig,
    input_df: pd.DataFrame,
    device: str,
) -> pd.DataFrame:
    """
    Run predictions using the given model.

    Parameters
    ----------
    model_dir : Path
        Directory containing the model files.
    input_df : pd.DataFrame
        DataFrame containing the input features for the model.
    device : str
        Device to run the model on, e.g., 'cpu' or 'cuda'.
    """
    # Get the input tensors
    X, X_loc = get_input_tensor(run_config, input_df, device)

    model.eval()
    with torch.no_grad():
        if model.uses_loc_inputs:
            pred_mean, pred_ln_std = model(X, X_loc).chunk(2, dim=-1)
        else:
            pred_mean, pred_ln_std = model(X).chunk(2, dim=-1)

        pred_std = torch.exp(pred_ln_std).cpu().numpy()
        pred_mean = pred_mean.cpu().numpy()

    if run_config.scale_ims:
        pred_mean, pred_std = revert_im_scaling(pred_mean, run_config, pred_std)

    pred_mean_df = pd.DataFrame(
        data=pred_mean, columns=run_config.pred_mean_keys, index=input_df.index
    )
    pred_std_df = pd.DataFrame(
        data=pred_std, columns=run_config.pred_std_keys, index=input_df.index
    )
    pred_df = pd.concat([input_df, pred_mean_df, pred_std_df], axis=1)

    return pred_df


def _add_sample_weights(
    run_config: GMMRunConfig,
    record_info_df: pd.DataFrame,
    source_df: pd.DataFrame,
    site_df: pd.DataFrame,
    site_event_df: pd.DataFrame,
):
    record_info_df.loc[:, "sample_weight"] = np.float32(1.0)
    if run_config.use_sample_weights:
        record_info_df.loc[:, "magnitude"] = source_df.loc[
            record_info_df.rel_int_id
        ].magnitude.values.astype(np.float32)
        record_info_df.loc[:, "rrup"] = site_event_df.loc[
            record_info_df.site_event_int_id
        ].rrup.values.astype(np.float32)
        record_info_df.loc[:, "vs30"] = site_df.loc[
            record_info_df.site_int_id
        ].vs30.values.astype(np.float32)
        record_info_df.loc[:, "tect_type"] = source_df.loc[
            record_info_df.rel_int_id
        ].tect_type.values.astype(str)
        record_info_df.loc[:, "hypo_depth"] = source_df.loc[
            record_info_df.rel_int_id
        ].hypo_depth.values.astype(np.float32)

        if run_config.apply_mag_sample_weighting:
            record_info_df = get_mag_weights(
                record_info_df, max_weight=run_config.max_mag_weight
            )
        if run_config.apply_rrup_sample_weighting:
            record_info_df = get_rrup_weights(
                record_info_df, max_weight=run_config.max_rrup_weight
            )
        if run_config.apply_vs30_sample_weighting:
            record_info_df = get_vs30_weights(
                record_info_df, max_weight=run_config.max_vs30_weight
            )
        if run_config.apply_tect_type_sample_weighting:
            record_info_df = get_tect_type_weights(
                record_info_df, max_weight=run_config.max_tect_type_weight
            )
        if run_config.apply_depth_sample_weighting:
            record_info_df = get_depth_weights(
                record_info_df, max_weight=run_config.max_depth_weight
            )

        record_info_df["sample_weight"] += np.clip(
            record_info_df.get("mag_weight", 0).values
            + record_info_df.get("rrup_weight", 0).values
            + record_info_df.get("vs30_weight", 0).values
            + record_info_df.get("tect_type_weight", 0).values
            + record_info_df.get("depth_weight", 0).values,
            0,
            run_config.total_max_weight,
        )

        assert not record_info_df["sample_weight"].isna().any()

    return record_info_df


def load_config(config_ffp: Path) -> GMMRunConfig | LocAdjRunConfig:
    """
    Load the run configuration from the given file path.

    Parameters
    ----------
    config_ffp : Path
        File path to the configuration file.

    Returns
    -------
    GMMRunConfig | LocAdjRunConfig
        The loaded run configuration.
    """
    config_dict = mlt.utils.load_yaml(config_ffp)
    try:
        return GMMRunConfig.from_dict(config_dict)
    except (ValueError, KeyError):
        pass

    try:
        return LocAdjRunConfig.from_dict(config_dict)
    except (ValueError, KeyError):
        pass

    raise ValueError("Invalid configuration file")


def run_full_training(
    run_config: GMMRunConfig | LocAdjRunConfig,
    n_sites: int | None = None,
    id_suffix: str | None = None,
):
    """Runs training with all available data (except test events)"""
    id_suffix = f"_{id_suffix}" if id_suffix is not None else ""
    (
        out_dir := run_config.results_dir
        / f"{mlt.utils.create_run_id(False)}{id_suffix}"
    ).mkdir(parents=False, exist_ok=False)

    log_ffp = out_dir / "nn_train_cv.log"
    logger = utils.setup_logging(log_ffp, console_level=logging.DEBUG)
    print("Writing logs to:", log_ffp)

    # Get event and site data
    with DuckIMDB(run_config.imdb_ffp, readonly=True) as imdb:
        event_df = imdb.get_event_df()
        site_df = imdb.get_site_df(min_grid_level=0, add_nztm=True)

    if run_config.extra_basin_sites:
        # Take all level 0 sites and level 2  sites that are in a basin
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

    start = time.time()
    run_model_training(
        output_dir=out_dir,
        run_config=run_config,
        event_df=event_df,
        site_df=site_df,
        train_events=events,
        train_sites=sites,
        save_train_results=False,
    )
    logger.info(
        f"Took: {(time.time() - start) / 60} minutes to complete model training."
    )


def get_input_dfs(run_config: BaseRunConfig, record_int_ids: np.ndarray):
    """
    Creates the site, source and site-event dataframes for the specified record ids.
    """
    # Load the relevant data
    with DuckIMDB(run_config.imdb_ffp, readonly=True) as imdb:
        site_df = imdb.get_site_df(add_nztm=True)
        event_df = imdb.get_event_df()
        rel_df = imdb.get_rel_df()
        record_info_df = imdb.get_record_info_df(record_int_ids=record_int_ids)
        record_info_df["site_event_int_id"] = utils.get_site_event_int_id(
            record_info_df.site_int_id.values, record_info_df.event_int_id.values
        )
        site_event_df = imdb.get_site_event_df(
            site_event_int_ids=record_info_df.site_event_int_id.values
        )

    ### Build the input dataframes
    # Site inputs
    input_df = record_info_df[
        ["site_int_id", "rel_int_id", "event_int_id", "site_event_int_id"]
    ].copy()
    input_df[run_config.site_inputs] = site_df.loc[
        input_df.site_int_id.values, run_config.site_inputs
    ].values
    assert np.isin(run_config.site_inputs, site_df.columns).all()

    # Source inputs
    if "tect_type" in run_config.source_inputs:
        input_df["tect_type"] = event_df.loc[input_df.event_int_id].tect_type.values
    if "dip" in run_config.source_inputs:
        input_df["dip"] = event_df.loc[input_df.event_int_id].dip.values
    if "dtop" in run_config.source_inputs:
        input_df["dtop"] = event_df.loc[input_df.event_int_id].dtop.values
    if "dbottom" in run_config.source_inputs:
        input_df["dbottom"] = event_df.loc[input_df.event_int_id].dbottom.values
    if "is_point_source" in run_config.source_inputs:
        input_df["is_point_source"] = (
            event_df.loc[input_df.event_int_id].fault_type.values == "DS_POINT_SOURCE"
        )
    if "magnitude" in run_config.source_inputs:
        input_df["magnitude"] = rel_df.loc[input_df.rel_int_id].magnitude.values
    if "hypo_depth" in run_config.source_inputs:
        input_df["hypo_depth"] = rel_df.loc[input_df.rel_int_id].hypo_depth.values
    if "rake" in run_config.source_inputs:
        input_df["rake"] = rel_df.loc[input_df.rel_int_id].rake.values
    assert np.isin(run_config.source_inputs, input_df.columns).all()

    # Site-Event inputs
    input_df[run_config.source_to_site_inputs] = site_event_df.loc[
        input_df.site_event_int_id.values, run_config.source_to_site_inputs
    ].values
    assert np.isin(run_config.source_to_site_inputs, input_df.columns).all()

    input_df = input_df.loc[input_df.rrup < run_config.max_rrup]

    return input_df


def get_input_tensor(
    run_config: BaseRunConfig,
    input_df: pd.DataFrame,
    device: str,
    return_feature_names: bool = False,
):
    """
    Pre-processes the input DataFrame and converts it to a tensor for model input.
    Also processes the loc inputs if it is a location-based model.
    """
    # Pre-process the input DataFrame
    pre_site_df = preprocessing.preprocess_site_features(
        input_df,
        run_config.site_inputs,
    )
    pre_source_df = preprocessing.preprocess_source_features(
        input_df, run_config.source_inputs
    )
    pre_source_site_df = preprocessing.preprocess_event_site_features(
        input_df, run_config.source_to_site_inputs, run_config.max_rrup
    )

    pre_input_df = pd.concat([pre_site_df, pre_source_df, pre_source_site_df], axis=1)
    X = torch.from_numpy(pre_input_df.values).to(dtype=torch.float32, device=device)

    X_loc, loc_feature_names = None, None
    if isinstance(run_config, LocAdjRunConfig) and run_config.loc_inputs is not None:
        pre_loc_input_df = preprocessing.preprocess_site_features(
            input_df, run_config.loc_inputs
        )
        X_loc = torch.from_numpy(pre_loc_input_df.values).to(
            dtype=torch.float32, device=device
        )
        loc_feature_names = pre_loc_input_df.columns.tolist()

    if return_feature_names:
        return X, X_loc, pre_input_df.columns.tolist(), loc_feature_names

    return X, X_loc


def compute_full_test_results(model_dir: Path, device: str):
    """Compute test results for the specified full model."""
    run_config = load_config(model_dir / "run_config.yaml")

    logger.info("Preparing input dataframe")
    with DuckIMDB(run_config.imdb_ffp, readonly=True) as imdb:
        record_info_df = imdb.get_record_info_df(events=run_config.test_events)
        site_df = imdb.get_site_df(add_nztm=True, min_grid_level=1, max_grid_level=1)

        assert record_info_df.event_id.unique().size == run_config.test_events.size
        record_info_df = record_info_df.loc[
            record_info_df.site_int_id.isin(site_df.index)
        ]
        im_df = imdb.get_im_data(
            run_config.ims, record_int_ids=record_info_df.index.values
        )

    input_df = get_input_dfs(run_config, record_info_df.index.values)

    if isinstance(run_config, LocAdjRunConfig) and run_config.loc_inputs is not None:
        input_df = input_df.join(site_df[run_config.loc_inputs], on="site_int_id")

    logger.info("Running predictions")
    model = torch.load(
        model_dir / "model.pt", weights_only=False, map_location=torch.device(device)
    )
    test_results_df = run_predictions(model, run_config, input_df, device)

    # Add true IM values & Save
    test_results_df = test_results_df.join(im_df, on="record_int_id")
    test_results_df.to_parquet(model_dir / "test_results.parquet")
