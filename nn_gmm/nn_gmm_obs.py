import copy
import os
import logging
import typing
from pathlib import Path
from dataclasses import dataclass

import torch
import numpy as np
import pandas as pd
from sklearn import model_selection as ms
import ml_tools as mlt


from . import obs_data as obsd
from . import preprocessing as pre
from . import nn_gmm
from . import data


logger = logging.getLogger(__name__)


@dataclass
class FineTuneConfig:

    seed: int
    """Random seed."""

    rel_nzgmdb_ffp: str
    """Relative file path to NZGMDB data CSV file."""

    validation_split: float
    """Proportion of data to use for validation."""

    l2_reg: float
    """L2 regularization parameter."""

    batch_size: int
    """Batch size for fine-tuning."""

    learning_rate: float
    """Learning rate for fine-tuning."""

    device: str
    """Device to use for training"""

    def __post_init__(self):
        self._base_run_config = None

    @property
    def base_run_config(self) -> nn_gmm.RunConfig | None:
        return self._base_run_config

    @base_run_config.setter
    def base_run_config(self, value: nn_gmm.RunConfig):
        if self._base_run_config is not None:
            raise ValueError("Base RunConfig already set.")
        self._base_run_config = value

    @property
    def nzgmdb_ffp(self) -> Path:
        return Path(os.environ["wdata"]) / self.rel_nzgmdb_ffp

    def to_dict(self) -> dict:
        return {
            "seed": self.seed,
            "rel_nzgmdb_ffp": self.rel_nzgmdb_ffp,
            "validation_split": self.validation_split,
            "l2_reg": self.l2_reg,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "device": self.device
        }

    def to_yaml(self, ffp: Path):
        """Save the RunConfig to a YAML file."""
        mlt.utils.write_to_yaml(self.to_dict(), ffp)

    @classmethod
    def from_dict(cls, d: dict):
        return cls(**d)

    @classmethod
    def from_yaml(cls, ffp: Path, device: str | None = None):
        config_dict = mlt.utils.load_yaml(ffp)
        if device is not None:
            config_dict["device"] = device

        if device is None and "device" not in config_dict:
            raise ValueError(
                "Device must be specified either in " \
                "the config file or as an argument."
            )

        return cls.from_dict(config_dict)


def obs_fine_tune_cv(results_dir: Path, tune_config: FineTuneConfig, output_dir: Path):
    """Fine-tune the CV-trained NN models using observed data."""

    logger.info("Loading observed data...")
    obs_data = obsd.load_obs_nzgmdb(tune_config.nzgmdb_ffp)

    # Drop small events
    logger.info("Filtering observed data...")
    obs_data = obs_data.metadata_filter({"mag": (5.5, 9.0)})
    logger.info(
        f"After filtering - Events: {len(obs_data.events)}, Sites: {len(obs_data.sites)}, Records: {obs_data.n_records}"
    )

    # events, sites = obs_data.events, obs_data.sites

    model_dirs = sorted([r for r in results_dir.glob("cv_*") if r.is_dir()])
    for model_dir in model_dirs:
        run_fine_tune(tune_config, model_dir, obs_data, output_dir / model_dir.name)

    print("wtf")


def run_fine_tune(
    tune_config: FineTuneConfig,
    model_dir: Path,
    obs_data: obsd.ObservedData,
    output_dir: Path,
):
    """Performs fine-tuning for a single CV model."""
    tune_config = copy.deepcopy(tune_config)
    tune_config.base_run_config = nn_gmm.RunConfig.from_yaml(model_dir / "run_config.yaml")

    events, sites = obs_data.events, obs_data.sites

    # Split into train and validation
    train_events, val_events = ms.train_test_split(
        events,
        test_size=tune_config.validation_split,
        random_state=tune_config.seed,
    )
    train_sites, val_sites = ms.train_test_split(
        sites,
        test_size=tune_config.validation_split,
        random_state=tune_config.seed,
    )
    logger.info(
        f"Number of events - Training: {len(train_events)}, Validation: {len(val_events)}"
    )
    logger.info(
        f"Number of sites - Training: {len(train_sites)}, Validation: {len(val_sites)}"
    )
    train_record_ids = obs_data.record_df[
        np.isin(obs_data.record_df.event_id, train_events)
        & np.isin(obs_data.record_df.site_id, train_sites)
    ].index.values.astype(str)
    val_records = obs_data.record_df[
        np.isin(obs_data.record_df.event_id, val_events)
        & np.isin(obs_data.record_df.site_id, val_sites)
    ].index.values.astype(str)
    logger.info(
        f"Number of records - Training: {len(train_record_ids)}, Validation: {len(val_records)}"
    )

    # Pre-processing
    site_df = obs_data.site_df.copy()
    site_df[obsd.ObservedData.SiteColEnums.Z1P0] /= 1000
    pre_site_df = pre.preprocess_site_features(site_df, tune_config.base_run_config.site_inputs)
    event_df = obs_data.event_df.copy().rename(
        columns={
            obsd.ObservedData.EventColEnums.MAG: "magnitude",
            obsd.ObservedData.EventColEnums.ZTOR: "dtop",
            obsd.ObservedData.EventColEnums.ZBOR: "dbottom",
        }
    )
    pre_source_df = pre.preprocess_source_features(event_df, tune_config.base_run_config.source_inputs)
    pre_event_site_df = pre.preprocess_event_site_features(
        obs_data.record_df, tune_config.base_run_config.source_to_site_inputs, tune_config.base_run_config.max_rrup
    )

    train_obs_dataset = data.ObservedDataset(
        pre_site_df.loc[train_sites],
        pre_source_df.loc[train_events],
        pre_event_site_df.loc[train_record_ids],
        obs_data.record_df.loc[train_record_ids, tune_config.base_run_config.ims],
        obs_data.record_df.loc[
            train_record_ids,
            [
                obsd.ObservedData.EventColEnums.EVENT_ID,
                obsd.ObservedData.SiteColEnums.SITE_ID,
            ],
        ],
        tune_config,
    )

    val_obs_dataset = data.ObservedDataset(
        pre_site_df.loc[val_sites],
        pre_source_df.loc[val_events],
        pre_event_site_df.loc[val_records],
        obs_data.record_df.loc[val_records, tune_config.base_run_config.ims],
        obs_data.record_df.loc[
            val_records,
            [ obsd.ObservedData.EventColEnums.EVENT_ID, obsd.ObservedData.SiteColEnums.SITE_ID ],
        ],
        tune_config,
    )

    train_loader = data.CustomDataLoader(train_obs_dataset, tune_config.batch_size, shuffle=True)
    val_loader = data.CustomDataLoader(val_obs_dataset, tune_config.batch_size, shuffle=False)

    model = torch.load(model_dir / "model.pt", weights_only=False, map_location=tune_config.device)
    metrics, _, __ = nn_gmm.train(model, train_loader, val_loader, 10, l2_reg=tune_config.l2_reg, learning_rate=tune_config.learning_rate)

    # Get validation predictions
    val_preds = nn_gmm.get_dataset_predictions(model, val_obs_dataset, tune_config.base_run_config)

    # Write outputs
    output_dir.mkdir(parents=False, exist_ok=False)
    torch.save(model, output_dir / "model.pt")
    metrics_df = pd.DataFrame(metrics)
    val_preds.to_parquet(output_dir / "val_predictions.parquet")
