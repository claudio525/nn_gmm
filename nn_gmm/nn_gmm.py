import os
import time
import logging
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd

import ml_tools as mlt

from . import imdb
from . import preprocessing
from . import data
from . import constants
from . import utils

logger = logging.getLogger(__name__)


@dataclass
class RunConfig:

    seed: int

    rel_imdb_ffp: str
    """Relative path to the IMDB file."""

    max_rrup: float
    """Maximum Rrup distance to consider."""

    ignore_events: list[str]
    """List of events to ignore."""

    device: str
    """Device to use"""

    im_set: str
    """IM set to use"""

    site_inputs: list[str]
    """Model site inputs"""

    source_inputs: list[str]
    """Model source inputs"""

    source_to_site_inputs: list[str]
    """Model source to site inputs"""

    batch_size: int
    """Batch size for training"""

    @property
    def imdb_ffp(self) -> Path:
        """
        Get the absolute path to the IMDB file.

        Returns
        -------
        Path
            Absolute path to the IMDB file.
        """
        return Path(os.environ["wdata"]) / self.rel_imdb_ffp

    def to_dict(self) -> dict:
        """
        Convert the RunConfig object to a dictionary.

        Returns
        -------
        dict
            Dictionary representation of the RunConfig object.
        """
        return {
            "seed": int(self.seed),
            "rel_imdb_ffp": str(self.rel_imdb_ffp),
            "max_rrup": list(self.max_rrup),
            "ignore_events": self.ignore_events,
            "device": self.device,
            "im_set": list(self.im_set),
            "site_inputs": list(self.site_inputs),
            "source_inputs": list(self.source_inputs),
            "source_to_site_inputs": list(self.source_to_site_inputs),
            "batch_size": float(self.batch_size),
        }

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

        return cls(**config_dict)

    @classmethod
    def from_dict(cls, d: dict):
        return cls(**d)

    @classmethod
    def from_yaml(cls, ffp: Path):
        return cls.from_dict(mlt.utils.load_yaml(ffp))

    @property
    def ims(self) -> np.ndarray:
        return np.array(constants.IM_SET_MAPPING[self.im_set])


def run_model_training(
    run_config: RunConfig,
    event_df: pd.DataFrame,
    site_df: pd.DataFrame,
    train_events: list[str],
    val_events: list[str],
    train_sites: list[str],
    val_sites: list[str],
):
    events = np.concatenate([train_events, val_events])
    sites = np.concatenate([train_sites, val_sites])

    with imdb.IMDB(run_config.imdb_ffp) as db:
        source_df = db.get_rel_df(events=events)
        site_event_df = db.get_site_event_df(sites=sites, max_rrup=run_config.max_rrup)
        record_info_df = db.get_record_info_df(events=events, sites=sites)

    # Drop records that based not in site_event_df (due to max_rrup)
    record_info_df["site_event_int_id"] = utils.get_site_event_int_id(
        record_info_df.site_int_id.values, record_info_df.event_int_id.values
    )
    drop_mask = ~record_info_df.site_event_int_id.isin(site_event_df.index.values)
    if np.any(drop_mask):
        record_info_df = record_info_df.loc[~drop_mask]
        logger.info(
            f"Dropping {drop_mask.sum()} records that are not in the site_event_df"
        )

    # Add event level source data
    source_df["tect_type"] = event_df.loc[source_df.event_int_id].tect_type.values
    source_df["dip"] = event_df.loc[source_df.event_int_id].dip.values
    source_df["dtop"] = event_df.loc[source_df.event_int_id].dtop.values
    source_df["dbottom"] = event_df.loc[source_df.event_int_id].dbottom.values

    # Run preprocessing
    pre_site_df = preprocessing.pre_process_site_features(
        site_df, run_config.site_inputs
    )

    pre_source_df = preprocessing.pre_process_source_features(
        source_df, run_config.source_inputs
    )

    pre_site_event_df = preprocessing.pre_process_event_site_features(
        site_event_df, run_config.source_to_site_inputs, run_config.max_rrup
    )

    # Get the record ids for the training and validation sets
    train_record_ids = record_info_df.loc[
        record_info_df.event_id.isin(train_events)
        & record_info_df.site_id.isin(train_sites)
    ].index.values.astype(int)
    val_record_ids = record_info_df.loc[
        record_info_df.event_id.isin(val_events)
        & record_info_df.site_id.isin(val_sites)
    ].index.values.astype(int)

    # train_record_ids = np.sort(train_record_ids)
    # val_record_ids = np.sort(val_record_ids)

    train_dataset = data.IMDBDataset(
        run_config.imdb_ffp,
        train_record_ids,
        run_config.ims,
        pre_site_df,
        pre_source_df,
        pre_site_event_df,
        record_info_df,
    )
    val_dataset = data.IMDBDataset(
        run_config.imdb_ffp,
        val_record_ids,
        run_config.ims,
        pre_site_df,
        pre_source_df,
        pre_site_event_df,
        record_info_df,
    )

    train_dataloader = data.CustomDataLoader(
        train_dataset, batch_size=run_config.batch_size, shuffle=True
    )
    val_dataloader = data.CustomDataLoader(
        val_dataset, batch_size=run_config.batch_size, shuffle=False
    )

    start_time = time.time()
    for cur_batch in train_dataloader:
        cur_time = time.time()
        print("Batch time: ", cur_time - start_time)
        start_time = cur_time

    print("wtf")
