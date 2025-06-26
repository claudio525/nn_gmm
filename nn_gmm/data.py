import abc
import itertools
import logging
import dataclasses
from pathlib import Path
import time
import typing

import psutil
import pandas as pd
import numpy as np
import torch

from .imdb import IMDB
from . import utils

if typing.TYPE_CHECKING:
    from . import nn_gmm


logger = logging.getLogger(__name__)


@dataclasses.dataclass
class BaseBatchData(abc.ABC):

    @abc.abstractmethod
    def to_tensor(self, device: str = None) -> "BaseBatchData":
        pass


class BaseDataset(abc.ABC):

    @abc.abstractmethod
    def get_batch(self, indices: np.ndarray) -> BaseBatchData:
        pass

    @abc.abstractmethod
    def __len__(self) -> int:
        pass


class BatchData(BaseBatchData):
    """
    Represents a single batch
    """

    def __init__(
        self,
        record_int_ids: np.ndarray,
        X: np.ndarray,
        y: np.ndarray,
    ):
        self.y = y
        self.X = X

        self.record_int_ids = record_int_ids

    @property
    def n_samples(self) -> int:
        return self.y.shape[0]

    def to_tensor(self, device: str = None) -> "BatchData":
        """
        Convert the data to PyTorch tensors
        """
        if device is None:
            self.y = torch.from_numpy(self.y).to(dtype=torch.float32)
            self.X = torch.from_numpy(self.X).to(dtype=torch.float32)
        else:
            self.y = torch.from_numpy(self.y).to(dtype=torch.float32, device=device)
            self.X = torch.from_numpy(self.X).to(dtype=torch.float32, device=device)

        return self

    def __repr__(self):
        return f"{self.__class__.__name__}(record_int_ids={self.record_int_ids.shape}, X={self.X.shape}, y={self.y.shape})"


class CustomDataLoader:
    """
    Loosely based on
    https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/27014/6
    """

    def __init__(self, dataset: BaseDataset, batch_size: int, shuffle: bool):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Calculate number of batches
        self.n_samples = len(self.dataset)
        self.n_batches = int(np.ceil(self.n_samples // self.batch_size))

    def __iter__(self):
        if self.shuffle:
            self.indices = np.random.permutation(self.n_samples)
        else:
            self.indices = np.arange(self.n_samples)
        self.i = 0
        return self

    def __len__(self) -> int:
        return self.n_batches

    def __next__(self) -> BatchData:
        if self.i >= len(self.dataset):
            raise StopIteration

        batch_ind = self.indices[self.i : min(self.i + self.batch_size, self.n_samples)]
        self.i += self.batch_size

        # Get, convert and return batch
        return self.dataset.get_batch(batch_ind).to_tensor()


class IMDBDataset(BaseDataset):

    def __init__(
        self,
        imdb_ffp: Path,
        record_int_ids: np.ndarray,
        ims: np.ndarray,
        site_df: pd.DataFrame,
        source_df: pd.DataFrame,
        site_event_df: pd.DataFrame,
        record_info_df: pd.DataFrame,
        run_config: "nn_gmm.RunConfig",
        is_train: bool,
    ):
        """
        Parameters
        ----------
        imdb_ffp : Path
            The path to the IMDB file
        record_int_ids : np.ndarray
            The record ids to get the IM data for
        ims : np.ndarray
            The IMs to get the data for
        site_df : pd.DataFrame
            The site data
        source_df : pd.DataFrame
            The source data
        site_event_df : pd.DataFrame
            The site event data
        record_info_df : pd.DataFrame
            The record info data
        run_config : nn_gmm.RunConfig
            The run configuration
        is_train : bool
            Whether the dataset is for training
        """
        super().__init__()

        self.imdb = IMDB(imdb_ffp, readonly=True, memory_map_size=30, cache_size=5000)
        self.imdb.open()

        self.record_int_ids = record_int_ids
        self.ims = ims
        self.site_df = site_df
        self.source_df = source_df
        self.site_event_df = site_event_df
        self.record_info_df = record_info_df

        # Check required memory
        mem_req = self.record_int_ids.size * self.ims.size * 8 / 1e9
        logger.info(f"Memory required: {mem_req:.2f} GB")

        if mem_req > (total_mem_avail := psutil.virtual_memory().total / 1e9):
            logger.warning(f"Not enough memory available ({total_mem_avail:.2f} GB)")
            raise MemoryError("Not enough memory available")

        # Load IM data into memory
        logger.info(
            f"Loading IM data for {self.record_int_ids.size} records into memory, will use {mem_req:.2f}GB"
        )
        self._im_data = self.imdb.get_im_data_tmp_table(self.ims, self.record_int_ids)
        # Drop records with zero IM values
        zero_record_ids = self._im_data.loc[self._im_data.sum(axis=1) == 0].index.values
        if len(zero_record_ids) > 0:
            logger.warning(
                f"Dropping {len(zero_record_ids)} records with zero IM values"
            )
            self._im_data = self._im_data.loc[
                ~self._im_data.index.isin(zero_record_ids)
            ]
            self.record_int_ids = self.record_int_ids[
                ~np.isin(self.record_int_ids, zero_record_ids)
            ]
        # Convert to log
        self._im_data = np.log(self._im_data)

        if run_config.scale_ims:
            logger.info("Scaling IM data, shape {self._im_data.shape}")
            # Compute scale parameters
            if is_train:
                logger.info("Calculating mean and std for IM data")
                run_config.im_scale_params = {
                    "mean": self._im_data.mean(axis=0),
                    "std": self._im_data.std(axis=0),
                }
            # Scale the IM data
            self._im_data = (
                self._im_data - run_config.im_scale_params["mean"]
            ) / run_config.im_scale_params["std"]

    def __del__(self):
        try:
            self.imdb.close()
        except Exception as e:
            logger.error(f"Error closing IMDB: {e}")

    def __len__(self) -> int:
        return self.record_int_ids.size

    def get_batch(self, indices: np.ndarray) -> BaseBatchData:
        y = self._im_data.loc[self.record_int_ids[indices]].values

        site_int_ids = self.record_info_df.loc[
            self.record_int_ids[indices]
        ].site_int_id.values
        event_int_ids = self.record_info_df.loc[
            self.record_int_ids[indices]
        ].event_int_id.values
        site_event_int_ids = utils.get_site_event_int_id(site_int_ids, event_int_ids)

        site_data = self.site_df.loc[site_int_ids]
        source_data = self.source_df.loc[
            self.record_info_df.loc[self.record_int_ids[indices]].rel_int_id.values
        ]
        site_event_data = self.site_event_df.loc[site_event_int_ids]

        X = np.concatenate(
            (
                site_data.values,
                source_data.values,
                site_event_data.values,
            ),
            axis=1,
        )

        return BatchData(self.record_int_ids[indices], X, y)


def imdb_get_im_data_batched(
    imdb: IMDB, record_int_ids: np.ndarray, ims: np.ndarray
) -> pd.DataFrame:
    """
    Get IM data from the IMDB in batches
    Most likely better to use the IMDB.get_im_data_tmp_table() method

    Parameters
    ----------
    imdb : IMDB
        The IMDB object, needs to be opened
    record_int_ids : np.ndarray
        The record ids to get the IM data for
    ims : np.ndarray
        The IMs to get the data for

    Returns
    -------
    pd.DataFrame
        The IM data for the given record ids and IMs
    """
    batch_size = 100_000
    n_batches = int(np.ceil(len(record_int_ids) / batch_size))

    im_data = []
    logger.info(
        f"Getting IM data for {len(record_int_ids)} records in {n_batches} batches"
    )
    start_time = time.time()
    for i in range(n_batches):
        start = i * batch_size
        end = min((i + 1) * batch_size, len(record_int_ids))
        cur_record_int_ids = record_int_ids[start:end]

        if i % 10 == 0 and i > 0:
            logger.info(f"Resetting IMDB connection, batch {i} of {n_batches}")
            imdb.close()
            imdb.open()

        cur_im_data = imdb.get_im_data(ims, record_int_ids=cur_record_int_ids)
        im_data.append(cur_im_data)
    logger.info(
        f"Took {time.time() - start_time:.2f} seconds to get IM data for {len(record_int_ids)} records"
    )

    return pd.concat(im_data, axis=0)

def get_similar_records(run_config: "nn_gmm.RunConfig", fixed_inputs: dict, limits: dict, record_int_ids: np.ndarray = None):
    logging.info("Getting similar records for fixed inputs:")
    for k, v in fixed_inputs.items():
        if k in limits:
            logging.info(f"{k}: {v} ± {limits[k]}")
        else:
            logging.info(f"{k}: {v} (no limits defined)")

    with IMDB(run_config.imdb_ffp, readonly=True) as imdb:
        site_df  = imdb.get_site_df()
        event_df = imdb.get_event_df()
        rel_df = imdb.get_rel_df()

    # Valid sites
    for k in run_config.site_inputs:
        if k in fixed_inputs and k in limits:
            site_df = site_df.loc[site_df[k].between(
                fixed_inputs[k] - limits[k][0],
                fixed_inputs[k] + limits[k][1],
            )]
        else:
            logger.info(f"Skipping site input {k}, not in fixed inputs or limits")
    logger.info(f"Found {len(site_df)} valid sites")

    # Valid events & realisations
    for k in run_config.source_inputs:
        if (k in fixed_inputs and k in limits) or k == "tect_type":
            if k == "tect_type":
                event_df = event_df.loc[event_df.tect_type == fixed_inputs[k]]
            # Event property
            elif k in event_df:
                event_df = event_df.loc[event_df[k].between(
                    fixed_inputs[k] - limits[k][0],
                    fixed_inputs[k] + limits[k][1],
                )]
            # Realisation property
            else:
                rel_df = rel_df.loc[rel_df[k].between(
                    fixed_inputs[k] - limits[k][0],
                    fixed_inputs[k] + limits[k][1],
                )]
        else:
            logger.info(f"Skipping source input {k}, not in fixed inputs or limits")
    
    rel_df = rel_df.loc[rel_df.event_int_id.isin(event_df.index)]
    logger.info(f"Found {len(event_df)} valid events and {len(rel_df)} valid realisations")


    comb = np.array(list(itertools.product(site_df.index, event_df.index)))
    site_event_int_ids = utils.get_site_event_int_id(comb[:, 0], comb[:, 1])
    with IMDB(run_config.imdb_ffp, readonly=True) as imdb:
        site_event_df = imdb.get_site_event_df(
            site_event_int_ids=site_event_int_ids,
        )

    # Valid site-event pairs
    for k in run_config.source_to_site_inputs:
        if k in fixed_inputs and k in limits:
            site_event_df = site_event_df.loc[
                site_event_df[k].between(
                    fixed_inputs[k] - limits[k][0],
                    fixed_inputs[k] + limits[k][1],
                )
            ]
        else:
            logger.info(f"Skipping event-site input {k}, not in fixed inputs or limits")
    logger.info(f"Found {len(site_event_df)} valid site-event pairs")

    with IMDB(run_config.imdb_ffp, readonly=True) as imdb:
        record_info_df = imdb.get_record_info_df(
            events=event_df.event_id.values.astype(str),
            sites=site_df.site_id.values.astype(str),
        )

    # Site, Event and Realisation filtering
    record_mask = record_info_df.site_int_id.isin(site_df.index) & \
                  record_info_df.event_int_id.isin(event_df.index) & \
                  record_info_df.rel_int_id.isin(rel_df.index)
    record_info_df = record_info_df.loc[record_mask]

    # Site-Event filtering
    record_info_df["site_event_int_id"] = utils.get_site_event_int_id(
        record_info_df.site_int_id.values,
        record_info_df.event_int_id.values,
    )
    record_info_df = record_info_df.loc[
        record_info_df.site_event_int_id.isin(site_event_df.index)
    ]

    if record_int_ids is not None:
        record_info_df = record_info_df.loc[record_info_df.index.isin(record_int_ids)]

    return record_info_df.index.values
