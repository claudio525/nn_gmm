import abc
import itertools
import logging
import dataclasses
from pathlib import Path
import time
import typing

import einops
import psutil
import pandas as pd
import numpy as np
import torch

from .imdb import DuckIMDB
from . import utils

if typing.TYPE_CHECKING:
    from . import nn_gmm
    from . import nn_gmm_obs


logger = logging.getLogger(__name__)


@dataclasses.dataclass
class BaseBatchData(abc.ABC):
    pass


class BaseDataset(abc.ABC):

    @abc.abstractmethod
    def get_batch(self, indices: np.ndarray) -> BaseBatchData:
        pass

    @abc.abstractmethod
    def __len__(self) -> int:
        pass


class SimBatchData(BaseBatchData):
    """
    Represents a single batch for
    training/inference on simulation data
    """

    def __init__(
        self,
        record_int_ids: np.ndarray,
        X: torch.Tensor,
        y: torch.Tensor,
        X_loc: torch.Tensor | None = None,
        sample_weights: torch.Tensor | None = None,
    ):
        self.y = y
        self.X = X
        self.X_loc = X_loc
        self.sample_weights = sample_weights
        self.record_int_ids = record_int_ids

    @property
    def n_samples(self) -> int:
        return self.y.shape[0]

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(record_int_ids={self.record_int_ids.shape}, X={self.X.shape}, "
            f"y={self.y.shape}, X_loc={self.X_loc.shape if self.X_loc is not None else None}, "
            f"sample_weights={self.sample_weights.shape if self.sample_weights is not None else None})"
        )


class ObsBatchData(BaseBatchData):
    """
    Represents a single batch for
    training/inference on observed data
    """

    def __init__(
        self,
        record_ids: np.ndarray,
        X: torch.Tensor,
        y: torch.Tensor,
        sample_weights: torch.Tensor | None = None,
    ):
        self.y = y
        self.X = X
        self.sample_weights = sample_weights

        self.record_ids = record_ids

    @property
    def n_samples(self) -> int:
        return self.y.shape[0]

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(record_ids={self.record_ids.shape}, X={self.X.shape}, "
            f"y={self.y.shape}, sample_weights={self.sample_weights.shape if self.sample_weights is not None else None})"
        )


class CustomDataLoader:
    """
    Loosely based on
    https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/27014/6
    """

    def __init__(
        self,
        dataset: BaseDataset,
        batch_size: int,
        shuffle: bool,
        use_torch: bool = True,
        device: torch.device = "cpu",
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.use_torch = use_torch
        self.device = device

        # Calculate number of batches
        self.n_samples = len(self.dataset)
        self.n_batches = int(np.ceil(self.n_samples // self.batch_size))

    def __iter__(self):
        if self.shuffle:
            # self.indices = np.random.permutation(self.n_samples)
            self.indices = (
                torch.randperm(self.n_samples, device=self.device)
                if self.use_torch
                else np.random.permutation(self.n_samples)
            )
        else:
            # self.indices = np.arange(self.n_samples)
            self.indices = (
                torch.arange(self.n_samples, device=self.device)
                if self.use_torch
                else np.arange(self.n_samples)
            )
        self.i = 0
        return self

    def __len__(self) -> int:
        return self.n_batches

    def __next__(self) -> BaseBatchData:
        if self.i >= len(self.dataset):
            raise StopIteration

        batch_ind = self.indices[self.i : min(self.i + self.batch_size, self.n_samples)]
        self.i += self.batch_size

        # Get, convert and return batch
        batch = self.dataset.get_batch(batch_ind)
        return batch


class BaseIMDBDataset(BaseDataset):
    def __init__(
        self,
        imdb_ffp: Path,
        record_int_ids: np.ndarray,
        ims: np.ndarray,
        site_df: pd.DataFrame,
        source_df: pd.DataFrame,
        site_event_df: pd.DataFrame,
        record_info_df: pd.DataFrame,
        device: str,
        scale_ims: bool,
        im_scale_params: dict | None = None,
        loc_df: pd.DataFrame | None = None,
    ):
        """
        Parameters
        ----------
        imdb_ffp : Path
            The path to the IMDB file
        record_int_ids : np.ndarray
            The record ids to include
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
        im_scale_params : dict | None
            The scaling parameters for the IM data
        """
        super().__init__()

        self.device = device
        self.ims = ims
        self.record_int_ids = np.sort(record_int_ids)
        self.im_scale_params = im_scale_params
        self._return_X_loc = loc_df is not None

        # Check required memory
        mem_req = self.record_int_ids.size * self.ims.size * 4 / 1e9
        logger.info(f"Memory required: {mem_req:.2f} GB")

        if mem_req > (total_mem_avail := psutil.virtual_memory().total / 1e9):
            logger.warning(f"Not enough memory available ({total_mem_avail:.2f} GB)")
            raise MemoryError("Not enough memory available")

        # Load IM data into memory
        logger.info(
            f"Loading IM data for {self.record_int_ids.size} "
            f"records into memory, will use {mem_req:.2f}GB"
        )

        with DuckIMDB(imdb_ffp, readonly=True) as imdb:
            self._im_data = imdb.get_im_data(self.ims, self.record_int_ids).sort_index()

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

        # Filter dataframes to only include relevant records
        self.record_info_df = record_info_df.loc[self.record_int_ids]
        self.site_df = site_df.loc[
            site_df.index.isin(self.record_info_df.site_int_id)
        ].sort_index()
        self.source_df = source_df.loc[
            source_df.index.isin(self.record_info_df.rel_int_id)
        ].sort_index() 
        self.site_event_df = site_event_df.loc[
            site_event_df.index.isin(self.record_info_df.site_event_int_id)
        ].sort_index() 

        self.loc_df = None
        if self._return_X_loc:
            self.loc_df = loc_df.loc[
                loc_df.index.isin(self.record_info_df.site_int_id)
            ].sort_index()

        # Convert to log
        self._im_data = np.log(self._im_data)

        if scale_ims:
            logger.info(f"Scaling IM data, shape {self._im_data.shape}")
            # Compute scale parameters
            if self.im_scale_params is None:
                logger.info(
                    "No IM scale parameters provided, calculating mean and std for IM data"
                )
                self.im_scale_params = {
                    "mean": self._im_data.mean(axis=0),
                    "std": self._im_data.std(axis=0),
                }
            # Scale the IM data
            self._im_data = (
                self._im_data - self.im_scale_params["mean"]
            ) / self.im_scale_params["std"]

    def __len__(self) -> int:
        return self.record_int_ids.size


class OptimizedIMDBDataset(BaseIMDBDataset):

    def __init__(
        self,
        imdb_ffp: Path,
        record_int_ids: np.ndarray,
        ims: np.ndarray,
        site_df: pd.DataFrame,
        source_df: pd.DataFrame | None,
        site_event_df: pd.DataFrame | None,
        record_info_df: pd.DataFrame,
        device: str,
        scale_ims: bool,
        im_scale_params: dict | None = None,
        im_weights: dict | None = None,
        loc_df: pd.DataFrame | None = None,
    ) -> None:
        super().__init__(
            imdb_ffp,
            record_int_ids,
            ims,
            site_df,
            source_df,
            site_event_df,
            record_info_df,
            device,
            scale_ims,
            im_scale_params=im_scale_params,
            loc_df=loc_df,
        )

        assert np.all(self.record_int_ids == self._im_data.index.values)
        assert np.all(self.record_info_df.index.values == self.record_int_ids)

        self._index_to_site_ix = self.site_df.index.get_indexer(
            self.record_info_df.loc[self.record_int_ids].site_int_id.values
        )
        self._index_to_source_ix = (
            self.source_df.index.get_indexer(
                self.record_info_df.loc[self.record_int_ids].rel_int_id.values
            )
        )
        self._index_to_site_event_ix = (
            self.site_event_df.index.get_indexer(
                self.record_info_df.loc[self.record_int_ids].site_event_int_id.values
            )
        )

        self._im_data_tensor = torch.tensor(
            self._im_data.values, device=device, dtype=torch.float32
        )
        self._site_data_tensor = torch.tensor(
            self.site_df.values,
            device=device,
            dtype=torch.float32,
        ) 
        self._source_data_tensor = torch.tensor(
            self.source_df.values, device=device, dtype=torch.float32
        ) 
        self._site_event_tensor = torch.tensor(
            self.site_event_df.values, device=device, dtype=torch.float32
        ) 

        if self._return_X_loc:
            self._loc_data_tensor = torch.tensor(
                self.loc_df.values,
                device=device,
                dtype=torch.float32,
            )

        self._sample_weight_tensor = torch.tensor(
            self.record_info_df["sample_weight"].values,
            device=device,
            dtype=torch.float32,
        )
        if im_weights is not None:
            im_weights = np.ones(ims.size, dtype=float)
            for cur_im, w in im_weights.items():
                im_ix = np.flatnonzero(self.ims == cur_im)
                if im_ix.size == 0:
                    logger.warning(f"IM {cur_im} not found in dataset IMs")
                else:
                    im_weights[im_ix] = w

            self._sample_weight_tensor = einops.repeat(
                self._sample_weight_tensor, "n -> n im", im=ims.size
            ).clone()
            self._sample_weight_tensor *= torch.tensor(
                im_weights, device=device, dtype=torch.float32
            )

    def get_batch(self, indices: np.ndarray) -> BaseBatchData:
        """
        Get a batch of data for the given indices.

        indices: np.ndarray
            GM record indices
        """
        y = self._im_data_tensor[indices, :]

        site_data = torch.atleast_2d(
            self._site_data_tensor[self._index_to_site_ix[indices], :]
        )
        source_data = torch.atleast_2d(
            self._source_data_tensor[self._index_to_source_ix[indices], :]
        )
        site_event_data = torch.atleast_2d(
            self._site_event_tensor[self._index_to_site_event_ix[indices], :]
        )

        X = torch.cat(
            [site_data, source_data, site_event_data],
            dim=1,
        )

        X_loc = None
        if self._return_X_loc:
            X_loc = torch.atleast_2d(
                self._loc_data_tensor[self._index_to_site_ix[indices], :]
            )
            
        return SimBatchData(
            self.record_int_ids[indices],
            X,
            y,
            X_loc=X_loc,
            sample_weights=self._sample_weight_tensor[indices],
        )


class IMDBDataset(BaseIMDBDataset):

    def __init__(
        self,
        imdb_ffp: Path,
        record_int_ids: np.ndarray,
        ims: np.ndarray,
        site_df: pd.DataFrame,
        source_df: pd.DataFrame,
        site_event_df: pd.DataFrame,
        record_info_df: pd.DataFrame,
        device: str,
        scale_ims: bool,
        im_scale_params: dict | None = None,
        im_weights: dict | None = None,
    ) -> None:
        super().__init__(
            imdb_ffp,
            record_int_ids,
            ims,
            site_df,
            source_df,
            site_event_df,
            record_info_df,
            device,
            scale_ims,
            im_scale_params=im_scale_params,
        )

        if im_weights is not None:
            raise NotImplementedError(
                "IM-based weighting not implemented for this dataset"
            )

    def get_batch(self, indices: np.ndarray) -> BaseBatchData:
        """
        Get a batch of data for the given indices.

        indices: np.ndarray
            GM record indices
        """
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

        return SimBatchData(
            self.record_int_ids[indices],
            torch.from_numpy(X).to(dtype=torch.float32, device=self.device),
            torch.from_numpy(y).to(dtype=torch.float32, device=self.device),
        )


class ObservedDataset(BaseDataset):

    def __init__(
        self,
        site_df: pd.DataFrame,
        source_df: pd.DataFrame,
        event_site_df: pd.DataFrame,
        im_df: pd.DataFrame,
        record_info_df: pd.DataFrame,
        tune_config: "nn_gmm_obs.FineTuneConfig",
    ):
        super().__init__()

        self.tune_config = tune_config
        self.ims = self.tune_config.base_run_config.ims
        assert np.all(np.isin(self.ims, im_df.columns))

        self.site_df = site_df
        self.source_df = source_df
        self.event_site_df = event_site_df
        self.record_info_df = record_info_df
        self.im_df = im_df[self.ims].copy()
        assert self.record_info_df.index.equals(self.im_df.index)
        assert self.record_info_df.index.equals(self.event_site_df.index)
        self._record_ids = self.record_info_df.index.values.astype(str)

        # Pre-process IM data
        self.im_df = np.log(self.im_df)
        im_scale_params = self.tune_config.base_run_config.im_scale_params
        if im_scale_params is not None:
            self.im_df = (self.im_df - im_scale_params["mean"]) / im_scale_params["std"]

        # Create lookup indices
        self._index_to_site_ix = self.site_df.index.get_indexer(
            self.record_info_df.site_id.values
        )
        self._index_to_event_ix = self.source_df.index.get_indexer(
            self.record_info_df.event_id.values
        )

        self._site_data_tensor = torch.tensor(
            self.site_df.values, device=self.tune_config.device, dtype=torch.float32
        )
        self._source_data_tensor = torch.tensor(
            self.source_df.values, device=self.tune_config.device, dtype=torch.float32
        )
        self._event_site_tensor = torch.tensor(
            self.event_site_df.values,
            device=self.tune_config.device,
            dtype=torch.float32,
        )
        self._im_data_tensor = torch.tensor(
            self.im_df.values, device=self.tune_config.device, dtype=torch.float32
        )

    def __len__(self) -> int:
        return self.im_df.shape[0]

    def get_batch(self, indices: np.ndarray) -> BaseBatchData:
        """
        Get a batch of data for the given indices.

        indices: np.ndarray
            GM record indices
        """
        y = self._im_data_tensor[indices, :]

        site_data = self._site_data_tensor[self._index_to_site_ix[indices], :]
        source_data = self._source_data_tensor[self._index_to_event_ix[indices], :]
        event_site_data = self._event_site_tensor[indices, :]

        X = torch.cat(
            (site_data, source_data, event_site_data),
            dim=1,
        )

        return ObsBatchData(
            self._record_ids[indices],
            X,
            y,
        )


def get_similar_records(
    run_config: "nn_gmm.GMMRunConfig",
    fixed_inputs: dict,
    limits: dict,
    record_int_ids: np.ndarray = None,
):
    logging.info("Getting similar records for fixed inputs:")
    for k, v in fixed_inputs.items():
        if k in limits:
            logging.info(f"{k}: {v} ± {limits[k]}")
        else:
            logging.info(f"{k}: {v} (no limits defined)")

    with DuckIMDB(run_config.imdb_ffp, readonly=True) as imdb:
        site_df = imdb.get_site_df()
        event_df = imdb.get_event_df()
        rel_df = imdb.get_rel_df()

    # Valid sites
    for k in run_config.site_inputs:
        if k in fixed_inputs and k in limits:
            site_df = site_df.loc[
                site_df[k].between(
                    fixed_inputs[k] - limits[k][0],
                    fixed_inputs[k] + limits[k][1],
                )
            ]
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
                event_df = event_df.loc[
                    event_df[k].between(
                        fixed_inputs[k] - limits[k][0],
                        fixed_inputs[k] + limits[k][1],
                    )
                ]
            # Realisation property
            else:
                rel_df = rel_df.loc[
                    rel_df[k].between(
                        fixed_inputs[k] - limits[k][0],
                        fixed_inputs[k] + limits[k][1],
                    )
                ]
        else:
            logger.info(f"Skipping source input {k}, not in fixed inputs or limits")

    rel_df = rel_df.loc[rel_df.event_int_id.isin(event_df.index)]
    logger.info(
        f"Found {len(event_df)} valid events and {len(rel_df)} valid realisations"
    )

    comb = np.array(list(itertools.product(site_df.index, event_df.index)))
    if comb.shape[0] == 0:
        logger.warning("No valid site-event combinations found")
        return np.array([])
    site_event_int_ids = utils.get_site_event_int_id(comb[:, 0], comb[:, 1])
    with DuckIMDB(run_config.imdb_ffp, readonly=True) as imdb:
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

    with DuckIMDB(run_config.imdb_ffp, readonly=True) as imdb:
        record_info_df = imdb.get_record_info_df(
            events=event_df.event_id.values.astype(str),
            sites=site_df.site_id.values.astype(str),
        )

    # Site, Event and Realisation filtering
    record_mask = (
        record_info_df.site_int_id.isin(site_df.index)
        & record_info_df.event_int_id.isin(event_df.index)
        & record_info_df.rel_int_id.isin(rel_df.index)
    )
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
