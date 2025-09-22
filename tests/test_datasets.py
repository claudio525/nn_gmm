import time
import os
from pathlib import Path
import pytest
from unittest.mock import Mock

import torch
import numpy as np

import nn_gmm as nng


device = "cpu"
if torch.cuda.is_available():
    device = "cuda"


wdata = Path(os.environ["wdata"])
imdb_ffp = Path(wdata / "nn_gmm/20250606_CS200m_imdb.db")



@pytest.mark.parametrize("imdb_ffp,seed", [(imdb_ffp, 42), (imdb_ffp, 199), (imdb_ffp, 2)])
def test_datasets(imdb_ffp: Path, seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)

    with nng.IMDB(imdb_ffp, readonly=True) as imdb:
        site_df = imdb.get_site_df()
        event_df = imdb.get_event_df()

        sites = np.random.choice(
            site_df.site_id.values.astype(str), size=1000, replace=False
        )
        events = np.random.choice(
            event_df.event_id.values.astype(str), size=50, replace=False
        )

        # record_info_df = imdb.get_record_info_df()
        record_info_df = imdb.get_record_info_df(events=events, sites=sites)
        record_info_df["site_event_int_id"] = nng.utils.get_site_event_int_id(
            record_info_df.site_int_id.values, record_info_df.event_int_id.values
        )
        # record_int_ids = record_info_df.loc[record_info_df.event_id.isin(events) & record_info_df.site_id.isin(sites)].index.values
        record_int_ids = record_info_df.index.values

        source_df = imdb.get_rel_df(events)
        site_event_df = imdb.get_site_event_df(
            site_event_int_ids=record_info_df.site_event_int_id.values
        )

    source_features = ["magnitude", "rake"]
    site_features = ["vs30", "z1p0", "z2p5"]
    site_event_features = ["rrup", "rjb", "rx", "ry"]
    record_info_df["sample_weight"] = 1.0

    mock_run_config = Mock()
    mock_run_config.scale_ims = False
    mock_run_config.device = device
    # mock_run_config.seed = seed

    dataset_1 = nng.data.IMDBDataset(
        imdb_ffp,
        record_int_ids,
        np.array(nng.constants.PSA_KEYS),
        site_df[site_features],
        source_df[source_features],
        site_event_df[site_event_features],
        record_info_df,
        mock_run_config,
        is_train=False,
    )

    dataset_2 = nng.data.OptimizedIMDBDataset(
        imdb_ffp,
        record_int_ids,
        np.array(nng.constants.PSA_KEYS),
        site_df[site_features],
        source_df[source_features],
        site_event_df[site_event_features],
        record_info_df,
        mock_run_config,
        is_train=False,        
    )

    assert len(dataset_1) == len(dataset_2)

    for ix in range(10):
        indices = np.random.choice(len(dataset_1), 4096, replace=False)

        start = time.time()
        batch_1 = dataset_1.get_batch(indices)
        print(f"Took: {time.time() - start} for original")
        start = time.time()
        batch_2 = dataset_2.get_batch(indices)
        print(f"Took: {time.time() - start} for optimized")

        np.testing.assert_array_equal(batch_1.X.numpy(force=True), batch_2.X.numpy(force=True))
        np.testing.assert_array_equal(batch_1.y.numpy(force=True), batch_2.y.numpy(force=True))