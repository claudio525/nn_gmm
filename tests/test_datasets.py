import time
import os
from pathlib import Path
import pytest
from unittest.mock import Mock

import torch
import pandas as pd
import numpy as np

import nn_gmm as nng


device = "cpu"
if torch.cuda.is_available():
    device = "cuda"


wdata = Path(os.environ["wdata"])
sqlite_imdb_ffp = Path(wdata / "nn_gmm/20250606_CS200m_imdb.db")
duckdb_imdb_ffp = Path(wdata / "nn_gmm/20251009_CS200m_imdb.duckdb")


@pytest.mark.parametrize(
    "imdb_ffp,seed",
    [
        (sqlite_imdb_ffp, 42),
        (sqlite_imdb_ffp, 199),
        (sqlite_imdb_ffp, 2),
        (duckdb_imdb_ffp, 42),
        (duckdb_imdb_ffp, 199),
        (duckdb_imdb_ffp, 2),
    ],
)
def test_datasets(imdb_ffp: Path, seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)

    imdb_type = nng.IMDB if imdb_ffp.suffix == ".db" else nng.DuckIMDB
    with imdb_type(imdb_ffp, readonly=True) as imdb:
        site_df = imdb.get_site_df(add_nztm=True)
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

    record_info_df["sample_weight"] = 1.0

    site_inputs = ["vs30", "z1p0", "z2p5"]
    source_inputs = ["magnitude", "rake"]
    site_event_inputs = ["rrup", "rjb", "rx", "ry"]

    # mock_run_config = Mock()
    # mock_run_config.scale_ims = False
    # mock_run_config.device = device
    # mock_run_config.apply_im_weighting = False
    # mock_run_config.site_inputs = ["vs30", "z1p0", "z2p5"]
    # mock_run_config.source_inputs = ["magnitude", "rake"]
    # mock_run_config.site_event_inputs = ["rrup", "rjb", "rx", "ry"]
    # mock_run_config.using_loc_model = False
    # mock_run_config.loc_model_inputs = None

    dataset_1 = nng.data.IMDBDataset(
        imdb_ffp,
        record_int_ids,
        np.array(nng.constants.PSA_KEYS),
        site_df[site_inputs],
        source_df[source_inputs],
        site_event_df[site_event_inputs],
        record_info_df,
        device,
        scale_ims=True
    )

    dataset_2 = nng.data.OptimizedIMDBDataset(
        imdb_ffp,
        record_int_ids,
        np.array(nng.constants.PSA_KEYS),
        site_df[site_inputs],
        source_df[source_inputs],
        site_event_df[site_event_inputs],
        record_info_df,
        device,
        scale_ims=True
    )

    assert len(dataset_1) == len(dataset_2)

    # Compare the scale parameters
    scale_params_1 = dataset_1.im_scale_params
    scale_params_2 = dataset_2.im_scale_params
    for cur_key in scale_params_1.keys():
        pd.testing.assert_series_equal(
            scale_params_1[cur_key], scale_params_2[cur_key], check_names=True
        )
    
    # Compare IM values
    for ix in range(10):
        indices = np.random.choice(len(dataset_1), 4096, replace=False)

        start = time.time()
        batch_1 = dataset_1.get_batch(indices)
        print(f"Took: {time.time() - start} for original")
        start = time.time()
        batch_2 = dataset_2.get_batch(indices)
        print(f"Took: {time.time() - start} for optimized")

        np.testing.assert_array_equal(
            batch_1.X.numpy(force=True), batch_2.X.numpy(force=True)
        )
        np.testing.assert_array_equal(
            batch_1.y.numpy(force=True), batch_2.y.numpy(force=True)
        )
