"""Script for creating a site-source db that contains
site-source features such as rupture directivity (theta & s)
"""
import time
import multiprocessing as mp
import argparse
from pathlib import Path
from typing import Dict

import h5py
import pandas as pd
import numpy as np

import nn_gmm


def _process_realisation(
    srf_info_ffp: Path, fault_station_lookup: Dict, sites_df: pd.DataFrame
):
    # Get the fault and realisation name, some special handling for validation events/"realisation"
    fault = srf_info_ffp.name.split("_")[0] if "_" in srf_info_ffp.name else srf_info_ffp.name.split(".")[0]
    realisation = srf_info_ffp.name.split(".")[0]
    realisation = realisation if fault != realisation else f"{fault}_{realisation}"

    if fault not in fault_station_lookup.keys():
        print(f"Fault {fault} is not in the IMDB, skipping")
        return None
    stations = fault_station_lookup[fault]

    print(f"Processing realisation {realisation}")
    with h5py.File(srf_info_ffp, "r") as f:
        srf_info = dict(f.attrs)

    theta_values, s_values = nn_gmm.compute_theta_s(
        srf_info["corners"],
        srf_info["strike"],
        srf_info["rake"],
        (srf_info["hlon"], srf_info["hlat"]),
        sites_df.loc[stations, ["lon", "lat"]].values,
        n_procs=1,
    )

    result_df = sites_df.loc[stations, ["lon", "lat"]].copy()
    result_df["theta"] = theta_values
    result_df["s"] = s_values

    result_df.set_index(
        np.char.add(realisation + "_", result_df.index.values.astype(str)), inplace=True
    )

    return result_df


def _get_stations(imdb_ffp: Path, fault: str):
    with pd.HDFStore(imdb_ffp, "r") as imdb:
        cur_df = imdb[fault]

        stations = np.unique(
            np.stack(np.char.split(cur_df.index.values.astype(str), "_"))[:, 2]
        )

    return fault, stations


def main(
    sources_dir: Path,
    imdb_ffp: Path,
    sites_ffp: Path,
    output_ffp: Path,
    n_procs: int = 8,
):
    # Load site locations
    sites_df = pd.read_csv(sites_ffp, index_col="station")

    # Create fault - station lookup
    print("Creating fault - station lookup")
    with pd.HDFStore(imdb_ffp, "r") as imdb:
        faults = [fault.lstrip("/") for fault in list(imdb.keys())]

    with mp.Pool(processes=n_procs) as pool:
        results = pool.starmap(
            _get_stations, [(imdb_ffp, cur_fault) for cur_fault in faults]
        )
    fault_station_lookup = {fault: stations for fault, stations in results}

    print("Processing faults")
    fault_dirs = list(sources_dir.iterdir())
    with pd.HDFStore(output_ffp, "w") as store:
        for ix, cur_fault_ffp in enumerate(fault_dirs):
            print(f"Processing {cur_fault_ffp.name}, {ix+1}/{len(fault_dirs)}")
            fault_srf_info_ffps = cur_fault_ffp.glob("Srf/*.info")
            if n_procs == 1:
                results = []
                for cur_srf_ffp in fault_srf_info_ffps:
                    results.append(
                        _process_realisation(cur_srf_ffp, fault_station_lookup, sites_df)
                    )
            else:
                with mp.Pool(processes=n_procs) as pool:
                    results = pool.starmap(
                        _process_realisation,
                        [
                            (cur_srf_ffp, fault_station_lookup, sites_df)
                            for cur_srf_ffp in fault_srf_info_ffps
                        ],
                    )

            # Write the results for the current fault
            results = [result for result in results if result is not None]
            if len(results) > 0:
                store[cur_fault_ffp.name] = pd.concat(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("sources_dir", type=str, help="Path to the sources directory")
    parser.add_argument("imdb_ffp", type=str, help="Path to the corresponding IMDB")
    parser.add_argument("sites_ffp", type=str, help="Path to the site csv")
    parser.add_argument("output_ffp", type=str, help="Path of the output db")
    parser.add_argument(
        "--n_procs", type=int, help="Number of processes to use", default=8
    )

    args = parser.parse_args()

    main(
        Path(args.sources_dir),
        Path(args.imdb_ffp),
        Path(args.sites_ffp),
        Path(args.output_ffp),
        n_procs=args.n_procs,
    )
