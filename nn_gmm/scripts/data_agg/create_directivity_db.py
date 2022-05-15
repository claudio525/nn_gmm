"""Script for creating a site-source db that contains
"""
import time
import multiprocessing as mp
import argparse
from pathlib import Path
from typing import Dict, List

import h5py
import pandas as pd
import numpy as np
import typer

import nn_gmm
from nn_gmm import console
from IM_calculation.source_site_dist import src_site_dist
from qcore import srf, geo


def _get_stations(imdb_ffp: Path, fault: str):
    with pd.HDFStore(str(imdb_ffp), "r") as imdb:
        cur_df = imdb[fault]

        stations = np.unique(
            np.stack(np.char.split(cur_df.index.values.astype(str), "_"))[:, 2]
        )

    return fault, stations


def _process_realisation(
    realisation: str,
    source: str,
    srf_points: np.ndarray,
    header: List[Dict],
    sites_df: pd.DataFrame,
    srf_info_dir: Path,
):
    cur_id_prefix = (
        f"{realisation}_" if realisation != source else f"{source}_{source}_"
    )

    # Find the srf info file
    srf_info_ffp = next(srf_info_dir.glob(f"{realisation}.info"))
    with h5py.File(srf_info_ffp, "r") as f:
        srf_info = dict(f.attrs)

    # Compute S and Theta
    theta_values, s_values = nn_gmm.compute_theta_s(
        srf_info["corners"],
        srf_info["strike"],
        srf_info["rake"],
        (srf_info["hlon"], srf_info["hlat"]),
        sites_df.loc[:, ["lon", "lat"]].values,
        n_procs=1,
    )
    result_df = pd.DataFrame(
        index=np.char.add(cur_id_prefix, sites_df.index.values.astype(str)),
        columns=["theta_custom", "S_custom"],
        data=np.stack([theta_values, s_values], axis=1),
    )

    # Compute U & T
    T, U = src_site_dist.calc_rx_ry_GC2(
        srf_points,
        header,
        sites_df.loc[:, ["lon", "lat"]].values,
        hypocentre_origin=True,
    )
    result_df["T"] = T
    result_df["U"] = U

    # Compute
    # S = Horizontal length of the rupture travel between  the site and the origin
    # D = The effective rupture travel width, measured from the hypocentre to the
    # shallowest depth of the rupture plane
    result_df["S_bayless"] = result_df.U
    if srf_points.size > 1:
        # Get the start and end coordinates of the trace
        trace_start = srf_points[0, :2]

        # srf points are per plane and then vary along strike, then depth per plane
        trace_end_ix = (
            (
                header[-1]["nstrike"] * header[-1]["ndip"]  # Start point of the last plane
            )
            - header[-1]["nstrike"]  # Add number of strike points
            + 1
        )
        trace_end = srf_points[-trace_end_ix, :2]

        trace_T, trace_U = src_site_dist.calc_rx_ry_GC2(
            srf_points,
            header,
            np.stack((trace_start, trace_end), axis=0),
            hypocentre_origin=True,
        )

        result_df.loc[result_df.U > trace_U.max(), "S_bayless"] = trace_U.max()
        result_df.loc[result_df.U < trace_U.min(), "S_bayless"] = trace_U.min()

    # Point-source
    else:
        result_df["S_bayless"] = 0

    result_df["D_bayless"] = np.max([cur_plane["dhyp"] for cur_plane in header])
    return result_df


def main(
    srf_data_dir: Path = typer.Argument(
        ..., help="Path to the srf data directory (one srf per source)"
    ),
    srf_info_dir: Path = typer.Argument(..., help="Path to the srf info data directory (one per realisation)"),
    srf_header_ffp: Path = typer.Argument(..., help="Path to the srf header data"),
    imdb_ffp: Path = typer.Argument(..., help="Path to the corresponding IMDB"),
    sites_ffp: Path = typer.Argument(..., help="Path to the sites csv"),
    output_ffp: Path = typer.Argument(..., help="Path of the output db"),
    n_procs: int = typer.Option(default=8, help="Number of processes to use"),
):
    # Load site locations
    sites_df = pd.read_csv(sites_ffp, index_col="station")

    # Create fault - station lookup
    print("Creating fault - station lookup")
    with pd.HDFStore(str(imdb_ffp), "r") as imdb:
        faults = np.asarray([fault.lstrip("/") for fault in list(imdb.keys())])

    with mp.Pool(processes=n_procs) as pool:
        results = pool.starmap(
            _get_stations, [(imdb_ffp, cur_fault) for cur_fault in faults]
        )
    fault_station_lookup = {fault: stations for fault, stations in results}

    # Load the srf points & headers
    srf_ffps = list(srf_data_dir.glob("*.srf"))
    srf_faults = np.asarray([cur_ffp.stem for cur_ffp in srf_ffps])
    srf_headers = pd.read_pickle(srf_header_ffp)

    # Check if there are any faults without an srf file
    srf_missing = faults[~nn_gmm.pandas_isin(faults, srf_faults)]
    if srf_missing.size > 0:
        console.print(
            f"[orange]The following faults are missing srf files: {srf_missing}[/]"
        )

    console.print("Processing faults")
    with pd.HDFStore(str(output_ffp), "a") as store:
        for ix, (cur_source, cur_source_ffp) in enumerate(zip(srf_faults, srf_ffps)):
            console.print(f"Processing {cur_source}, {ix+1}/{len(srf_faults)}")

            if f"/{cur_source}" in store.keys():
                console.print(f"An entry for source {cur_source} already exists. Skipping")
                continue

            if cur_source not in fault_station_lookup.keys():
                console.print(f"[orange1]No sites for fault {cur_source}. Skipping![/]")
                continue

            # Get the relevant sites and srf data
            cur_sites_df = sites_df.loc[fault_station_lookup[cur_source]]
            cur_srf_points = srf.read_srf_points(str(cur_source_ffp))
            cur_headers = {
                cur_rel: cur_header
                for cur_rel, cur_header in srf_headers.items()
                if cur_rel == cur_source or cur_rel.startswith(f"{cur_source}_")
            }

            # Process the realisation for the current fault
            if n_procs == 1:
                results = []
                for cur_rel, cur_header in cur_headers.items():
                    cur_result = _process_realisation(
                        cur_rel,
                        cur_source,
                        cur_srf_points.astype(np.float32),
                        cur_header,
                        cur_sites_df,
                        srf_info_dir / cur_source,
                    )
                    results.append(cur_result)
            else:
                with mp.Pool(n_procs) as p:
                    results = p.starmap(
                        _process_realisation,
                        [
                            (
                                cur_rel,
                                cur_source,
                                cur_srf_points.astype(np.float32),
                                cur_header,
                                cur_sites_df,
                                srf_info_dir / cur_source
                            )
                            for cur_rel, cur_header in cur_headers.items()
                        ],
                    )

            # Write the results for the current fault
            results = [result for result in results if result is not None]
            if len(results) > 0:
                store[cur_source] = pd.concat(results, axis=0)


if __name__ == "__main__":
    typer.run(main)

    # parser = argparse.ArgumentParser()
    # parser.add_argument("sources_dir", type=str, help="Path to the sources directory")
    # parser.add_argument("imdb_ffp", type=str, help="")
    # parser.add_argument("sites_ffp", type=str, help="Path to the site csv")
    # parser.add_argument("output_ffp", type=str, help="Path of the output db")
    # parser.add_argument(
    #     "--n_procs", type=int, help="Number of processes to use", default=8
    # )
    #
    # args = parser.parse_args()
    #
    # main(
    #     Path(args.sources_dir),
    #     Path(args.imdb_ffp),
    #     Path(args.sites_ffp),
    #     Path(args.output_ffp),
    #     n_procs=args.n_procs,
    # )
