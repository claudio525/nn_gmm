import os
import glob
import argparse
import multiprocessing as mp

import numpy as np
import pandas as pd


def process_fault(input_dir: str, fault_name: str):
    # Find all the IM csv files
    im_files = glob.glob(os.path.join(input_dir, fault_name, "*", "IM_calc", "*.csv"))

    # Get the number of IMs & stations in the IM csv files
    # Note: Assumes that all IM csv files for a specific fault have the
    # same number of IMs & stations across all realisations
    ref_df = pd.read_csv(im_files[0], index_col=0, engine="c", sep=",").sort_index()
    stations = ref_df.index.values.astype(str)
    ims = ref_df.columns.values.astype(str)
    ims = ims[ims != "component"]  # Drop the component column

    n_rels, n_stations = len(im_files), stations.shape[0]

    # Pre-allocate the 2D-array, shape: [n_station * n_realisations, n_ims]
    im_data = np.full((stations.size * n_rels, ims.size), np.nan)
    ids = []
    for ix, cur_im_file in enumerate(im_files):
        rel_id = os.path.basename(cur_im_file.split("_")[-1].split(".")[0])
        cur_df = pd.read_csv(cur_im_file, index_col=0, engine="c", sep=",").sort_index()

        assert np.all(cur_df.index.values.astype(str) == stations)
        assert np.all(np.isin(ims, cur_df.columns.values)), (
            f"The IMs are not matching across all IM csv files, "
            f"e.g. {os.path.basename(im_files[0])} and {os.path.basename(cur_im_file)}"
        )

        im_data[ix * n_stations : (ix + 1) * n_stations, :] = cur_df.loc[
            stations, ims
        ].values
        ids.append(np.char.add(f"{fault_name}_{rel_id}_", stations))

    # Sanity check
    assert ~np.any(np.isnan(im_data))

    # Create the dataframe
    im_df = pd.DataFrame(index=np.concatenate(ids), data=np.log(im_data), columns=ims)

    return fault_name, im_df


def main(input_dir: str, output_ffp: str, n_procs: int = 4):
    if os.path.isfile(output_ffp):
        print("The output file already exist, quitting!")
        exit()

    faults = np.asarray(next(os.walk(input_dir))[1])

    print("Retrieving IM data")
    if n_procs == 1:
        result = [process_fault(input_dir, cur_fault) for cur_fault in faults]
    else:
        with mp.Pool(processes=n_procs) as p:
            result = p.starmap(
                process_fault, [(input_dir, cur_fault) for cur_fault in faults]
            )

    # Write to db
    print(f"Writing to db")
    with pd.HDFStore(output_ffp, "w") as store:
        for fault_name, fault_im_data in result:
            store[fault_name] = fault_im_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "input_dir",
        type=str,
        help="Base path of the location of all IM csv, expects the following "
        "directory structure /fault_name/sim_name/IM_calc/sim_name.csv",
    )
    parser.add_argument("output_ffp", type=str, help="Ouput file path")
    parser.add_argument(
        "--n_procs", type=int, help="Number of processes to use", default=4
    )

    args = parser.parse_args()

    main(args.input_dir, args.output_ffp, n_procs=args.n_procs)
