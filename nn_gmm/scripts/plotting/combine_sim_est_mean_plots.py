#!/usr/bin/env python3
import glob
import argparse
import multiprocessing as mp
from pathlib import Path

import nn_gmm


def main(input_dir: Path, n_procs: int = 4):
    sim_ffps = sorted(input_dir.glob("*sim*.png"))
    est_mean_ffps = sorted(input_dir.glob("*est_mean*.png"))

    assert len(sim_ffps) == len(est_mean_ffps)

    with mp.Pool(processes=n_procs) as pool:
        pool.starmap(
            nn_gmm.combine_imgs,
            [
                (
                    sim_ffp,
                    est_mean_ffp,
                    input_dir / str(sim_ffp).replace("sim", f"sim_vs_est_mean"),
                )
                for sim_ffp, est_mean_ffp in zip(sim_ffps, est_mean_ffps)
            ],
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=str)
    parser.add_argument(
        "--n_procs", type=int, help="Number of processes to use", default=16
    )
    args = parser.parse_args()

    main(Path(args.input_dir), n_procs=args.n_procs)
