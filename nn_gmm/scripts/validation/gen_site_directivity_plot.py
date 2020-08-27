import argparse
import multiprocessing as mp
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")

import nn_gmm
from qcore import srf, geo


def gen_plot(
    site_lon: float,
    site_lat: float,
    srf_ffp: Path,
    srf_info_ffp: Path,
    output: Path,
    site_name: str = "Site",
    vec_dist: float = 10,
):
    with h5py.File(srf_info_ffp, "r") as f:
        srf_info = dict(f.attrs)

    seg_bounds = srf.get_bounds(str(srf_ffp))
    srf_points = srf.read_srf_points(str(srf_ffp))  # Lon, lat, depth
    hypo_lon, hypo_lat = srf_info["hlon"], srf_info["hlat"]

    hypo_seg_ix = nn_gmm.get_hypo_seg_ix(seg_bounds, hypo_lon, hypo_lat)

    theta, s, theta_values, s_values, debug_details = nn_gmm.compute_site_theta_s(
        site_name,
        site_lon,
        site_lat,
        seg_bounds,
        srf_info["strike"],
        srf_info["rake"],
        hypo_seg_ix,
        (hypo_lon, hypo_lat),
        verbose=True,
        debug=True,
    )

    fig = plt.figure(figsize=(21, 13.5), dpi=144)
    plt.scatter(
        srf_points[:, 0][::2],
        srf_points[:, 1][::2],
        c=srf_points[:, 2][::2],
        label="srf points",
        s=1.0,
    )
    plt.scatter(site_lon, site_lat, label="Site", marker="v", c="k")
    plt.annotate(site_name, (site_lon, site_lat))
    plt.scatter(
        srf_info["hlon"],
        srf_info["hlat"],
        label="Hypocentre",
        marker="o",
        c="k",
        s=50.0,
    )

    # Plot the segment boundaries
    for cur_bounds in seg_bounds:
        cur_bounds = np.asarray(cur_bounds)
        plt.plot(cur_bounds[:, 0], cur_bounds[:, 1], c="g")
        plt.plot(
            [cur_bounds[-1, 0], cur_bounds[0, 0]],
            [cur_bounds[-1, 1], cur_bounds[0, 1]],
            c="g",
        )

    dp_fn = lambda values: [f"{val:.1f}" for val in values]
    for (
        ix,
        (
            (prev_lon, prev_lat),
            (cur_lon, cur_lat),
            cur_strike,
            cur_site_bearing,
            cur_rake_bearing,
        ),
    ) in enumerate(debug_details):
        _plot_vectors(
            prev_lon,
            prev_lat,
            cur_strike,
            cur_site_bearing,
            cur_rake_bearing,
            vec_dist,
            labels=ix == 0,
        )

        plt.plot(
            [prev_lon, cur_lon],
            [prev_lat, cur_lat],
            c="darkred",
            marker="x",
            linewidth=1.0,
            label="s" if ix==0 else None,
        )

    plt.title(
        f"{srf_ffp.name.split('.')[0]} - {site_name} - Theta: {theta:.1f}, "
        f"s: {s:.1f}\nTheta values: {dp_fn(theta_values)}\ns values: {dp_fn(s_values)}"
    )

    plt.legend()
    if output.is_dir():
        plt.savefig(output / f"{site_name}.png")
    else:
        plt.savefig(output)


def _plot_vectors(
    lon: float,
    lat: float,
    strike: float,
    site_bearing: float,
    rake_bearing: float,
    vec_dist: float,
    labels: bool = True,
):
    # Plot North
    north_lat, north_lon = geo.ll_shift(lat, lon, vec_dist, 0)
    plt.plot([lon, north_lon], [lat, north_lat], c="k", linestyle="-", linewidth=1.0)

    # Plot South
    north_lat, north_lon = geo.ll_shift(lat, lon, vec_dist, 180)
    plt.plot([lon, north_lon], [lat, north_lat], c="k", linestyle="-", linewidth=1.0)

    # Plot East
    north_lat, north_lon = geo.ll_shift(lat, lon, vec_dist, 90)
    plt.plot([lon, north_lon], [lat, north_lat], c="k", linestyle="-", linewidth=1.0)

    # Plot West
    north_lat, north_lon = geo.ll_shift(lat, lon, vec_dist, 270)
    plt.plot([lon, north_lon], [lat, north_lat], c="k", linestyle="-", linewidth=1.0)

    # Plot strike
    strike_lat, strike_lon = geo.ll_shift(lat, lon, vec_dist, strike)
    plt.plot(
        [lon, strike_lon],
        [lat, strike_lat],
        label="strike" if labels else None,
        c="b",
        linestyle="-",
        linewidth=1.0,
    )

    # Plot rake bearing
    rake_lat, rake_lon = geo.ll_shift(lat, lon, vec_dist, rake_bearing)
    plt.plot(
        [lon, rake_lon],
        [lat, rake_lat],
        label="Rake bearing" if labels else None,
        c="r",
        linestyle="-",
        linewidth=1.0,
    )
    rake_lat, rake_lon = geo.ll_shift(
        lat,
        lon,
        vec_dist,
        rake_bearing + 180 if rake_bearing + 180 < 360 else rake_bearing - 180,
    )
    plt.plot([lon, rake_lon], [lat, rake_lat], c="r", linestyle="-", linewidth=1.0)

    # Plot site bearing
    site_lat, site_lon = geo.ll_shift(lat, lon, vec_dist, site_bearing)
    plt.plot(
        [lon, site_lon],
        [lat, site_lat],
        label="Site bearing" if labels else None,
        c="m",
        linestyle="-",
        linewidth=1.0,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("srf_ffp", type=str, help="Path to the srf file")
    parser.add_argument("srf_info_ffp", type=str, help="Path to the srf info file")
    parser.add_argument(
        "output",
        type=str,
        help="File path for the outputs. Directory if site_ffp is specified,"
        " otherwise a file path",
    )

    parser.add_argument(
        "--site_ffp",
        type=str,
        help="Path to a file specifying the sites to \
    create plots for. Requires columns [name, lon, lat]",
        default=None,
    )

    parser.add_argument("--site_lon", type=float, help="Longitute of the site")
    parser.add_argument("--site_lat", type=float, help="Latitude of the site")
    parser.add_argument(
        "--site_name", type=str, help="Name of the site", default="Site"
    )

    parser.add_argument(
        "--vector_distance",
        type=float,
        help="The length of the vectors in km",
        default=10,
    )

    args = parser.parse_args()

    assert args.site_ffp is not None or (
        args.site_lon is not None and args.site_lat is not None
    )

    if args.site_ffp is None:
        gen_plot(
            args.site_lon,
            args.site_lat,
            Path(args.srf_ffp),
            Path(args.srf_info_ffp),
            Path(args.output),
            site_name=args.site_name,
            vec_dist=args.vector_distance,
        )
    else:
        df = pd.read_csv(args.site_ffp, index_col="name")

        with mp.Pool(8) as pool:
            pool.starmap(
                gen_plot,
                [
                    (
                        site_details.lon,
                        site_details.lat,
                        Path(args.srf_ffp),
                        Path(args.srf_info_ffp),
                        Path(args.output),
                        site_name,
                        args.vector_distance,
                    )
                    for site_name, site_details in df.iterrows()
                ],
            )
