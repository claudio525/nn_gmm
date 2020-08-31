"""Contains the code for computing the directivity features s and theta"""
from pathlib import Path
import multiprocessing as mp
from collections import namedtuple
from typing import Tuple, List, Union, Dict

import pandas as pd
import numpy as np
from shapely.geometry import Point, Polygon
from shapely.ops import nearest_points

from qcore import geo


def _compute_rake_bearing(strike: float, rake: float):
    return strike - rake if strike - rake > 0 else 360 - (strike - rake)


def _compute_theta(rake_bearing: float, site_bearing: float):
    theta = np.abs(geo.angle_diff(rake_bearing, site_bearing))
    return theta if theta < 90 else 180 - theta


def get_hypo_seg_ix(seg_bounds, hypo_lon: float, hypo_lat: float):
    hypo_point = Point(hypo_lon, hypo_lat)
    contains_hypo = np.flatnonzero(
        [
            True if Polygon(cur_bounds).contains(hypo_point) else False
            for cur_bounds in seg_bounds
        ]
    )

    # If none of the segments contains the hypocentre
    # choose the closest segment
    if not np.any(contains_hypo):
        dist = [Polygon(cur_bounds).distance(hypo_point) for cur_bounds in seg_bounds]
        return np.argmin(dist)
    else:
        return contains_hypo[0]


def _process_site(
    site_lon: float,
    site_lat: float,
    seg_bounds: List,
    strike_values: List[float],
    rake: float,
    hypo_coords: Tuple[float, float],
):
    directivity_processor = FaultDirectivityProcessor(
        seg_bounds, strike_values, rake, hypo_coords, verbose=False
    )
    return directivity_processor.compute_site_theta_s(site_lon, site_lat)


def compute_theta_s(
    seg_bounds: List,
    strike_values: List[float],
    rake: float,
    hypo_coords: Tuple[float, float],
    site_coords: np.ndarray,
    verbose: bool = False,
    n_procs: int = 1,
):
    """Computes the directivity parameters theta and s

    Parameters
    ----------
    seg_bounds: list of list of floats
        The lon, lat values for the corners of the different
        fault segments
    strike_values: list of floats
        The strike values for the different fault segments
        Segment order has to be the same as seg_bounds
    rake: float
    hypo_coords: tuple of two floats
        Lon, Lat of hypocentre
    site_coords: numpy array of floats
        Coordinates for the sites
        Shape: [n_sites, 2]
    verbose: bool, optional
    n_procs: int, optional

    Returns
    -------
    """
    if n_procs == 1:
        # Compute s and theta for each site
        directivity_processor = FaultDirectivityProcessor(
            seg_bounds, strike_values, rake, hypo_coords, verbose=verbose
        )
        theta_values, s_values = [], []
        for cur_site_lon, cur_site_lat in site_coords:
            cur_theta, cur_s = directivity_processor.compute_site_theta_s(
                cur_site_lon, cur_site_lat,
            )
            theta_values.append(cur_theta)
            s_values.append(cur_s)
    else:
        with mp.Pool(processes=n_procs) as pool:
            results = pool.starmap(
                _process_site,
                [
                    (
                        cur_site_lon,
                        cur_site_lat,
                        seg_bounds,
                        strike_values,
                        rake,
                        hypo_coords,
                    )
                    for cur_site_lon, cur_site_lat in site_coords
                ],
            )
        theta_values = [result[0] for result in results]
        s_values = [result[1] for result in results]

    return theta_values, s_values

class Location:
    def __init__(self, lon: float, lat: float):
        self.lon, self.lat = lon, lat
        self.point = Point(lon, lat)

    def __str__(self):
        return self.point.wkt


class Segment:
    def __init__(self, bounds: np.ndarray, strike: float):
        self.strike = strike
        self.bounds = bounds

        self.poly = Polygon(self.bounds)


class FaultDirectivityProcessor:

    SegmentResult = namedtuple(
        "SegmentResult",
        [
            "s",
            "theta",
            "next_seg_ix",
            "ix_dir",
            "start_loc",
            "end_loc",
            "segment",
            "rake_bearing",
            "site_bearing",
        ],
    )

    def __init__(
        self,
        seg_bounds: List,
        strike_values: List[float],
        rake: float,
        hypo_coords: Tuple[float, float],
        verbose: bool = False,
    ):
        self.segments = [
            Segment(np.asarray(cur_bounds), cur_strike)
            for cur_bounds, cur_strike in zip(seg_bounds, strike_values)
        ]

        self.rake = rake
        self.hypo = Location(*hypo_coords)
        self.hypo_seg_ix = get_hypo_seg_ix(seg_bounds, hypo_coords[0], hypo_coords[1])

        self.verbose = verbose

    def _process_seg(
        self,
        segment: Segment,
        start_loc: Location,
        seg_ix: int,
        site_loc: Location,
        ix_dir: int = None,
    ):
        assert segment.poly.distance(start_loc.point) < 1e-03

        rake_bearing = (
            segment.strike - self.rake
            if segment.strike - self.rake > 0
            else 360 - (segment.strike - self.rake)
        )

        # Find the closest point for the current segment to the site
        end_loc = Location(*nearest_points(segment.poly, site_loc.point)[0].coords[0])

        # If the current segment is not the closest, then the endpoint
        # has to be on the boundary of the next segment
        next_seg_ix, ix_dir = self.get_next_seg_ix(
            end_loc, seg_ix, site_loc, ix_direction=ix_dir
        )
        if next_seg_ix is not None:
            end_loc = Location(
                *nearest_points(self.segments[next_seg_ix].poly, end_loc.point)[
                    0
                ].coords[0]
            )

        site_bearing = geo.ll_bearing(
            start_loc.lon, start_loc.lat, site_loc.lon, site_loc.lat,
        )

        # Compute s for the current segment
        s = geo.get_distances(
            np.asarray([[start_loc.lon, start_loc.lat]]), end_loc.lon, end_loc.lat,
        )[0]
        theta = _compute_theta(rake_bearing, site_bearing)

        if self.verbose:
            print(
                f"Segment {seg_ix} - Strike: {segment.strike:.1f}, Rake: {self.rake:.1f}, Rake bearing: {rake_bearing:.1f}, \n"
                f"\tSite bearing: {site_bearing:.1f}, s: {s:.3f}, theta: {theta:.1f}"
            )

        return self.SegmentResult(
            s,
            theta,
            next_seg_ix,
            ix_dir,
            start_loc,
            end_loc,
            segment,
            rake_bearing,
            site_bearing,
        )

    def get_next_seg_ix(
        self,
        prev_end_loc: Location,
        prev_ix: int,
        site_loc: Location,
        ix_direction: int = None,
    ):
        """Gets the next segment closer to the site for a multi-segment finite fault

        Parameters
        ----------
        prev_ix: int
            Index (into seg_bounds) of the previous segment
        site_loc: Point
            Site of interest
        ix_direction: int, optional
            The direction of the segments iteration, can
            either be +1 or -1
            If None then the direction is determined
            by checking both


        Returns
        -------
        ix: int
            Index of the next segment
        ix_direction: int
            If ix_direction was specified just returns that,
            otherwise returns the direction of the next closest segment
        """
        prev_dist = prev_end_loc.point.distance(site_loc.point)
        if ix_direction is None:
            # Previous segment is not at the start/end of the fault
            if 0 < prev_ix < len(self.segments) - 1:
                next_seg_poly_1_dist = self.segments[prev_ix - 1].poly.distance(
                    prev_end_loc.point
                )
                next_seg_poly_2_dist = self.segments[prev_ix + 1].poly.distance(
                    prev_end_loc.point
                )

                ix_direction = -1 if next_seg_poly_1_dist < next_seg_poly_2_dist else +1
            # Previous segment is either at the start or end of the fault,
            # only one possible direction
            else:
                ix_direction = +1 if prev_ix == 0 else -1

        # Check if the next segment (based on ix_direction) would be closer
        # to the site
        if 0 <= prev_ix + ix_direction < len(self.segments):
            if (
                self.segments[prev_ix + ix_direction].poly.distance(site_loc.point)
                < prev_dist
            ):
                return prev_ix + ix_direction, ix_direction

        # Current segment is the closest
        return None, None

    def compute_site_theta_s(
        self,
        site_lon: float,
        site_lat: float,
        debug: bool = False,
        site_name: str = None,
    ):
        if self.verbose and site_name is not None:
            print(f"Current site: {site_name}")

        site_loc = Location(site_lon, site_lat)
        seg_results = []

        seg_result = self._process_seg(
            self.segments[self.hypo_seg_ix], self.hypo, self.hypo_seg_ix, site_loc, None
        )
        seg_results.append(seg_result)

        cur_seg_ix, ix_dir = seg_result.next_seg_ix, seg_result.ix_dir
        prev_seg_result = seg_result

        # Site is in hypocentre segment boundaries
        if self.segments[self.hypo_seg_ix].poly.contains(site_loc.point):
            if self.verbose:
                print(f"------------------------------------------------\n")
            if debug:
                return seg_result.theta, seg_result.s, seg_results
            return seg_result.theta, seg_result.s

        # Process the relevant segments
        while cur_seg_ix is not None:
            seg_result = self._process_seg(
                self.segments[cur_seg_ix],
                prev_seg_result.end_loc,
                cur_seg_ix,
                site_loc,
                ix_dir,
            )
            seg_results.append(seg_result)

            prev_seg_result, cur_seg_ix = seg_result, seg_result.next_seg_ix

        # Adding a tiny amount in case an s-value is zero, as it can't be zero
        # when taking the weighted average
        s_values = [cur_result.s + 1e-10 for cur_result in seg_results]
        theta_values = [cur_result.theta for cur_result in seg_results]

        s, theta = np.sum(s_values), np.average(theta_values, weights=s_values)
        if debug:
            return theta, s, seg_results

        return theta, s
