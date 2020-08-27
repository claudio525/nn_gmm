"""Contains the code for computing the directivity features s and theta"""
from pathlib import Path
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
    return theta if theta < 90 else theta - 90

def get_hypo_seg_ix(seg_bounds, hypo_lon: float, hypo_lat: float):
    hypo_point = Point(hypo_lon, hypo_lat)
    return np.flatnonzero(
        [
            True if Polygon(cur_bounds).contains(hypo_point) else False
            for cur_bounds in seg_bounds
        ])[0]


def get_next_seg_ix(
    seg_bounds: List[List[float]],
    prev_seg_closest_loc: Tuple[float, float],
    prev_ix: int,
    site_point: Point,
    ix_direction: int = None,
):
    """Gets the next segment closer to the site for a multi-segment finite fault

    Parameters
    ----------
    seg_bounds: list of list of floats
        The lon, lat values for the corners of the different
        fault segments
    prev_seg_closest_loc: tuple of two floats
        The lon, lat values of the closest location to the site
        for the previous segment
    prev_ix: int
        Index (into seg_bounds) of the previous segment
    site_point: Point
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
    prev_seg_closest_point = Point(*prev_seg_closest_loc)

    prev_dist = prev_seg_closest_point.distance(site_point)
    if ix_direction is None:
        # Previous segment is not at the start/end of the fault
        if 0 < prev_ix < len(seg_bounds) - 1:
            next_seg_poly_1_dist = Polygon(seg_bounds[prev_ix - 1]).distance(
                prev_seg_closest_point
            )
            next_seg_poly_2_dist = Polygon(seg_bounds[prev_ix + 1]).distance(
                prev_seg_closest_point
            )

            ix_direction = -1 if next_seg_poly_1_dist < next_seg_poly_2_dist else +1
        # Previous segment is either at the start or end of the fault,
        # only one possible direction
        else:
            ix_direction = +1 if prev_ix == 0 else -1

    # Check if the next segment (based on ix_direction) would be closer
    # to the site
    if 0 <= prev_ix + ix_direction < len(seg_bounds):
        next_poly = Polygon(seg_bounds[prev_ix + ix_direction])

        if next_poly.distance(site_point) < prev_dist:
            return prev_ix + ix_direction, ix_direction

    # Current segment is the closest
    return None, None


def compute_theta_s(
    seg_bounds: List,
    strike_values: List[float],
    rake: float,
    hypo_loc: Tuple[float, float],
    site_locs: pd.DataFrame,
    verbose: bool = False,
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
    hypo_loc: tuple of two floats
        Lon, Lat of hypocentre
    site_locs: dataframe
        The site locations for which to compute theta and s
        Index has to be the site name, and requires
        columns ["lon", "lat"]
    verbose: bool, optional

    Returns
    -------
    """
    # Get the fault segment that contains the hypocentre
    hypo_seg_ix = get_hypo_seg_ix(seg_bounds, hypo_loc[0], hypo_loc[1])

    # Compute s and theta for each site
    theta_dict, s_dict = {}, {}
    for cur_site_name, cur_site_details in site_locs.iterrows():

        if verbose:
            print(
                f"Total s: {s_dict[cur_site_name]:.3f}, Averaged theta: {theta_dict[cur_site_name]:.1f}"
            )
            print(f"------------------------------------------------\n")

        cur_theta, cur_s = compute_site_theta_s(
            cur_site_name,
            cur_site_details.lon,
            cur_site_details.lat,
            seg_bounds,
            strike_values,
            rake,
            hypo_seg_ix,
            hypo_loc,
            verbose=verbose,
            debug=False,
        )
        theta_dict[cur_site_name], s_dict[cur_site_name] = cur_theta, cur_s

        result_df = pd.merge(
            pd.Series(s_dict),
            pd.Series(theta_dict),
            how="inner",
            left_on=True,
            right_on=True,
            validate="one_to_one",
        )
        assert result_df.shape[0] == len(s_dict)

        return result_df


def compute_site_theta_s(
    site_name: str,
    site_lon: float,
    site_lat: float,
    seg_bounds: List,
    strike_values: List[float],
    rake: float,
    hypo_seg_ix: int,
    hypo_loc: Tuple[float, float],
    verbose: bool = False,
    debug: bool = False,
):
    if verbose:
        print(f"Current site: {site_name}")
    debug_details = []

    site_lon, site_lat = site_lon, site_lat
    site_point = Point(site_lon, site_lat)
    s_values, theta_values = [], []

    # Compute s and theta for the hypocentre segment
    # Get the closest point for the segment (wrt. site)
    hypo_seg_poly = Polygon(seg_bounds[hypo_seg_ix])
    hypo_lon, hypo_lat = hypo_loc[0], hypo_loc[1]
    cur_seg_closest_loc = nearest_points(hypo_seg_poly, site_point)[0].coords[0]

    cur_strike = strike_values[hypo_seg_ix]
    cur_site_bearing = geo.ll_bearing(hypo_lon, hypo_lat, site_lon, site_lat)
    cur_rake_bearing = _compute_rake_bearing(cur_strike, rake)
    s_values.append(
        geo.get_distances(
            np.asarray([[hypo_lon, hypo_lat]]),
            cur_seg_closest_loc[0],
            cur_seg_closest_loc[1],
        )[0]
    )
    theta_values.append(_compute_theta(cur_rake_bearing, cur_site_bearing))
    debug_details.append((hypo_loc, cur_seg_closest_loc, cur_strike, cur_site_bearing, cur_rake_bearing))
    prev_seg_closest_loc = cur_seg_closest_loc

    if verbose:
        print(
            f"Hypocentre segment ({hypo_seg_ix}) - Strike: {strike_values[hypo_seg_ix]:.1f}, "
            f"Rake: {rake:.1f}, Rake bearing: {cur_rake_bearing:.1f}, \n"
            f"\tSite bearing {cur_site_bearing:.1f}, s: {s_values[-1]:.3f}, "
            f"theta: {theta_values[-1]:.1f}"
        )

    # Site is in hypocentre segment boundaries
    if hypo_seg_poly.contains(site_point):
        if verbose:
            print(f"------------------------------------------------\n")
        if debug:
            return theta_values[-1], s_values[-1], theta_values, s_values, debug_details
        return theta_values[-1], s_values[-1]

    # Iterate over the relevant segments
    cur_seg_ix, ix_dir = get_next_seg_ix(
        seg_bounds, prev_seg_closest_loc, hypo_seg_ix, site_point
    )
    while cur_seg_ix is not None:
        cur_strike = strike_values[cur_seg_ix]
        cur_rake_bearing = (
            cur_strike - rake if cur_strike - rake > 0 else 360 - (cur_strike - rake)
        )

        cur_poly = Polygon(seg_bounds[cur_seg_ix])
        cur_seg_closest_point = nearest_points(cur_poly, site_point)[0]
        cur_seg_closest_loc = cur_seg_closest_point.coords[0]
        cur_site_bearing = geo.ll_bearing(
            cur_seg_closest_loc[0], cur_seg_closest_loc[1], site_lon, site_lat,
        )

        # Closest point
        cur_s = geo.get_distances(
            np.asarray([[prev_seg_closest_loc[0], prev_seg_closest_loc[1]]]),
            cur_seg_closest_loc[0],
            cur_seg_closest_loc[1],
        )[0]
        cur_theta = _compute_theta(cur_rake_bearing, cur_site_bearing)

        if verbose:
            print(
                f"Segment {cur_seg_ix} - Strike: {cur_strike:.1f}, Rake: {rake:.1f}, Rake bearing: {cur_rake_bearing:.1f}, \n"
                f"\tSite bearing: {cur_site_bearing:.1f}, s: {cur_s:.3f}, theta: {cur_theta:.1f}"
            )

        s_values.append(cur_s)
        theta_values.append(cur_theta)
        debug_details.append((prev_seg_closest_loc, cur_seg_closest_loc, cur_strike, cur_site_bearing, cur_rake_bearing))

        prev_seg_closest_loc = cur_seg_closest_loc
        cur_seg_ix, ix_dir = get_next_seg_ix(
            seg_bounds,
            prev_seg_closest_loc,
            cur_seg_ix,
            site_point,
            ix_direction=ix_dir,
        )

    s, theta = np.sum(s_values), np.average(theta_values, weights=s_values)
    if debug:
        return theta, s, theta_values, s_values, debug_details

    return theta, s
