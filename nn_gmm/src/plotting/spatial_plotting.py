import pandas as pd
from pathlib import Path
import tempfile
from typing import Tuple, Union

import numpy as np
import pygmt
import xarray as xr
from scipy import interpolate


def gen_region_fig(
    title: str = None, region: Union[str, Tuple[float, float, float, float]] = "NZ"
):
    """Creates a basic figure for the specified region
    Todo: Extend this one day to include topo etc. (see plot_items)
    """
    frame_args = ["af", "xaf+lLongitude", "yaf+lLatitude"]
    if title is not None:
        frame_args.append(f'+t"{title}"')

    fig = pygmt.Figure()
    fig.basemap(region=region, projection="j172/1.5", frame=frame_args)

    # Plots the coast (sea & inland lakes/rivers)
    fig.coast(
        shorelines=["1/0.1p,black", "2/0.1p,black"],
        resolution="f",
        land="#666666",
        water="skyblue",
    )

    return fig


def plot_grid(
    fig: pygmt.Figure,
    grid: xr.DataArray,
    cmap: str,
    cmap_limits: Tuple[float, float, float],
    cmap_limit_colors: Tuple[str, str],
    cb_label: str = None,
    reverse_cmap: bool = False,
    log_cmap: bool = False,
):
    """
    Plots the given grid as a colourmap & contours
    Also adds a colour bar

    Parameters
    ----------
    fig: Figure
    grid: DataArray
        The data grid to plot
        Has to have the coordinates lat & lon (in that order),
        along with a data value
    cmap: string
        The "master" colourmap to use (see gmt documentation)
    cmap_limits: triplet of floats
        The min, max & step value for colour map
        Number of colours is therefore given by
        (cpt_limits[1] - cpt_limits[0]) / cpt_step
    cmap_limit_colors: pair of strings
        The colours to use for regions that are
        outside the specified colourmap limits
    reverse_cmap: bool, optional
        Reverses the order of the colours
    log_cmap: bool, optional
        Create a log10 based colourmap
        Expects the cmap_limits to be log10(z)
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)

        # Set the background & foreground colour for the colormap
        pygmt.config(
            COLOR_BACKGROUND=cmap_limit_colors[1], COLOR_FOREGROUND=cmap_limit_colors[0]
        )

        # Need two CPTs, otherwise the contours will be plotted every cb_step
        # And using "interval" directly in the contour call means that they don't
        # line up with the colour map
        cpt_ffp, cpt_ffp_ct = (
            str(tmp_dir / "cur_cpt_1.cpt"),
            str(tmp_dir / "cur_cpt_2.cpt"),
        )
        pygmt.makecpt(
            cmap=cmap,
            series=[cmap_limits[0], cmap_limits[1], cmap_limits[2]],
            output=cpt_ffp,
            reverse=reverse_cmap,
            log=log_cmap,
        )
        pygmt.makecpt(
            cmap=cmap,
            series=[cmap_limits[0], cmap_limits[1], cmap_limits[2] * 2],
            output=cpt_ffp_ct,
            reverse=reverse_cmap,
            log=log_cmap,
        )

        # Plot the grid
        fig.grdimage(
            grid, cmap=cpt_ffp, transparency=0, interpolation="c", nan_transparent=True
        )

        # Plot the contours
        fig.grdcontour(
            annotation="-",
            interval=cpt_ffp_ct,
            grid=grid,
            limit=[cmap_limits[0], cmap_limits[1]],
            pen="0.1p",
        )

        # Add a colorbar, with an annotated tick every second colour step,
        # and un-annotated tick with every other colour step
        cb_frame = [f"a+{cmap_limits[2] * 2:.3f}f+{cmap_limits[2]:.3f}"]
        if cb_label is not None:
            cb_frame.append(f'x+l"{cb_label}"')
        fig.colorbar(
            cmap=cpt_ffp, frame=cb_frame,
        )


def create_grid(
    data_df: pd.DataFrame,
    data_key: str,
    grid_spacing: str = "200e/200e",
    region: str = "NZ",
    interp_method: str = "CloughTorcher",
):
    """
    Creates a regular grid from the available unstructured data

    Parameters
    ----------
    data_df: DataFrame
        Unstructured data to be gridded
    grid_spacing: string
        Grid spacing to use, uses gmt gridding
        functionality, see "spacing" in
        (https://www.pygmt.org/latest/api/generated/pygmt.grdlandmask.html)

        Short summary of most relevant usage:
        For gridline every x (unit), use "{x}{unit}/{x}{unit}",
            where unit is one of metres (e), kilometres (k)
        To use a specific number of gridlines use "{x}+n/{x}+n",
            where x is the number of gridlines
    region: str or quadruplet of floats
        Region name or (xmin/xmax/ymin/ymax)

    Returns
    -------
    grid: DataArray
    """
    # Create the land/water mask
    land_mask = pygmt.grdlandmask(
        region=region, spacing=grid_spacing, maskvalues=[0, 1], resolution="f"
    )

    # Use land/water mask to create meshgrid
    x1, x2 = np.meshgrid(land_mask.lon.values, land_mask.lat.values)

    # Interpolate available data onto meshgrid
    if interp_method == "CloughTorcher":
        interp = interpolate.CloughTocher2DInterpolator(
            np.stack((data_df.lon.values, data_df.lat.values), axis=1),
            data_df[data_key].values,
        )
    elif interp_method == "nearest":
        interp = interpolate.NearestNDInterpolator(
            np.stack((data_df.lon.values, data_df.lat.values), axis=1),
            data_df[data_key].values,
        )
    elif interp_method == "linear":
        interp = interpolate.LinearNDInterpolator(
            np.stack((data_df.lon.values, data_df.lat.values), axis=1),
            data_df[data_key].values,
        )
    else:
        raise ValueError(
            "Invalid interpolation method specified, "
            "has to be one of [CloughTorcher, nearest, linear]"
        )

    grid_values = interp(x1, x2)

    # Create XArray grid
    grid = xr.DataArray(
        grid_values.reshape(land_mask.lat.size, land_mask.lon.size),
        dims=("lat", "lon"),
        coords={"lon": np.unique(x1), "lat": np.unique(x2)},
    )

    # Change water values to nan
    grid.values[~land_mask.astype(bool)] = np.nan

    return grid
