import gc
import tempfile
from typing import Tuple, Iterable, Callable, Dict, List, Any, Union
from pathlib import Path
from collections import namedtuple

import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import matplotlib

import empirical.util.classdef as classdef
import empirical.util.empirical_factory as emp_factory
from visualization.gmt.plotting import plot_multiple, plot_single

from nn_gmm.src.model import GMM
from nn_gmm.src.utils import get_station_from_id, get_station_lookup, to_path, to_list
from nn_gmm.src import data
from nn_gmm.src.eval import get_realisation_residuals
from .plotting_funcs import *

DataSpec = namedtuple("DataSpec", ["name", "fn", "fn_name"])


class DataExpPlotGen:
    """Computes plots for feature exploration"""

    def __init__(self, data_dir: Path, output_dir: Path, data_names: List[str] = None):
        self.output_dir = output_dir
        self.data_dir = data_dir

        self._data_df = None

        if data_names is not None:
            self._load_data(data_names)

    def _load_data(self, data_names: Union[str, List[str]]):
        data_names = [data_names] if isinstance(data_names, str) else data_names

        ds = data.load_dataset(
            [self.data_dir],
            data.load_feature_details(self.data_dir),
            int(1e6),
            shuffle_buffer=None,
            block_size=1024,
        )

        dfs = []
        for cur_batch in ds.as_numpy_iterator():
            cur_df = pd.DataFrame.from_dict(
                {key: cur_batch[key] for key in data_names + ["id"]}
            )
            cur_df["id"] = cur_df.id.str.decode("UTF-8")
            cur_df.set_index("id", inplace=True)

            dfs.append(cur_df)

        df = pd.concat(dfs)

        self._data_df = (
            df
            if self._data_df is None
            else self._data_df.merge(df, how="inner", left_index=True, right_index=True)
        )

    def _get_data_df(self, data_names: Union[str, List[str], np.ndarray]):
        data_names = np.asarray(
            [data_names] if isinstance(data_names, str) else data_names
        )

        if self._data_df is None:
            self._load_data(list(data_names))

        mask = np.isin(data_names, self._data_df.columns.astype(str))
        if np.any(~mask):
            self._load_data(list(data_names[~mask]))

        return self._data_df.loc[:, data_names]

    def _get_data_from_specs(self, data_spec: DataSpec) -> Tuple[np.ndarray, str]:
        name, trans_fn, fn_name = data_spec if isinstance(data_spec, DataSpec) else (data_spec, None, None)
        data_df = self._get_data_df(name)

        values = data_df[name].values
        if trans_fn is not None:
            values = trans_fn(values)

        if fn_name is not None:
            name = f"{fn_name}_{name.replace('.', 'p')}"

        return values, name

    def gen_hist_plots(
        self,
        data_specs: Union[str, DataSpec, List[Union[str, DataSpec]]],
        n_bins: int = 25,
    ):
        """
        Generates a histogram for the specified data type

        Parameters
        ----------
        data_specs: str or Dataspec, or list of
            Specifies the data for which to create the histogram
            Either just the name of the data or if a transformation should
            be applied a tuple of the format (name, fn, fn_name)
        n_bins: int
        """
        data_specs = data_specs if isinstance(data_specs, list) else [data_specs]
        for cur_data_spec in data_specs:
            fig = plt.figure(figsize=(18, 13.5))

            values, cur_name = self._get_data_from_specs(cur_data_spec)

            plt.hist(values, bins=n_bins)
            plt.xlabel(cur_name)
            plt.ylabel("Count")
            plt.title(f"{cur_name}")

            plt.grid()

            fig.savefig(self.output_dir / f"{cur_name}_hist.png")

        plt.close()

    def gen_multi_hist_plot(
        self,
        data_spec_hist: Union[str, DataSpec],
        data_spec_x: Union[str, DataSpec],
        bin_x: Union[int, np.ndarray, Callable[[float, float], np.ndarray]],
        data_spec_y: Union[str, DataSpec] = None,
        n_bins: int = 10,
    ):
        def _get_bin_values(
            bin_arg: Union[int, np.ndarray, Callable[[float, float], np.ndarray]],
            values: np.ndarray,
        ):
            if isinstance(bin_arg, int):
                return np.linspace(values.min(), values.max(), bin_x + 1)
            elif isinstance(bin_arg, np.ndarray):
                return bin_arg
            else:
                return bin_arg(values.min(), values.max())

        values_hist, name_hist = self._get_data_from_specs(data_spec_hist)
        values_x, name_x = self._get_data_from_specs(data_spec_x)
        if data_spec_y is None:
            bin_x = _get_bin_values(bin_x, values_x)

            bin_x_ind = np.digitize(values_x, bin_x)

            n_cols = len(bin_x) - 1
            fig = multi_fig((9, 6), 1, n_cols)
            for cur_bin_ix in range(n_cols):
                # Bin indices start from 1
                cur_bin_ix += 1

                if not np.any(bin_x_ind == cur_bin_ix):
                    continue

                cur_ax = fig.add_subplot(1, n_cols, cur_bin_ix)

                cur_mask = bin_x_ind == cur_bin_ix
                cur_ax.hist(values_hist[cur_mask], bins=n_bins)

                cur_ax.text(
                    0.99,
                    0.99,
                    f"{name_x}: {bin_x[cur_bin_ix - 1]}-{bin_x[cur_bin_ix]}, "
                    f"N: {np.count_nonzero(cur_mask)}",
                    horizontalalignment="right",
                    verticalalignment="top",
                    transform=cur_ax.transAxes,
                )
                cur_ax.grid(linestyle="--", linewidth=0.25, alpha=0.75)
                cur_ax.set_xlabel(name_x)
                cur_ax.set_ylabel("Count")

            fig.tight_layout()
            fig.savefig(self.output_dir / f"{name_hist}_multi_hist_{name_x}.png")
        else:
            raise NotImplementedError()

        plt.close()

    def gen_2d_hist_plot(
        self,
        data_spec_x: Union[str, DataSpec],
        data_spec_y: Union[str, DataSpec],
        n_bins: int = 10,
        weights: Union[str, DataSpec] = None,
        log_z: bool = True,
    ):
        """Generates a 2D histogram (with count as colour) the two specified data keys"""
        data_spec_x = (
            data_spec_x if isinstance(data_spec_x, tuple) else DataSpec(data_spec_x, None, None)
        )
        data_spec_y = (
            data_spec_y if isinstance(data_spec_y, tuple) else DataSpec(data_spec_y, None, None)
        )

        values_x, name_x = self._get_data_from_specs(data_spec_x)
        values_y, name_y = self._get_data_from_specs(data_spec_y)
        value_weights, _ = (
            self._get_data_from_specs(weights) if weights is not None else (None, None)
        )

        cur_name = f"{name_x}_{name_y}"

        fig = plt.figure(figsize=(18, 13.5))
        plt.hist2d(
            values_x,
            values_y,
            bins=n_bins,
            norm=matplotlib.colors.LogNorm() if log_z else None,
            weights=value_weights,
        )
        plt.xlabel(name_x)
        plt.ylabel(name_y)
        plt.colorbar()
        plt.title(f"{cur_name} - 2D Histogram")

        fig.tight_layout()
        fig.savefig(self.output_dir / f"{cur_name}_2d_hist_{n_bins}.png")

        plt.close()