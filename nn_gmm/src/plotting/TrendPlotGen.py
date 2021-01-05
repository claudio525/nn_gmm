"""Class for generating plots that show model behaviour with respect to one
or more inputs"""

from pathlib import Path
from typing import Union, Tuple, Dict

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import empirical.util.classdef as classdef
import empirical.util.empirical_factory as emp_factory

import ml_tools
from .. import GMM
from . import plotting_funcs as plt_funcs


class TrendPlotGen:
    def __init__(self, model: GMM, train_data_dir: Path, val_data_dir: Path = None):
        self.model = model

    def add_est_trend(
        self,
        feature: str,
        feature_values: np.ndarray,
        const_features: pd.Series,
        im: str,
        ax: plt.Axes,
        locations: pd.DataFrame = None,
        scatter_kwargs: Dict = None
    ):
        """
        Adds the model's prediction when varying a single
        feature and keeping the other's constant
        
        Parameters
        ----------
        feature: str
            Name of the feature to vary
        feature_values: array of floats
            Values at which to evaluate
        const_features: series
            Values of the other features, which are kept constant
        im: str
            IM of interest
        ax: Axes
        locations: dataframe, optional
            If specified, then the model is evaluated
            at all of the given locations (and all are plotted)
        scatter_kwargs: dictionary
            Kwargs to be passed to the scatter call
        """
        # Get the predictions
        feature_df, (est_mean_df, est_std_df) = self.get_est_site_values(
            feature, feature_values, const_features, locations=locations
        )

        # Plot
        scatter_kwargs = scatter_kwargs if scatter_kwargs is not None else {}
        scatter_kwargs = {"s": 1.0, "color": 'k', **scatter_kwargs}
        ax.scatter(feature_df[feature], est_mean_df[im], **scatter_kwargs)

    def add_B13(
        self,
        x_var: str,
        x_values,
        const_features: pd.Series,
        im: str,
        ax: plt.Axes = None,
    ):
        """Adds the B13 predictions to thje given Axis"""
        b13_df = self.get_B13_values(im, x_var, x_values, const_features)

        if b13_df is not None:
            plt_funcs.add_emp(b13_df, label="Bradley 2013", ax=ax)

    def add_data_in_range(
        self,
        feature: str,
        feature_range: Tuple[float, float],
        range_feature_dict: Dict[str, Tuple[float, float]],
        im: str,
        data_dir: Path,
        data_details: Dict,
        ax: plt.Axes,
        n_points: int = 10_000,
        scatter_kwargs: Dict = None,
    ):
        feature_values, im_values = self.get_data_in_range(
            feature, feature_range, range_feature_dict, im, data_dir, data_details
        )

        scatter_kwargs = scatter_kwargs if scatter_kwargs is not None else {}
        scatter_kwargs = {"s": 1.0, "alpha": 0.3, **scatter_kwargs}

        point_ind = np.arange(feature_values.shape[0])
        if n_points is not None:
            point_ind = np.random.choice(point_ind, n_points, replace=False)

        ax.scatter(feature_values[point_ind], np.exp(im_values[point_ind]), **scatter_kwargs)

    def get_data_in_range(
        self,
        feature: str,
        feature_range: Tuple[float, float],
        range_feature_dict: Dict[str, Tuple[float, float]],
        im: str,
        data_dir: Path,
        data_details: Dict,
    ):
        ds = ml_tools.data.load_dataset(
            data_details,
            int(1e6),
            data_dirs=data_dir,
            shuffle_buffer=None,
            n_open_files=128,
        )

        im_values, feature_values = [], []
        for cur_batch in ds.as_numpy_iterator():
            mask = (cur_batch[feature] > feature_range[0]) & (
                cur_batch[feature] < feature_range[1]
            )

            for (
                cur_range_feature,
                (cur_centre, cur_variance),
            ) in range_feature_dict.items():
                cur_min, cur_max = cur_centre - cur_variance, cur_centre + cur_variance
                mask &= (cur_batch[cur_range_feature] > cur_min) & (
                    cur_batch[cur_range_feature] < cur_max
                )

            im_values.append(cur_batch[im][mask])
            feature_values.append(cur_batch[feature][mask])

        im_values = np.concatenate(im_values)
        feature_values = np.concatenate(feature_values)
        return feature_values, im_values

    def create_single_trend_plot(
        self,
        x_var: str,
        x_values: np.ndarray,
        const_features: pd.Series,
        im: str,
        output_ffp: Path,
        locations: pd.DataFrame = None,
        range_var_dict: Dict[str, Tuple[float, float]] = None,
        data_dir: Path = None,
        data_details: Dict = None,
    ):
        fig, ax = plt.subplots(figsize=(7, 5.5))
        if range_var_dict is not None and data_dir is not None and data_details is not None:
            self.add_data_in_range(x_var, (1.0, 300), range_var_dict, im, data_dir, data_details, ax)

        self.add_est_trend(x_var, x_values, const_features, im,
                                    locations=locations, ax=ax)

        self.add_B13(x_var, x_values, const_features, im)

        ax.grid(linestyle="--", linewidth=0.2)
        ax.set_yscale("log")
        ax.set_xlabel(x_var)
        ax.set_ylabel(im)

        fig.tight_layout()
        fig.savefig(str(output_ffp))

    def get_est_site_values(
        self,
        feature_key: str,
        feature_values: np.ndarray,
        const_values: pd.Series,
        locations: pd.DataFrame = None,
    ) -> Tuple[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
        """Computes the estimated mean & std using the
        given NN model, varied along one feature at each of the
        given locations

        Parameters
        ----------
        feature_key: str, key
            The feature to vary
        feature_values: numpy array of floats
            The feature values at which to compute the IM value
        const_values: series
            The constant feature values to use
        locations: dataframe
            Locations of interest,
            required columns [lon, lat]

        Returns
        -------
        feature_df: dataframe
            The feature dataframe used
        mean_est_df: dataframe
        std_est_df: dataframe
            The mean and std dataframes estimations
        """
        if locations is not None:
            feature_df = pd.DataFrame(
                data=np.concatenate(
                    (
                        np.repeat(locations.values, len(feature_values), axis=0),
                        np.tile(feature_values, locations.shape[0])[:, np.newaxis],
                    ),
                    axis=1,
                ),
                columns=list(locations.columns) + [feature_key],
            )
        else:
            feature_df = pd.DataFrame(data=feature_values, columns=[feature_key])

        constants_df = pd.DataFrame(
            data=np.repeat(
                const_values.values[np.newaxis, :], feature_df.shape[0], axis=0
            ),
            columns=const_values.index.values,
        )

        # Drop the column that is varied
        if feature_key in constants_df.columns:
            constants_df.drop(columns=[feature_key], inplace=True)

        # Merge
        feature_df = feature_df.merge(constants_df, left_index=True, right_index=True)

        # Run estimation
        mean_est_df, std_est_df = self.model.predict(feature_df, pre_process=True)
        mean_est_df[["lon", "lat", feature_key]] = feature_df[
            ["lon", "lat", feature_key]
        ]
        if std_est_df is not None:
            std_est_df[["lon", "lat", feature_key]] = feature_df[
                ["lon", "lat", feature_key]
            ]
        return feature_df, (mean_est_df, std_est_df)

    def get_B13_values(
        self,
        im: str,
        feature_key: str,
        feature_values: np.ndarray,
        const_values: pd.Series,
    ) -> Union[pd.DataFrame, None]:
        """Computes the IM values using the Bradley 2013 GMPE model,
        with one feature being varied and all others being kept constant

        Parameters
        ----------
        im: str
            The IM of interest
        feature_key: str, key
            The feature to vary
        feature_values: numpy array of floats
            The feature values at which to compute the IM value
        const_values: series
            The constant feature values to use

        Returns
        -------
        dataframe
            index: varied feature values
            columns: mu and sigma
        """
        if im.lower() not in ["pga", "pgv"] and not im.lower().startswith("psa"):
            return None

        period = 0 if im.lower() in ["pga", "pgv"] else float(im.split("_")[-1])
        im_values = []
        im_sigmas = []
        if feature_key == "rrup":
            fault = classdef.Fault(
                Mw=const_values.mag,
                rake=const_values.rake,
                dip=const_values.dip,
                ztor=const_values.ztor,
            )

            for cur_rrup in feature_values:
                cur_site = classdef.Site(
                    rrup=cur_rrup,
                    rjb=cur_rrup,
                    rx=cur_rrup,
                    hw=True,
                    rtvz=0,
                    vs30=const_values.vs30,
                    vs30measured=False,
                )
                emp_result = emp_factory.compute_gmm(
                    fault, cur_site, classdef.GMM.Br_10, im, [period]
                )
                if period == 0:
                    cur_mean, (cur_sigma, _, __) = emp_result
                else:
                    cur_mean, (cur_sigma, _, __) = emp_result[0]

                im_values.append(cur_mean)
                im_sigmas.append(cur_sigma)
            return pd.DataFrame(
                index=feature_values,
                columns=["mu", "sigma"],
                data=np.asarray([im_values, im_sigmas]).T,
            )

        elif feature_key == "mag":
            site = classdef.Site(
                rrup=const_values.rrup,
                rjb=const_values.rrup,
                rx=const_values.rrup,
                hw=True,
                rtvz=0,
                vs30=const_values.vs30,
                vs30measured=False,
            )

            for cur_mag in feature_values:
                cur_fault = classdef.Fault(
                    Mw=cur_mag,
                    rake=const_values.rake,
                    dip=const_values.dip,
                    ztor=const_values.ztor,
                )

                emp_result = emp_factory.compute_gmm(
                    cur_fault, site, classdef.GMM.Br_10, im, [period]
                )
                if period == 0:
                    cur_mean, (cur_sigma, _, __) = emp_result
                else:
                    cur_mean, (cur_sigma, _, __) = emp_result[0]

                im_values.append(cur_mean)
                im_sigmas.append(cur_sigma)
            return pd.DataFrame(
                index=feature_values,
                columns=["mu", "sigma"],
                data=np.asarray([im_values, im_sigmas]).T,
            )

        elif feature_key == "vs30":
            fault = classdef.Fault(
                Mw=const_values.mag,
                rake=const_values.rake,
                dip=const_values.dip,
                ztor=const_values.ztor,
            )

            for cur_vs30 in feature_values:
                cur_site = classdef.Site(
                    rrup=const_values.rrup,
                    rjb=const_values.rjb,
                    rx=const_values.rx,
                    hw=True,
                    rtvz=0,
                    vs30=cur_vs30,
                    vs30measured=True,  # not sure about this parameter
                )
                emp_result = emp_factory.compute_gmm(
                    fault, cur_site, classdef.GMM.Br_10, im, [period]
                )
                if period == 0:
                    cur_mean, (cur_sigma, _, __) = emp_result
                else:
                    cur_mean, (cur_sigma, _, __) = emp_result[0]

                im_values.append(cur_mean)
                im_sigmas.append(cur_sigma)
            return pd.DataFrame(
                index=feature_values,
                columns=["mu", "sigma"],
                data=np.asarray([im_values, im_sigmas]).T,
            )
