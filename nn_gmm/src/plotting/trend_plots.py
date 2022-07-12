from typing import Dict, Sequence, Union, List
from pathlib import Path

import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ml_tools

from empirical.util import classdef
from empirical.util import empirical_factory as emp_factory

from . import plotting_utils as plt_utils
from nn_gmm.src.ResultDB import ResultDB
from nn_gmm.src.model import GMM
from nn_gmm.src.console import console

MAG_DATA_IGNORE_FEATURES = [
    "mag",
    "rake",
    "dip",
    "width",
    "ztor",
    "hdepth",
    "is_point_source",
    "active_shallow",
    "volcanic",
]
RRUP_DATA_IGNORE_FEATURES = ["rrup", "rx", "ry", "rjb"]
VS30_DATA_IGNORE_FEATURES = ["vs30", "v500", "z1p0", "z2p5"]


def gen_mag_trend_plot(
    const_features: Dict,
    ims: Sequence[str],
    output_dir: Path,
    model_dirs: Sequence[Path],
    model_names: Sequence[str] = None,
    data_constraints: Dict = None,
    interval_half_size: float = 0.1,
    plt_kwargs: Dict = None,
    add_Br13: bool = True,
):
    mag_values = np.arange(3, 8.6, 0.1)
    gen_trend_plot(
        "mag",
        mag_values,
        None,
        const_features,
        ims,
        output_dir,
        model_dirs,
        model_names=model_names,
        data_constraints=data_constraints,
        interval_half_size=interval_half_size,
        plt_kwargs=plt_kwargs,
        add_Br13=add_Br13,
    )


def gen_trend_plot(
    feature: str,
    feature_values: np.ndarray,
    dependent_features: Union[Dict, None],
    const_features: Dict,
    ims: Sequence[str],
    output_dir: Path,
    model_dirs: Sequence[Path],
    model_names: Sequence[str] = None,
    data_constraints: Dict = None,
    interval_half_size: float = 0.1,
    plt_kwargs: Dict = None,
    add_Br13: bool = True,
):
    assert feature in [
        "rrup",
        "mag",
        "vs30",
    ], "Trend plots are currently only support for Rrup, Magnitude and Vs30"

    if len(model_dirs) > 1:
        console.print(
            f"[orange1]This function assumes that all models used the "
            f"same training and validation data. Otherwise this plot is not valid![/]"
        )

    plt_kwargs = (
        {**plt_utils.DEFAULT_PLOT_KWARGS, **plt_kwargs}
        if plt_kwargs is not None
        else plt_utils.DEFAULT_PLOT_KWARGS
    )

    if model_names is None:
        model_names = [cur_model_dir.stem for cur_model_dir in model_dirs]

    output_dir.mkdir(exist_ok=True, parents=True)

    # Get predictions
    model_estimations, br13_esimations = _get_predictions(
        feature,
        feature_values,
        dependent_features,
        const_features,
        ims,
        model_dirs,
        add_Br13=add_Br13,
    )

    # Load the training and validation data
    columns = ims + list(const_features.keys())
    train_data = ResultDB.get_data_static(
        model_dirs[0] / "train_predictions.hdf5", columns
    )
    val_data = ResultDB.get_data_static(model_dirs[0] / "val_predictions.hdf5", columns)

    # Get relevant feature constraints
    comp_data_constraints = _get_feature_contraints(
        const_features, train_data, interval_half_size,
    )
    data_constraints = {**comp_data_constraints, **data_constraints}
    train_data_mask, val_data_mask = _apply_feature_constraints(
        feature, train_data, val_data, const_features, data_constraints
    )

    # Save the model inputs & data constraints
    ml_tools.utils.write_to_yaml(
        {
            cur_key: dict(
                value=const_features[cur_key], constraints=data_constraints[cur_key]
            )
            for cur_key in const_features.keys()
            if cur_key in train_data.columns
        },
        output_dir / "inputs.yaml",
        clobber=True,
    )

    # Generate the plots
    for im in ims:
        fig, ax = plt.subplots(figsize=(16, 10), dpi=200)

        _plot_models(
            ax,
            im,
            feature_values,
            model_estimations,
            model_names,
            plt_kwargs,
            br13_estimations=br13_esimations,
        )

        # Plot the datapoints
        ax.scatter(
            train_data.loc[train_data_mask, feature].values,
            np.exp(train_data.loc[train_data_mask, im].values),
            alpha=0.3,
            marker="o",
            c="gray",
        )
        ax.scatter(
            val_data.loc[val_data_mask, feature].values,
            np.exp(val_data.loc[val_data_mask, im].values),
            alpha=0.3,
            marker="x",
            c="c",
        )

        # Add constraint text
        if feature == "rrup":
            ax.text(
                0.01,
                0.01,
                f"{plt_utils.get_feature_name('mag')} = {const_features['mag']:.2f} "
                f"({data_constraints['mag'][0]:.2f}, {data_constraints['mag'][1]:.2f})\n"
                f"{plt_utils.get_feature_name('vs30')} = {const_features['vs30']} "
                f"({int(data_constraints['vs30'][0])}, {int(data_constraints['vs30'][1])})",
                horizontalalignment="left",
                verticalalignment="bottom",
                transform=ax.transAxes,
            )
            ax.loglog()
        if feature == "mag":
            ax.text(
                0.01,
                0.01,
                f"{plt_utils.get_feature_name('rrup')} = {const_features['rrup']:.2f} "
                f"({data_constraints['rrup'][0]:.2f}, {data_constraints['rrup'][1]:.2f})\n"
                f"{plt_utils.get_feature_name('vs30')} = {const_features['vs30']} "
                f"({int(data_constraints['vs30'][0])}, {int(data_constraints['vs30'][1])})",
                horizontalalignment="left",
                verticalalignment="bottom",
                transform=ax.transAxes,
            )
            ax.semilogy()
        if feature == "vs30":
            ax.text(
                0.01,
                0.01,
                f"{plt_utils.get_feature_name('mag')} = {const_features['mag']:.2f} "
                f"({data_constraints['mag'][0]:.2f}, {data_constraints['mag'][1]:.2f})\n"
                f"{plt_utils.get_feature_name('rrup')} = {const_features['rrup']} "
                f"({int(data_constraints['rrup'][0])}, {int(data_constraints['rrup'][1])})",
                horizontalalignment="left",
                verticalalignment="bottom",
                transform=ax.transAxes,
            )
            ax.loglog()

        ax.set_xlabel(plt_utils.get_feature_name(feature))
        ax.set_ylabel(plt_utils.get_im_name(im))
        ax.legend()
        ax.grid(linestyle="--", linewidth=0.5, alpha=0.5, which="both")

        ax.set_xlim(feature_values.min(), feature_values.max())

        fig.tight_layout()
        fig.savefig(output_dir / f"{im.replace('.', 'p')}_{feature}_trend.png")

        plt.close(fig)
    del train_data, val_data


def _get_predictions(
    feature: str,
    feature_values: np.ndarray,
    dependent_feature_values: Dict,
    const_features: Dict,
    ims: Sequence[str],
    model_dirs: Sequence[Path],
    add_Br13: bool = True,
):
    """
    Runs the model predictions
    for the specified models & input data
    """

    # Get the model predictions
    model_estimations = []
    for cur_model_dir in model_dirs:
        model = GMM.load(cur_model_dir)
        model_estimations.append(
            _get_model_predictions(
                model,
                feature,
                feature_values,
                const_features,
                dependent_feature_values=dependent_feature_values,
            )
        )
        del model

    # Get Br13 GMM values
    br13_estimations = None
    if add_Br13:
        br13_estimations = {}
        for im in ims:
            br13_estimations[im] = _get_B13_values(
                im, feature, feature_values, pd.Series(const_features)
            )

    return model_estimations, br13_estimations


def _apply_feature_constraints(
    feature: str,
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    const_features: Dict,
    data_constraints: Dict,
):
    """
    Applies the given feature constraints

    Returns a training & validation mask, specifying which entries
    meet the constraints
    """
    # Apply the feature constraints
    train_data_mask = np.ones(train_data.shape[0], dtype=bool)
    val_data_mask = np.ones(val_data.shape[0], dtype=bool)
    # console.print(
    #     f"Initial Count: Train - {np.count_nonzero(train_data_mask)}, Val - {np.count_nonzero(val_data_mask)}"
    # )
    for cur_feature in const_features.keys():
        # Ignore feature of interest
        if (
            (feature == "rrup" and cur_feature in RRUP_DATA_IGNORE_FEATURES)
            or (feature == "mag" and cur_feature in MAG_DATA_IGNORE_FEATURES)
            or (feature == "vs30" and cur_feature == "vs30")
            or cur_feature not in train_data.columns
        ):
            continue

        # Boolean value features
        if cur_feature in ["is_point_source", "active_shallow", "volcanic"]:
            train_data_mask[
                train_data[cur_feature] != data_constraints[cur_feature]
            ] = False
            val_data_mask[
                val_data[cur_feature] != data_constraints[cur_feature]
            ] = False
        # Multiple constraints for this feature (combined via OR)
        elif isinstance(data_constraints[cur_feature][0], tuple) or isinstance(
            data_constraints[cur_feature][0], list
        ):
            # Require separate mask, as main mask (train_data_mask) is true by default
            cur_train_data_mask = np.zeros(train_data.shape[0], dtype=bool)
            cur_val_data_mask = np.zeros(val_data.shape[0], dtype=bool)
            # Select the ones to keep
            for cur_constraints in data_constraints[cur_feature]:
                cur_train_data_mask[
                    (train_data[cur_feature].values >= cur_constraints[0])
                    & (train_data[cur_feature].values <= cur_constraints[1])
                ] = True
                cur_val_data_mask[
                    (val_data[cur_feature].values >= cur_constraints[0])
                    & (val_data[cur_feature].values <= cur_constraints[1])
                ] = True

            # Apply to main mask
            train_data_mask[~cur_train_data_mask] = False
            val_data_mask[~cur_val_data_mask] = False
        # Single (min, max) constraint for this feature
        else:
            train_data_mask[
                (train_data[cur_feature].values < data_constraints[cur_feature][0])
                | (train_data[cur_feature].values > data_constraints[cur_feature][1])
            ] = False
            val_data_mask[
                (val_data[cur_feature].values < data_constraints[cur_feature][0])
                | (val_data[cur_feature].values > data_constraints[cur_feature][1])
            ] = False

        # console.print(
        #     f"After {cur_feature}: Train - {np.count_nonzero(train_data_mask)}, "
        #     f"Val - {np.count_nonzero(val_data_mask)} "
        # )
    return train_data_mask, val_data_mask


def _plot_models(
    ax: plt.Axes,
    im: str,
    feature_values: np.ndarray,
    model_estimations: Sequence[pd.DataFrame],
    model_names: Sequence[str],
    plt_kwargs: Dict,
    br13_estimations: Dict = None,
):
    # Plot the model values
    for cur_name, cur_est_df, cur_c in zip(
        model_names,
        model_estimations,
        sns.color_palette(n_colors=len(model_names)),
    ):
        ax.plot(
            feature_values,
            np.exp(cur_est_df[im]),
            label=cur_name,
            c=cur_c,
            linewidth=plt_kwargs["linewidth"],
        )

    if br13_estimations is not None:
        ax.plot(
            feature_values,
            br13_estimations[im].mu,
            label="Br13",
            c="r",
            linewidth=plt_kwargs["linewidth"],
        )

        ax.plot(
            feature_values,
            np.exp(np.log(br13_estimations[im].mu) + br13_estimations[im].sigma),
            c="r",
            linestyle="--",
            linewidth=plt_kwargs["linewidth"],
        )

        ax.plot(
            feature_values,
            np.exp(np.log(br13_estimations[im].mu) - br13_estimations[im].sigma),
            c="r",
            linestyle="--",
            linewidth=plt_kwargs["linewidth"],
        )


def _get_model_predictions(
    model: GMM,
    feature: str,
    feature_values: np.ndarray,
    const_features: Dict[str, Union[float, int]],
    dependent_feature_values: Dict[str, np.ndarray] = None,
):
    """Gets the model predictions, based on the constant features,
    the feature of interest and any dependent features
    """

    # Get the model predictions
    X = pd.DataFrame(
        data=np.repeat(
            np.asarray(list(const_features.values()))[None, :],
            len(feature_values),
            axis=0,
        ),
        columns=list(const_features.keys()),
    )

    # Set the values of the feature of interest (i.e. X-axis)
    X[feature] = feature_values

    # Set the value of any dependent features
    if dependent_feature_values is not None:
        for cur_feature, cur_values in dependent_feature_values.items():
            X[cur_feature] = cur_values

    # Get the model predictions
    est_df, *_ = model.predict(X)
    return est_df


def _get_feature_contraints(
    const_features: Dict, train_data: pd.DataFrame, interval_half_size: float
):
    """Computes the data selection constraints for the given
    constant features and interval size"""
    cdf_step_size = 1 / train_data.shape[0]
    data_constraints = {}
    for cur_feature, cur_value in const_features.items():
        if cur_feature not in train_data.keys():
            continue

        cur_feature_data = train_data[cur_feature].sort_values()

        # Compute the constraints
        cur_nearest_ix = ml_tools.array_utils.find_nearest(
            cur_feature_data.values, cur_value
        )
        cur_min_value = float(
            cur_feature_data.iloc[
                max(cur_nearest_ix - int(interval_half_size / cdf_step_size), 0,)
            ]
        )
        cur_max_value = float(
            cur_feature_data.iloc[
                min(
                    cur_nearest_ix + int(interval_half_size / cdf_step_size),
                    train_data.shape[0] - 1,
                )
            ]
        )

        data_constraints[cur_feature] = (cur_min_value, cur_max_value)

    return data_constraints


def _get_B13_values(
    im: str, feature_key: str, feature_values: np.ndarray, const_values: pd.Series,
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


