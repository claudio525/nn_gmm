# import gc
# import tempfile
# from typing import Tuple, Iterable, Callable, Dict, List, Any, Union
# from pathlib import Path
# from collections import namedtuple
#
# import yaml
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import plotly.express as px
# import matplotlib
#
# import empirical.util.classdef as classdef
# import empirical.util.empirical_factory as emp_factory
# from visualization.gmt.plotting import plot_multiple, plot_single
#
# from nn_gmm.src.model import GMM
# from nn_gmm.src.utils import get_station_from_id, get_station_lookup, to_path, to_list
# from nn_gmm.src import data
# from nn_gmm.src.eval import get_realisation_residuals
# from .plotting_funcs import *
#
#
# class IMvsPlotGen:
#     """Computes IM vs features (such as Mw, Rrup, vs30) plots"""
#
#     CONST_DEFAULT_VALUES = pd.Series(
#         data={
#             "vs30": 388,
#             "vs500": 1.345,
#             "z1p0": 0.134,
#             "z2p5": 1.03,
#             "dip": 60,
#             "rake": 45,
#             "strike": 177,
#             "length": 18.9,
#             "width": 18.2775,
#             "hdepth": 8.990423e00,
#             "ztor": 0,
#             "dbottom": 16.8,
#             "is_point_source": 0,
#             "mag": 7.0,
#             "rjb": 91,
#             "rrup": 91,
#             "rx": 91,
#             "ry": 91,
#             "s": 2.760520e01,
#             "theta": 4.517924e01,
#             "tect_type": "ACTIVE_SHALLOW",
#         }
#     )
#
#     def __init__(self, model: GMM):
#         self.model = model
#
#     def get_B13_values(
#         self,
#         im: str,
#         feature_key: str,
#         feature_values: np.ndarray,
#         const_values: pd.Series,
#     ) -> Union[pd.DataFrame, None]:
#         """Computes the IM values using the Bradley 2013 GMPE model,
#         with one feature being varied and all others being kept constant
#
#         Parameters
#         ----------
#         im: str
#             The IM of interest
#         feature_key: str, key
#             The feature to vary
#         feature_values: numpy array of floats
#             The feature values at which to compute the IM value
#         const_values: series
#             The constant feature values to use
#
#         Returns
#         -------
#         dataframe
#             index: varied feature values
#             columns: mu and sigma
#         """
#         if im.lower() not in ["pga", "pgv"] and not im.lower().startswith("psa"):
#             return None
#
#         period = 0 if im.lower() in ["pga", "pgv"] else float(im.split("_")[-1])
#         im_values = []
#         im_sigmas = []
#         if feature_key == "rrup":
#             fault = classdef.Fault(
#                 Mw=const_values.mag,
#                 rake=const_values.rake,
#                 dip=const_values.dip,
#                 ztor=const_values.ztor,
#             )
#
#             for cur_rrup in feature_values:
#                 cur_site = classdef.Site(
#                     rrup=cur_rrup,
#                     rjb=cur_rrup,
#                     rx=cur_rrup,
#                     hw=True,
#                     rtvz=0,
#                     vs30=const_values.vs30,
#                     vs30measured=False,
#                 )
#                 emp_result = emp_factory.compute_gmm(
#                     fault, cur_site, classdef.GMM.Br_10, im, [period]
#                 )
#                 if period == 0:
#                     cur_mean, (cur_sigma, _, __) = emp_result
#                 else:
#                     cur_mean, (cur_sigma, _, __) = emp_result[0]
#
#                 im_values.append(cur_mean)
#                 im_sigmas.append(cur_sigma)
#             return pd.DataFrame(
#                 index=feature_values,
#                 columns=["mu", "sigma"],
#                 data=np.asarray([im_values, im_sigmas]).T,
#             )
#
#         elif feature_key == "mag":
#             site = classdef.Site(
#                 rrup=const_values.rrup,
#                 rjb=const_values.rjb,
#                 rx=const_values.rx,
#                 hw=True,
#                 rtvz=0,
#                 vs30=const_values.vs30,
#                 vs30measured=False,
#             )
#
#             for cur_mag in feature_values:
#                 cur_fault = classdef.Fault(
#                     Mw=cur_mag,
#                     rake=const_values.rake,
#                     dip=const_values.dip,
#                     ztor=const_values.ztor,
#                 )
#
#                 emp_result = emp_factory.compute_gmm(
#                     cur_fault, site, classdef.GMM.Br_10, im, [period]
#                 )
#                 if period == 0:
#                     cur_mean, (cur_sigma, _, __) = emp_result
#                 else:
#                     cur_mean, (cur_sigma, _, __) = emp_result[0]
#
#                 im_values.append(cur_mean)
#                 im_sigmas.append(cur_sigma)
#             return pd.DataFrame(
#                 index=feature_values,
#                 columns=["mu", "sigma"],
#                 data=np.asarray([im_values, im_sigmas]).T,
#             )
#
#         elif feature_key == "vs30":
#             fault = classdef.Fault(
#                 Mw=const_values.mag,
#                 rake=const_values.rake,
#                 dip=const_values.dip,
#                 ztor=const_values.ztor,
#             )
#
#             for cur_vs30 in feature_values:
#                 cur_site = classdef.Site(
#                     rrup=const_values.rrup,
#                     rjb=const_values.rjb,
#                     rx=const_values.rx,
#                     hw=True,
#                     rtvz=0,
#                     vs30=cur_vs30,
#                     vs30measured=True,  # not sure about this parameter
#                 )
#                 emp_result = emp_factory.compute_gmm(
#                     fault, cur_site, classdef.GMM.Br_10, im, [period]
#                 )
#                 if period == 0:
#                     cur_mean, (cur_sigma, _, __) = emp_result
#                 else:
#                     cur_mean, (cur_sigma, _, __) = emp_result[0]
#
#                 im_values.append(cur_mean)
#                 im_sigmas.append(cur_sigma)
#             return pd.DataFrame(
#                 index=feature_values,
#                 columns=["mu", "sigma"],
#                 data=np.asarray([im_values, im_sigmas]).T,
#             )
#
#     def get_est_site_values(
#         self,
#         feature_key: str,
#         feature_values: np.ndarray,
#         const_values: pd.Series,
#         locations: pd.DataFrame = None,
#     ) -> Tuple[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
#         """Computes the estimated mean & std using the
#         given NN model, varied along one feature at each of the
#         given locations
#
#         Parameters
#         ----------
#         feature_key: str, key
#             The feature to vary
#         feature_values: numpy array of floats
#             The feature values at which to compute the IM value
#         const_values: series
#             The constant feature values to use
#         locations: dataframe
#             Locations of interest,
#             required columsn [lon, lat]
#
#         Returns
#         -------
#         feature_df: dataframe
#             The feature dataframe used
#         mean_est_df: dataframe
#         std_est_df: dataframe
#             The mean and std datadrames estimations
#         """
#         if locations is not None:
#             feature_df = pd.DataFrame(
#                 data=np.concatenate(
#                     (
#                         np.repeat(locations.values, len(feature_values), axis=0),
#                         np.tile(feature_values, locations.shape[0])[:, np.newaxis],
#                     ),
#                     axis=1,
#                 ),
#                 columns=list(locations.columns) + [feature_key],
#             )
#         else:
#             feature_df = pd.DataFrame(data=feature_values, columns=[feature_key])
#
#         constants_df = pd.DataFrame(
#             data=np.repeat(
#                 const_values.values[np.newaxis, :], feature_df.shape[0], axis=0
#             ),
#             columns=const_values.index.values,
#         )
#
#         # Drop the column that is varied
#         constants_df.drop(columns=[feature_key], inplace=True)
#
#         # Merge
#         feature_df = feature_df.merge(constants_df, left_index=True, right_index=True)
#
#         # Run estimation
#         mean_est_df, std_est_df = self.model.predict(feature_df, pre_process=True)
#         mean_est_df[["lon", "lat", feature_key]] = feature_df[
#             ["lon", "lat", feature_key]
#         ]
#         if std_est_df is not None:
#             std_est_df[["lon", "lat", feature_key]] = feature_df[
#                 ["lon", "lat", feature_key]
#             ]
#         return feature_df, (mean_est_df, std_est_df)
#
#     def gen_plot(
#         self,
#         im: str,
#         feature_key: str,
#         feature_df: pd.DataFrame,
#         mean_est_df: pd.DataFrame,
#         locations: pd.DataFrame = None,
#         emp_df: pd.Series = None,
#         output_ffp: Union[str, Path] = None,
#     ):
#         """Generates an IM vs. feature plot
#
#         Parameters
#         ----------
#         im: str
#             IM of interest
#         feature_key: str
#             The feature that was varied
#         feature_df: dataframe
#             The feature values that were used for estimation
#         mean_est_df: dataframe
#             The estimated mean values
#         locations: dataframe, optional
#             The locations at which the model was evaluated
#         emp_df: dataframe, optional
#
#         output_ffp:
#
#         Returns
#         -------
#
#         """
#         output_ffp = output_ffp if isinstance(output_ffp, Path) else Path(output_ffp)
#
#         # Create the plot
#         # fig = plt.figure(figsize=(18, 13.5))
#         fig = plt.figure(figsize=(7, 5.5))
#
#         if locations is None:
#             plt.plot(feature_df[feature_key], mean_est_df[im], marker="x")
#         else:
#             if locations.shape[0] < len(MARKERS):
#                 for loc_ix in range(locations.shape[0]):
#                     cur_loc_mask = (feature_df["lon"] == locations.iloc[loc_ix].lon) & (
#                         feature_df["lat"] == locations.iloc[loc_ix].lat
#                     )
#                     plt.scatter(
#                         feature_df.loc[cur_loc_mask, feature_key],
#                         mean_est_df.loc[cur_loc_mask, im],
#                         marker=MARKERS[loc_ix],
#                     )
#             else:
#                 plt.scatter(feature_df[feature_key], mean_est_df[im], marker=".", s=1)
#
#             # Add mean & std line
#             feature_means = mean_est_df.groupby(feature_key).mean()
#             feature_stds = mean_est_df.groupby(feature_key).std()
#             plt.plot(
#                 feature_means.index.values,
#                 feature_means[im],
#                 linewidth=0.75,
#                 c="k",
#                 label="Mean of Sites - NN",
#             )
#             plt.plot(
#                 feature_means.index.values,
#                 feature_means[im] + feature_stds[im],
#                 linestyle="--",
#                 c="k",
#                 linewidth=0.75,
#                 label="Std of Sites - NN",
#             )
#             plt.plot(
#                 feature_means.index.values,
#                 feature_means[im] - feature_stds[im],
#                 linestyle="--",
#                 c="k",
#                 linewidth=0.75,
#             )
#
#         if emp_df is not None:
#             add_emp(emp_df)
#
#         plt.xlabel(feature_key)
#         plt.ylabel(im)
#         plt.yscale("log")
#         plt.grid(linestyle="--", linewidth=0.25)
#         plt.legend()
#         plt.title("{} {}".format(im, feature_key))
#
#         if output_ffp is not None:
#             if not output_ffp.parent.is_dir():
#                 output_ffp.parent.mkdir(parents=True)
#             plt.savefig(output_ffp, dpi=300)
#             plt.close()
#         else:
#             plt.show()
#
#         return fig
#
#     def gen_mean_plot(
#         self,
#         im: str,
#         feature_key: str,
#         feature_df: pd.DataFrame,
#         mean_est_df: pd.DataFrame,
#         std_est_df: pd.DataFrame,
#         emp_df: pd.Series = None,
#         output_ffp: Union[str, Path] = None,
#     ):
#         """Very similar to gen_plot, except that it takes the
#         mean of the means & stds at each location and plots those
#
#         This is not ideal, mainly exists for the presentation
#         """
#         # fig = plt.figure(figsize=(18, 13.5))
#         fig = plt.figure(figsize=(7, 5.5))
#
#         x = feature_df[feature_key].unique()
#         mean = mean_est_df.groupby(feature_key).mean()[im]
#         std = std_est_df.groupby(feature_key).mean()[im]
#         plt.plot(x, mean, c="k", linewidth=0.75, label="Mean NN")
#         plt.plot(
#             x, mean * np.exp(std), c="k", linewidth=0.75, linestyle="--", label="Std NN"
#         )
#         plt.plot(x, mean * np.exp(-std), c="k", linewidth=0.75, linestyle="--")
#
#         # Empirical plots
#         if emp_df is not None:
#             add_emp(emp_df)
#
#         plt.xlabel(feature_key)
#         plt.ylabel(im)
#         plt.yscale("log")
#         plt.grid(linestyle="--", linewidth=0.25)
#         plt.legend()
#         plt.title("{} {}".format(im, feature_key))
#
#         if output_ffp is not None:
#             if not output_ffp.parent.is_dir():
#                 output_ffp.parent.mkdir(parents=True)
#             plt.savefig(output_ffp, dpi=300)
#             plt.close()
#         else:
#             plt.show()
#
#         return fig
#
#     def gen_plots(
#         self,
#         ims: np.ndarray,
#         feature_dict: Dict[str, np.ndarray],
#         output_dir: Union[Path, None],
#         locations: pd.DataFrame = None,
#         mean: bool = False,
#     ):
#         """Generates plots for each IM
#
#         Parameters
#         ----------
#         See gen_plot
#         """
#         if not output_dir.is_dir():
#             output_dir.mkdir(parents=True)
#
#         for cur_feature, cur_feature_values in feature_dict.items():
#             (
#                 cur_feature_df,
#                 (cur_mean_est_df, cur_std_est_df,),
#             ) = self.get_est_site_values(
#                 cur_feature, cur_feature_values, self.CONST_DEFAULT_VALUES, locations
#             )
#
#             for cur_im in ims:
#                 cur_emp_df = self.get_B13_values(
#                     cur_im, cur_feature, cur_feature_values, self.CONST_DEFAULT_VALUES
#                 )
#
#                 if not mean:
#                     self.gen_plot(
#                         cur_im,
#                         cur_feature,
#                         cur_feature_df,
#                         cur_mean_est_df,
#                         locations,
#                         emp_df=cur_emp_df,
#                         output_ffp=output_dir
#                         / f"{cur_im.replace('.', 'p')}_{cur_feature}.png",
#                     )
#                 else:
#                     self.gen_mean_plot(
#                         cur_im,
#                         cur_feature,
#                         cur_feature_df,
#                         cur_mean_est_df,
#                         cur_std_est_df,
#                         emp_df=cur_emp_df,
#                         output_ffp=output_dir
#                         / f"{cur_im.replace('.', 'p')}_{cur_feature}.png",
#                     )