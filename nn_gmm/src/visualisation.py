import os
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


from . import evaluation

sns.set()


def multi_fig(
    ind_fig_size: Tuple[float, float], n_rows: int, n_cols: int, dpi: int = 100
) -> plt.Figure:
    fig_size = (ind_fig_size[0] * n_cols, ind_fig_size[1] * n_rows)
    return plt.figure(figsize=fig_size, dpi=dpi)


def create_multi_hist(
    df: pd.DataFrame,
    plot_ffp: str,
    title: str = None,
    bins=None,
    n_cols: int = 4,
    ind_fig_size: Tuple[int, int] = (25.0, 10.0),
):
    """Creates a single large figure with histograms plots for
    each of the columns in the provided dataframe
    """
    n_plots = len(df.columns)
    n_rows = np.ceil(n_plots / n_cols)

    fig = multi_fig(ind_fig_size, n_rows, n_cols)
    for ix, cur_col in enumerate(df.columns):
        ax = fig.add_subplot(n_rows, n_cols, ix + 1)
        sns.distplot(df.loc[:, cur_col], bins=bins, ax=ax, rug=False, kde=False)

    fig.suptitle(title)
    fig.savefig(plot_ffp)


def create_IM_res_hist(
    output_ffp: str,
    train_df: pd.DataFrame,
    im: str,
    val_df: pd.DataFrame = None,
    xlim_n_std: float = None,
):
    im_mean, im_std = f"{im}_mean", f"{im}_std"

    fig = plt.figure(figsize=(12, 6))
    ax1, ax2 = fig.add_subplot(1, 2, 1), fig.add_subplot(1, 2, 2)

    # Plot mean and std histogram
    for cur_key, ax in zip([im_mean, im_std], [ax1, ax2]):
        # Filter out data points that are not withing the specified limits
        # Added to prevent outliers extending x-axis to far
        if xlim_n_std is not None:
            std_lim = train_df[cur_key].std() * xlim_n_std
            min_x = -std_lim if -std_lim > train_df[cur_key].min() else train_df[cur_key].min()
            max_x = std_lim if std_lim < train_df[cur_key].max() else train_df[cur_key].max()
            train_mask = (train_df[cur_key].values > min_x) & (train_df[cur_key].values < max_x)
            val_mask = (val_df[cur_key].values > min_x) &  (val_df[cur_key].values < max_x)

            sns.distplot(train_df.loc[train_mask, cur_key], kde=False, ax=ax)
            if val_df is not None:
                sns.distplot(val_df.loc[val_mask, cur_key], kde=False, ax=ax)
        else:
            sns.distplot(train_df[cur_key], kde=False, ax=ax)
            if val_df is not None:
                sns.distplot(val_df[cur_key], kde=False, ax=ax)

    fig.tight_layout()
    fig.savefig(output_ffp)
    plt.close()


def visualisation(eval_result: evaluation.EvaluationResult, hist_x_lim: float = None):
    """"""
    print(
        f"=============================== Visualisation ==============================="
    )

    output_dir = os.path.join(eval_result.training_result.output_dir, "visualisation")
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    # Get the different IMs predicted
    ims = np.unique(
        [
            col.split("_")[0]
            if not col.startswith("pSA")
            else "_".join(col.split("_")[0:2])
            for col in eval_result.ln_res_train.columns
        ]
    )

    for im in ims:
        create_IM_res_hist(
            os.path.join(output_dir, f"ln_res_{im}.png"),
            eval_result.ln_res_train,
            im,
            eval_result.ln_res_val,
            xlim_n_std=hist_x_lim,
        )

    # print(f"Creating residual plots")
    # # Residual plots
    # create_multi_hist(
    #     eval_result.res_train,
    #     os.path.join(output_dir, "residual_train.png"),
    #     title="Training residual",
    # )
    # create_multi_hist(
    #     eval_result.res_val,
    #     os.path.join(output_dir, "residual_val.png"),
    #     title="Validation residual",
    # )
    #
    # # Ln residual plots
    # create_multi_hist(
    #     eval_result.ln_res_train,
    #     os.path.join(output_dir, "ln_residual_train.png"),
    #     title="Log training residual",
    # )
    # create_multi_hist(
    #     eval_result.ln_res_val,
    #     os.path.join(output_dir, "ln_residual_val.png"),
    #     title="Log validation residual",
    # )
    #
    # # Relative residual plots
    # create_multi_hist(
    #     eval_result.rel_res_train,
    #     os.path.join(output_dir, "relative_residual_train.png"),
    #     title="Relative training residual",
    # )
    # create_multi_hist(
    #     eval_result.rel_res_val,
    #     os.path.join(output_dir, "relative_residual_val.png"),
    #     title="Relative validation residual",
    # )
    #
    # return
