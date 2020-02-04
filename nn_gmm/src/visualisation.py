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
    n_plots = len(df.columns)
    n_rows = np.ceil(n_plots / n_cols)

    fig = multi_fig(ind_fig_size, n_rows, n_cols)
    for ix, cur_col in enumerate(df.columns):
        ax = fig.add_subplot(n_rows, n_cols, ix + 1)
        sns.distplot(df.loc[:, cur_col], bins=bins, ax=ax, rug=False, kde=False)

    fig.suptitle(title)
    fig.savefig(plot_ffp)


def visualisation(eval_result: evaluation.EvaluationResult):
    """"""
    print(
        f"=============================== Visualisation ==============================="
    )

    output_dir = os.path.join(eval_result.training_result.output_dir, "visualisation")
    os.mkdir(output_dir)

    print(f"Creating residual plots")
    # Residual plots
    create_multi_hist(
        eval_result.res_train,
        os.path.join(output_dir, "residual_train.png"),
        title="Training residual",
    )
    create_multi_hist(
        eval_result.res_val,
        os.path.join(output_dir, "residual_val.png"),
        title="Validation residual",
    )

    # Ln residual plots
    create_multi_hist(
        eval_result.ln_res_train,
        os.path.join(output_dir, "ln_residual_train.png"),
        title="Log training residual",
    )
    create_multi_hist(
        eval_result.ln_res_val,
        os.path.join(output_dir, "ln_residual_val.png"),
        title="Log validation residual",
    )

    # Relative residual plots
    create_multi_hist(
        eval_result.rel_res_train,
        os.path.join(output_dir, "relative_residual_train.png"),
        title="Relative training residual",
    )
    create_multi_hist(
        eval_result.rel_res_val,
        os.path.join(output_dir, "relative_residual_val.png"),
        title="Relative validation residual",
    )

    return
