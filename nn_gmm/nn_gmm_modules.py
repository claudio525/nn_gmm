from collections.abc import Sequence

import torch
import torch.nn as nn

import ml_tools as mlt


def create_multi_mlp(
    n_inputs: int,
    units: Sequence[int],
    n_outputs: int,
    act_fn_str: str | None,
    bias: bool = True,
    use_batch_norm: bool = False
) -> nn.Sequential:
    """Creates a multi-layer perceptron"""
    mlp = nn.Sequential()
    for ix, cur_n_units in enumerate(units):
        mlp.append(
            nn.Linear(
                n_inputs if ix == 0 else units[ix - 1],
                cur_n_units,
                bias=bias,
            )
        ),

        if use_batch_norm:
            mlp.append(nn.BatchNorm1d(cur_n_units))

        if act_fn_str is not None:
            mlp.append(mlt.torch.get_act_fn_layer(act_fn_str))

    mlp.append(
        nn.Linear(
            units[-1],
            n_outputs,
            bias=bias,
        )
    )

    return mlp

def get_n_params(model: nn.Module) -> int:
    """
    Get the number of parameters in a PyTorch model.

    Parameters
    ----------
    model : nn.Module
        The PyTorch model.

    Returns
    -------
    int
        The number of parameters in the model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

    