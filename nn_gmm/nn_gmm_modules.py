from collections.abc import Sequence

import torch
import torch.nn as nn

import ml_tools as mlt


class ClipLayer(nn.Module):
    """Layer that clips the layer input to a specified range."""
    
    def __init__(self, min_val: float = -5, max_val: float = 5):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x, self.min_val, self.max_val)

def create_multi_mlp(
    n_inputs: int,
    units: Sequence[int],
    n_outputs: int,
    act_fn_str: str | None,
    bias: bool = True,
    use_batch_norm: bool = False,
    dropout_rate: float | None = None,
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

        if dropout_rate is not None and dropout_rate > 0:
            mlp.append(nn.Dropout(dropout_rate))

    mlp.append(
        nn.Linear(
            units[-1],
            n_outputs,
            bias=bias,
        )
    )

    mlp.append(ClipLayer())

    return mlp

class NNCombined(nn.Module):

    def __init__(self, loc_model: nn.Module, core_model: nn.Module):
        super().__init__()
        self.loc_model = loc_model
        self.core_model = core_model

    def forward(self, X: torch.Tensor, X_loc: torch.Tensor) -> torch.Tensor:
        X_loc = self.loc_model(X_loc)
        X = torch.cat([X, X_loc], dim=-1)
        pred = self.core_model(X)
        return pred

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

    