import logging
from collections.abc import Sequence

import torch
import torch.nn as nn

import ml_tools as mlt

logger = logging.getLogger(__name__)

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


class BaseNNModel(nn.Module):
    """Base class for neural network models."""

    def __init__(self, model: nn.Module, max_norm: float | None = None):
        super().__init__()
        self.model = model
        self.max_norm = max_norm

        self._grad_norms = []

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.model(X)

    def apply_grad_clipping(self) -> None:
        """Apply gradient clipping to the model parameters."""
        if self.max_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_norm)

    def reset_logged_grad_norms(self) -> None:
        """Reset the logged gradient norms."""
        self._grad_norms = []

    def update_logged_grad_norms(self) -> None:
        """Add current grad norm to logged grad norms."""
        self._grad_norms.append(
            torch.nn.utils.get_total_norm(
                [param.grad for param in self.parameters() if param.grad is not None]
            )
        )

    def grad_norm_log_msg(self, reset: bool = True) -> str:
        """Get a log message for the gradient norms."""
        if len(self._grad_norms) == 0:
            return "No gradient norms recorded."

        grad_norms = torch.tensor(self._grad_norms)
        if reset:
            self.reset_logged_grad_norms()

        return (
            f"Gradient Norm: {torch.mean(grad_norms):.4f} ± {torch.std(grad_norms):.4f}, "
            f"Max: {torch.max(grad_norms):.4f}, Min: {torch.min(grad_norms):.4f}"
        )


class NNCombined(BaseNNModel):

    def __init__(
        self,
        loc_model: nn.Module,
        core_model: nn.Module,
        max_loc_grad_norm: float | None = None,
        max_core_grad_norm: float | None = None,
    ):
        nn.Module.__init__(self)
        self.loc_model = loc_model
        self.core_model = core_model

        self.max_loc_norm = max_loc_grad_norm
        self.max_core_norm = max_core_grad_norm

        self._grad_norms = []
        self._loc_grad_norms = []
        self._core_grad_norms = []

    def apply_grad_clipping(self) -> None:
        """Apply gradient clipping to the model parameters."""
        if self.max_loc_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.loc_model.parameters(), self.max_loc_norm
            )

        if self.max_core_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.core_model.parameters(), self.max_core_norm
            )

    def forward(self, X: torch.Tensor, X_loc: torch.Tensor) -> torch.Tensor:
        X_loc = self.loc_model(X_loc)
        X = torch.cat([X, X_loc], dim=-1)
        pred = self.core_model(X)
        return pred

    def reset_logged_grad_norms(self) -> None:
        """Reset the gradient norms."""
        self._grad_norms = []
        self._loc_grad_norms = []
        self._core_grad_norms = []

    def update_logged_grad_norms(self) -> None:
        """Update the gradient norms."""
        self._grad_norms.append(
            torch.nn.utils.get_total_norm(
                [param.grad for param in self.parameters() if param.grad is not None]
            )
        )

        self._loc_grad_norms.append(
            torch.nn.utils.get_total_norm(
                [
                    param.grad
                    for param in self.loc_model.parameters()
                    if param.grad is not None
                ]
            )
        )

        self._core_grad_norms.append(
            torch.nn.utils.get_total_norm(
                [
                    param.grad
                    for param in self.core_model.parameters()
                    if param.grad is not None
                ]
            )
        )

    def grad_norm_log_msg(self, reset: bool = True) -> str:
        """Get a log message for the gradient norms."""
        if len(self._grad_norms) == 0:
            return "No gradient norms recorded."

        grad_norms = torch.tensor(self._grad_norms)
        loc_grad_norms = torch.tensor(self._loc_grad_norms)
        core_grad_norms = torch.tensor(self._core_grad_norms)
        if reset:
            self.reset_logged_grad_norms()

        return (
            f"Gradient Norm: {torch.mean(grad_norms):.2f} ± {torch.std(grad_norms):.2f}, "
            f"Max: {torch.max(grad_norms):.2f}, Min: {torch.min(grad_norms):.2f}\n"
            f"  Location Sub-Model Gradient Norm: {torch.mean(loc_grad_norms):.2f} ± {torch.std(loc_grad_norms):.2f}, "
            f"Max: {torch.max(loc_grad_norms):.2f}, Min: {torch.min(loc_grad_norms):.2f}\n"
            f"  Core Model Gradient Norm: {torch.mean(core_grad_norms):.2f} ± {torch.std(core_grad_norms):.2f}, "
            f"Max: {torch.max(core_grad_norms):.2f}, Min: {torch.min(core_grad_norms):.2f}"
        )


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
