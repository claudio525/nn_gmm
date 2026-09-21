import logging
from collections.abc import Sequence

import ml_tools as mlt
import torch
from torch import nn

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
    add_clip_layer: bool = True,
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
            units[-1] if len(units) > 0 else n_inputs,
            n_outputs,
            bias=bias,
        )
    )

    if add_clip_layer:
        mlp.append(ClipLayer())

    return mlp


class BaseNNModel(nn.Module):
    """Base class for neural network models."""

    def __init__(
        self,
        model: nn.Module,
        max_norm: float | None = None,
        uses_loc_inputs: bool = False,
    ):
        super().__init__()
        self.model = model
        self.max_norm = max_norm
        self.uses_loc_inputs = uses_loc_inputs

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


class LocAdjModel(BaseNNModel):

    def __init__(
        self,
        loc_emb_model: nn.Module,
        adj_model: nn.Module,
        base_model: nn.Module,
        max_loc_emb_grad_norm: float | None = None,
        max_adj_grad_norm: float | None = None,
    ):
        nn.Module.__init__(self)
        self.loc_emb_model = loc_emb_model
        self.adj_model = adj_model
        self.base_model = base_model
        self.uses_loc_inputs = True

        # Freeze base model parameters
        for param in self.base_model.parameters():
            param.requires_grad = False

        self.max_loc_emb_norm = max_loc_emb_grad_norm
        self.max_adj_norm = max_adj_grad_norm

        self._grad_norms = []
        self._loc_emb_grad_norms = []
        self._adj_grad_norms = []

    def apply_grad_clipping(self) -> None:
        """Apply gradient clipping to the model parameters."""
        if self.max_loc_emb_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.loc_emb_model.parameters(), self.max_loc_emb_norm
            )

        if self.max_adj_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.adj_model.parameters(), self.max_adj_norm
            )

    def forward(self, X: torch.Tensor, X_loc: torch.Tensor) -> torch.Tensor:
        loc_emb = self.loc_emb_model(X_loc)
        base_emb = self.base_model(X)

        adj_pred = self.adj_model(torch.cat([base_emb, loc_emb], dim=-1))
        return adj_pred

    def reset_logged_grad_norms(self) -> None:
        """Reset the gradient norms."""
        self._grad_norms = []
        self._loc_emb_grad_norms = []
        self._adj_grad_norms = []

    def update_logged_grad_norms(self) -> None:
        """Update the gradient norms."""
        self._grad_norms.append(
            torch.nn.utils.get_total_norm(
                [param.grad for param in self.parameters() if param.grad is not None]
            )
        )

        self._loc_emb_grad_norms.append(
            torch.nn.utils.get_total_norm(
                [
                    param.grad
                    for param in self.loc_emb_model.parameters()
                    if param.grad is not None
                ]
            )
        )

        self._adj_grad_norms.append(
            torch.nn.utils.get_total_norm(
                [
                    param.grad
                    for param in self.adj_model.parameters()
                    if param.grad is not None
                ]
            )
        )

    def grad_norm_log_msg(self, reset: bool = True) -> str:
        """Get a log message for the gradient norms."""
        if len(self._grad_norms) == 0:
            return "No gradient norms recorded."

        grad_norms = torch.tensor(self._grad_norms)
        loc_grad_norms = torch.tensor(self._loc_emb_grad_norms)
        adj_grad_norms = torch.tensor(self._adj_grad_norms)
        if reset:
            self.reset_logged_grad_norms()

        return (
            f"Gradient Norm: {torch.mean(grad_norms):.2f} ± {torch.std(grad_norms):.2f}, "
            f"Max: {torch.max(grad_norms):.2f}, Min: {torch.min(grad_norms):.2f}\n"
            f"  Location Embedding Model Gradient Norm: {torch.mean(loc_grad_norms):.2f} ± {torch.std(loc_grad_norms):.2f}, "
            f"Max: {torch.max(loc_grad_norms):.2f}, Min: {torch.min(loc_grad_norms):.2f}\n"
            f"  Adjustment Model Gradient Norm: {torch.mean(adj_grad_norms):.2f} ± {torch.std(adj_grad_norms):.2f}, "
            f"Max: {torch.max(adj_grad_norms):.2f}, Min: {torch.min(adj_grad_norms):.2f}"
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
