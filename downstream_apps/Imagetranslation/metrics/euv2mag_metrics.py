"""
Metrics for EUV-to-magnetogram image translation.

``Euv2MagMetrics`` defines four metric sets, selected by the ``mode`` passed at
construction:

- "train_loss"    — the differentiable loss that drives backpropagation (MSE).
- "val_loss"      — what is logged as ``val_loss`` and therefore what ModelCheckpoint
                    selects on. Defaults to the same MSE as "train_loss".
- "train_metrics" — reported during training (MSE + MAE). Not backpropagated.
- "val_metrics"   — reported at validation (MSE + MAE). These do NOT select checkpoints;
                    "val_loss" does.

The dict keys returned by each method become the metric names in the logger.

**These are all in normalized space**, because that is what the model outputs and what the
target arrives as. An MSE of 0.1 here is 0.1 of a per-channel standard deviation, not
0.1 Gauss. For numbers in physical units, inverse-transform both sides first with
``dataset.inverse_transform_data`` (see the "THE THREE SPACES" block at the top of
``workshop_infrastructure/datasets/helio.py``) — the evaluation tail of the training
script does exactly that for its figure.

MAE is reported alongside MSE because magnetograms are dominated by quiet Sun near zero
with a thin tail of strong active-region field. MSE is driven almost entirely by that
tail; MAE says whether the quiet Sun is right. Watching only one of them hides half the
behaviour.
"""

from __future__ import annotations

import torch
import torchmetrics as tm


class Euv2MagMetrics:
    """Shape contract: ``preds`` and ``target`` are both (B, C, H, W), matching
    ``Euv2MagSuryaModel.forward`` and the ``euv2mag_target`` in the Lightning module.
    Every metric reduces over all axes to a scalar."""

    def __init__(self, mode: str):
        self.mode = mode
        # Cached once rather than rebuilt per call; moved to the right device lazily.
        self._mse = tm.MeanSquaredError()
        self._mae = tm.MeanAbsoluteError()

    def _ensure_device(self, preds: torch.Tensor) -> None:
        if self._mse.device != preds.device:
            self._mse = self._mse.to(preds.device)
        if self._mae.device != preds.device:
            self._mae = self._mae.to(preds.device)

    def train_loss(
        self, preds: torch.Tensor, target: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], list[float]]:
        """The backpropagated loss. Uses the functional form, not the cached torchmetrics
        module, because a Metric's internal state accumulation is not what you want in a
        differentiable path."""
        return {"mse": torch.nn.functional.mse_loss(preds, target)}, [1]

    def val_loss(
        self, preds: torch.Tensor, target: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], list[float]]:
        """What ModelCheckpoint monitors. The same MSE as ``train_loss`` by default —
        override here if you want checkpoints chosen on something else, e.g. an
        active-region-weighted error rather than a whole-disk average."""
        return self.train_loss(preds, target)

    def _reported(
        self, preds: torch.Tensor, target: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], list[float]]:
        self._ensure_device(preds)
        return {"mse": self._mse(preds, target), "mae": self._mae(preds, target)}, [1, 1]

    def train_metrics(self, preds, target):
        """Reported during training only; does not contribute to the loss."""
        return self._reported(preds, target)

    def val_metrics(self, preds, target):
        """Reported at validation only; does not influence checkpoint selection."""
        return self._reported(preds, target)

    def __call__(
        self, preds: torch.Tensor, target: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], list[float]]:
        match self.mode.lower():
            case "train_loss":
                return self.train_loss(preds, target)
            case "val_loss":
                return self.val_loss(preds, target)
            case "train_metrics":
                with torch.no_grad():
                    return self.train_metrics(preds, target)
            case "val_metrics":
                with torch.no_grad():
                    return self.val_metrics(preds, target)
            case _:
                raise NotImplementedError(
                    f"{self.mode!r} is not a valid Euv2MagMetrics mode. Valid modes are "
                    "'train_loss', 'val_loss', 'train_metrics', 'val_metrics'."
                )
