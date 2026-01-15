import torch
import torchmetrics as tm  # https://lightning.ai/docs/torchmetrics/stable/all-metrics.html

"""
Template metrics to be used for EVE spectral forecasting.

This mirrors FlareMetrics, but supports vector-valued targets:
- preds: (B, L) or (B, L, 1) or similar
- target: (B, L)

We compute:
- mse: mean squared error over all elements (batch and wavelength)
- rrse: relative root squared error over all elements (batch and wavelength)

The same mode pattern ("train_loss", "train_metrics", "val_metrics") is preserved.
"""


class EVEMetrics:
    def __init__(self, mode: str):
        """
        Initialize EVEMetrics class.

        Args:
            mode (str): Mode to use for metric evaluation. Can be "train_loss",
                        "train_metrics", or "val_metrics".
        """
        self.mode = mode

        # RRSE (root relative squared error)
        # Torchmetrics expects preds/target shapes like (N, ...) and reduces internally.
        self._rrse = tm.RelativeSquaredError(squared=False)

    def _ensure_device(self, preds: torch.Tensor):
        # Move metric module to the same device as preds, but only when needed
        if self._rrse.device != preds.device:
            self._rrse = self._rrse.to(preds.device)

    def _standardize_shapes(self, preds: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Ensure preds and target are float tensors with shape (B, L).

        - If preds has trailing singleton dims (e.g., (B, L, 1)), squeeze them.
        - If target has trailing singleton dims, squeeze them.
        """
        preds = preds.float()
        target = target.float()

        # Squeeze only singleton dimensions at the end (safe for typical model outputs)
        while preds.ndim > 2 and preds.shape[-1] == 1:
            preds = preds.squeeze(-1)
        while target.ndim > 2 and target.shape[-1] == 1:
            target = target.squeeze(-1)

        # If someone provides (B,) accidentally, make it (B, 1)
        if preds.ndim == 1:
            preds = preds.unsqueeze(-1)
        if target.ndim == 1:
            target = target.unsqueeze(-1)

        if preds.ndim != 2 or target.ndim != 2:
            raise ValueError(f"Expected preds and target to be 2D after standardization. "
                             f"Got preds {tuple(preds.shape)}, target {tuple(target.shape)}")

        if preds.shape != target.shape:
            raise ValueError(f"Shape mismatch after standardization: preds {tuple(preds.shape)} "
                             f"vs target {tuple(target.shape)}")

        return preds, target

    def train_loss(
        self, preds: torch.Tensor, target: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], list[float]]:
        """
        Calculate loss metrics for training.

        Returns:
            (metrics_dict, weights_list)
        """
        output_metrics = {}
        output_weights = []

        preds, target = self._standardize_shapes(preds, target)

        output_metrics["mse"] = torch.nn.functional.mse_loss(preds, target)
        output_weights.append(1)

        return output_metrics, output_weights

    def train_metrics(
        self, preds: torch.Tensor, target: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], list[float]]:
        """
        Calculate evaluation metrics for training (reporting only).
        """
        output_metrics = {}
        output_weights = []

        preds, target = self._standardize_shapes(preds, target)

        self._ensure_device(preds)
        output_metrics["rrse"] = self._rrse(preds, target)
        output_weights.append(1)

        return output_metrics, output_weights

    def val_metrics(
        self, preds: torch.Tensor, target: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], list[float]]:
        """
        Calculate metrics for validation.
        """
        output_metrics = {}
        output_weights = []

        preds, target = self._standardize_shapes(preds, target)

        output_metrics["mse"] = torch.nn.functional.mse_loss(preds, target)
        output_weights.append(1)

        self._ensure_device(preds)
        output_metrics["rrse"] = self._rrse(preds, target)
        output_weights.append(1)

        return output_metrics, output_weights

    def __call__(
        self, preds: torch.Tensor, target: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], list[float]]:
        """
        Dispatch to the appropriate metric set based on self.mode.
        """
        match self.mode.lower():

            case "train_loss":
                return self.train_loss(preds, target)

            case "train_metrics":
                with torch.no_grad():
                    return self.train_metrics(preds, target)

            case "val_metrics":
                with torch.no_grad():
                    return self.val_metrics(preds, target)

            case _:
                raise NotImplementedError(
                    f"{self.mode} is not implemented as a valid metric case."
                )
