import torch
import torchmetrics as tm

class EVEMetrics:
    def __init__(self, mode: str, silent: bool = False):
        """
        Args:
            mode: "train_loss", "train_metrics", or "val_metrics"
            silent: if True, __call__ returns a short string instead of metrics,
                    which avoids any rich display issues in broken torch installs.
        """
        self.mode = mode
        self.silent = silent
        self._rrse = tm.RelativeSquaredError(squared=False)

    def _ensure_device(self, preds: torch.Tensor):
        if self._rrse.device != preds.device:
            self._rrse = self._rrse.to(preds.device)

    def _standardize_shapes(self, preds: torch.Tensor, target: torch.Tensor):
        preds = preds.float()
        target = target.float()

        while preds.ndim > 2 and preds.shape[-1] == 1:
            preds = preds.squeeze(-1)
        while target.ndim > 2 and target.shape[-1] == 1:
            target = target.squeeze(-1)

        if preds.ndim == 1:
            preds = preds.unsqueeze(-1)
        if target.ndim == 1:
            target = target.unsqueeze(-1)

        if preds.ndim != 2 or target.ndim != 2:
            raise ValueError(
                f"Expected preds and target to be 2D after standardization. "
                f"Got preds {tuple(preds.shape)}, target {tuple(target.shape)}"
            )
        if preds.shape != target.shape:
            raise ValueError(
                f"Shape mismatch: preds {tuple(preds.shape)} vs target {tuple(target.shape)}"
            )

        return preds, target

    @staticmethod
    def _to_float_dict(metrics: dict[str, torch.Tensor]) -> dict[str, float]:
        """
        Convert 0-dim tensors to Python floats to avoid Tensor __repr__ / pretty-printing.
        """
        out = {}
        for k, v in metrics.items():
            # v is expected to be a scalar tensor
            out[k] = float(v.detach().cpu().item())
        return out

    def train_loss(self, preds: torch.Tensor, target: torch.Tensor):
        preds, target = self._standardize_shapes(preds, target)
        mse = torch.nn.functional.mse_loss(preds, target)
        metrics = {"mse": mse}
        weights = [1.0]
        return metrics, weights

    def train_metrics(self, preds: torch.Tensor, target: torch.Tensor):
        preds, target = self._standardize_shapes(preds, target)
        self._ensure_device(preds)
        rrse = self._rrse(preds, target)
        metrics = {"rrse": rrse}
        weights = [1.0]
        return metrics, weights

    def val_metrics(self, preds: torch.Tensor, target: torch.Tensor):
        preds, target = self._standardize_shapes(preds, target)

        mse = torch.nn.functional.mse_loss(preds, target)
        self._ensure_device(preds)
        rrse = self._rrse(preds, target)

        metrics = {"mse": mse, "rrse": rrse}
        weights = [1.0, 1.0]
        return metrics, weights

    def __call__(self, preds: torch.Tensor, target: torch.Tensor):
        """
        Returns:
          - If silent=False: (dict[str, float], list[float])
          - If silent=True: a short string (so Jupyter prints only a string)
        """
        mode = self.mode.lower()

        if mode == "train_loss":
            metrics, weights = self.train_loss(preds, target)

        elif mode == "train_metrics":
            with torch.no_grad():
                metrics, weights = self.train_metrics(preds, target)

        elif mode == "val_metrics":
            with torch.no_grad():
                metrics, weights = self.val_metrics(preds, target)

        else:
            raise NotImplementedError(f"{self.mode} is not implemented as a valid metric case.")

        # Convert tensors -> floats BEFORE returning to avoid torch Tensor printing.
        metrics_float = self._to_float_dict(metrics)

        if self.silent:
            # Return a string to ensure the cell output is benign and never touches torch Tensor __repr__.
            keys = ", ".join(metrics_float.keys())
            return f"{self.__class__.__name__}({self.mode}) computed: {keys}"

        return metrics_float, weights
