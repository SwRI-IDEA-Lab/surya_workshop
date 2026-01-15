import torch
import torchmetrics as tm  # Lots of possible metrics in here https://lightning.ai/docs/torchmetrics/stable/all-metrics.html

"""
Template metrics to be used for flare forecasting.  Within the FlareMetrics class,
different methods are defined to calculate metrics for training loss, as well as evaluation
metrics to report during training, and validation. The __call__ method allows for easy selection
of the appropriate metric set based on the provided mode.

The loss names used in the dictionary keys are propagated during the logging.
"""

class FlareMetrics:
    def __init__(self, mode: str):
        """
        Initialize FlareMetrics class.

        Args:
            mode (str): Mode to use for metric evaluation. Can be "train_loss",
                        "train_metrics", or "val_metrics".
        """
        self.mode = mode

        # Cache torchmetrics instances once (instead of recreating each call)
        self._rrse = tm.RelativeSquaredError(squared=False)

        # MRE via TorchMetrics: MeanAbsolutePercentageError
        # NOTE: returns a fraction (0.08 means 8%), not multiplied by 100.
        self._mape = tm.MeanAbsolutePercentageError()

    def _ensure_device(self, preds: torch.Tensor):
        # Move metric module to the same device as preds, but only when needed
        if self._rrse.device != preds.device:
            self._rrse = self._rrse.to(preds.device)    
             # MRE via TorchMetrics: MeanAbsolutePercentageError
            self._mape = self._mape.to(preds.device) 
                

    #added forRuntimeError: The size of tensor a (1280) must match the size of tensor b (2) at non-singleton dimension 1, 8pm
    def _reduce_preds(self, preds: torch.Tensor) -> torch.Tensor:
        # Convert to (B,1) if model returns sequences
        if preds.ndim == 3:
            preds = preds.mean(dim=1)  # (B,L,D)->(B,D)
        if preds.ndim == 2 and preds.shape[1] != 1:
            preds = preds.mean(dim=1, keepdim=True)  # (B,L)->(B,1)
        elif preds.ndim == 1:
            preds = preds.unsqueeze(1)
        return preds
    

    def _maybe_mask_zeros(
        self,
        preds_1d: torch.Tensor,
        target_1d: torch.Tensor,
        eps: float = 1e-8
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        MAPE/MRE can blow up if targets contain zeros or near-zeros.
        This masks those entries for stability.
        """
        mask = torch.abs(target_1d) > eps
        # If everything is masked (all targets ~0), fall back to original tensors
        if mask.any():
            return preds_1d[mask], target_1d[mask]
        return preds_1d, target_1d


    def train_loss(
        self, preds: torch.Tensor, target: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], list[float]]:
        """
        Calculate loss metrics for training.

        Args:
            preds (torch.Tensor): Model predictions.
            target (torch.Tensor): Ground truth labels.

        Returns:
            tuple[dict[str, torch.Tensor], list[float]]:
                - dict[str, torch.Tensor]: Dictionary containing the calculated loss metrics.
                                        Keys are metric names (e.g., "mse"), and values are the
                                        corresponding torch.Tensor values.
                - list[float]: List of weights for each calculated metric.
        """
        #added forRuntimeError: The size of tensor a (1280) must match the size of tensor b (2) at non-singleton dimension 1, 8pm
        preds = self._reduce_preds(preds)

        output_metrics = {}
        output_weights = []

        output_metrics["mse"] = torch.nn.functional.mse_loss(preds.squeeze(-1), target.squeeze(-1))
        output_weights.append(1)

        return output_metrics, output_weights

    def train_metrics(
        self, preds: torch.Tensor, target: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], list[float]]:
        """
        Calculate evaluation metrics for training.
        IMPORTANT:  These metrics are only for reporting purposes and do not
                    contribute to the training loss. Use only if you want to
                    monitor additional metrics during training.

        Args:
            preds (torch.Tensor): Model predictions.
            target (torch.Tensor): Ground truth labels.

        Returns:
            tuple[dict[str, torch.Tensor], list[float]]:
                - dict[str, torch.Tensor]: Dictionary containing the calculated evaluation metrics.
                                        Keys are metric names, and values are the corresponding torch.Tensor values.
                - list[float]: List of weights for each calculated metric.
        """
        #added forRuntimeError: The size of tensor a (1280) must match the size of tensor b (2) at non-singleton dimension 1, 8pm
        preds = self._reduce_preds(preds)

        output_metrics = {}
        output_weights = []

        self._ensure_device(preds)
        output_metrics["rrse"] = self._rrse(preds.squeeze(-1), target.squeeze(-1))
        output_weights.append(1)        


        return output_metrics, output_weights

    def val_metrics(
        self, preds: torch.Tensor, target: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], list[float]]:
        """
        Calculate metrics for validation.

        Args:
            preds (torch.Tensor): Model predictions.
            target (torch.Tensor): Ground truth labels.

        Returns:
            tuple[dict[str, torch.Tensor], list[float]]:
                - dict[str, torch.Tensor]: Dictionary containing the calculated metrics.
                                        Keys are metric names (e.g., "mse"), and values are the
                                        corresponding torch.Tensor values.
                - list[float]: List of weights for each calculated metric.
        """
        #added forRuntimeError: The size of tensor a (1280) must match the size of tensor b (2) at non-singleton dimension 1, 8pm
        preds = self._reduce_preds(preds)

        output_metrics = {}
        output_weights = []

        output_metrics["mse"] = torch.nn.functional.mse_loss(preds.squeeze(-1), target.squeeze(-1))
        output_weights.append(1)

        self._ensure_device(preds)
        output_metrics["rrse"] = self._rrse(preds.squeeze(-1), target.squeeze(-1))
        output_weights.append(1)  

        # # MRE implemented via TorchMetrics MeanAbsolutePercentageError (fraction)
        # p_mre, t_mre = self._maybe_mask_zeros(preds_, target_)
        # output_metrics["MAPE"] = self._mre(p_mre, t_mre)
        # output_weights.append(1)          
        # output_metrics["mape"] = self._mape(preds.squeeze(-1), target.squeeze(-1))
        # output_weights.append(1)

        return output_metrics, output_weights

    def __call__(
        self, preds: torch.Tensor, target: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], list[float]]:
        """
        Default method to evaluated all metrics.

        Parameters
        ----------
        preds : torch.Tensor
            Output target of the AI model. Shape depends on the application.
        target : torch.Tensor
            Ground truth to compare AI model output against

        Returns
        -------
        dict
            Dictionary with all metrics. Metrics aggregate over the batch. So the
            dicationary takes the shape [str, torch.Tensor] with the tensors having
            shape [].
        list
            List of weights for each calculated metric to enable giving a different
            weight to each loss term.
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
