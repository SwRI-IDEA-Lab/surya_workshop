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

    def _ensure_device(self, preds: torch.Tensor):
        # Move metric module to the same device as preds, but only when needed
        if self._rrse.device != preds.device:
            self._rrse = self._rrse.to(preds.device)        

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

        output_metrics = {}
        output_weights = []

        output_metrics["mse"] = torch.nn.functional.mse_loss(preds, target.squeeze(-1))
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
        output_metrics = {}
        output_weights = []

        self._ensure_device(preds)
        output_metrics["rrse"] = self._rrse(preds, target.squeeze(-1))
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

        output_metrics = {}
        output_weights = []

        output_metrics["mse"] = torch.nn.functional.mse_loss(preds, target.squeeze(-1))
        output_weights.append(1)

        self._ensure_device(preds)
        output_metrics["rrse"] = self._rrse(preds, target.squeeze(-1))
        output_weights.append(1)            

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
