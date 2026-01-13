import torch
import torchmetrics as tm  # Lots of possible metrics in here https://lightning.ai/docs/torchmetrics/stable/all-metrics.html

"""
Metrics for ribbon segmentation.  Within the RibbonSegmentationMetrics class,
different methods are defined to calculate metrics for training loss, as well as evaluation
metrics to report during training, and validation. The __call__ method allows for easy selection
of the appropriate metric set based on the provided mode.

The loss names used in the dictionary keys are propagated during the logging.
"""

class RibbonSegmentationMetrics:
    def __init__(self, mode: str, threshold: float = 0.5):
        """
        Initialize RibbonSegmentationMetrics class.

        Args:
            mode (str): Mode to use for metric evaluation. Can be "train_loss",
                        "train_metrics", or "val_metrics".
            threshold (float): Threshold for converting predictions to binary masks.
                              Default is 0.5.
        """
        self.mode = mode
        self.threshold = threshold

        # Cache torchmetrics instances once (instead of recreating each call)
        # For binary segmentation
        self._iou = tm.JaccardIndex(task="binary", threshold=threshold)

        self._precision = tm.Precision(task="binary", threshold=threshold)
        self._recall = tm.Recall(task="binary", threshold=threshold)
        self._f1 = tm.F1Score(task="binary", threshold=threshold)

    def _ensure_device(self, preds: torch.Tensor):
        # Move metric modules to the same device as preds, but only when needed
        if self._iou.device != preds.device:
            self._iou = self._iou.to(preds.device)
            self._precision = self._precision.to(preds.device)
            self._recall = self._recall.to(preds.device)
            self._f1 = self._f1.to(preds.device)        

    def train_loss(
        self, preds: torch.Tensor, target: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], list[float]]:
        """
        Calculate loss metrics for training.

        Args:
            preds (torch.Tensor): Model predictions (logits or probabilities).
                                 Shape: [batch_size, height, width] or [batch_size, 1, height, width]
            target (torch.Tensor): Ground truth binary masks.
                                  Shape: [batch_size, height, width] or [batch_size, 1, height, width]

        Returns:
            tuple[dict[str, torch.Tensor], list[float]]:
                - dict[str, torch.Tensor]: Dictionary containing the calculated loss metrics.
                                        Keys are metric names, and values are the
                                        corresponding torch.Tensor values.
                - list[float]: List of weights for each calculated metric.
        """

        output_metrics = {}
        output_weights = []

        # Binary Cross Entropy Loss (common for segmentation)
        # Convert target to float if it's not already (e.g., if it's uint8/Byte)
        target = target.float()
        output_metrics["bce"] = torch.nn.functional.binary_cross_entropy_with_logits(preds, target)
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
            preds (torch.Tensor): Model predictions (logits or probabilities).
                                 Shape: [batch_size, height, width] or [batch_size, 1, height, width]
            target (torch.Tensor): Ground truth binary masks.
                                  Shape: [batch_size, height, width] or [batch_size, 1, height, width]

        Returns:
            tuple[dict[str, torch.Tensor], list[float]]:
                - dict[str, torch.Tensor]: Dictionary containing the calculated evaluation metrics.
                                        Keys are metric names, and values are the corresponding torch.Tensor values.
                - list[float]: List of weights for each calculated metric.
        """
        output_metrics = {}
        output_weights = []

        self._ensure_device(preds)

        # Apply sigmoid to convert logits to probabilities if needed
        if preds.min() < 0 or preds.max() > 1:
            preds = torch.sigmoid(preds)

        # IoU (Intersection over Union) - primary metric for segmentation
        output_metrics["iou"] = self._iou(preds, target.int())
        output_weights.append(1)

        return output_metrics, output_weights

    def val_metrics(
        self, preds: torch.Tensor, target: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], list[float]]:
        """
        Calculate metrics for validation.

        Args:
            preds (torch.Tensor): Model predictions (logits or probabilities).
                                 Shape: [batch_size, height, width] or [batch_size, 1, height, width]
            target (torch.Tensor): Ground truth binary masks.
                                  Shape: [batch_size, height, width] or [batch_size, 1, height, width]

        Returns:
            tuple[dict[str, torch.Tensor], list[float]]:
                - dict[str, torch.Tensor]: Dictionary containing the calculated metrics.
                                        Keys are metric names, and values are the
                                        corresponding torch.Tensor values.
                - list[float]: List of weights for each calculated metric.
        """

        output_metrics = {}
        output_weights = []

        self._ensure_device(preds)

        # Apply sigmoid to convert logits to probabilities if needed
        if preds.min() < 0 or preds.max() > 1:
            preds = torch.sigmoid(preds)

        # Binary Cross Entropy Loss
        # Convert target to float if it's not already (e.g., if it's uint8/Byte)
        target = target.float()
        output_metrics["bce"] = torch.nn.functional.binary_cross_entropy(preds, target)
        output_weights.append(1)

        # IoU (Intersection over Union) - most common segmentation metric
        output_metrics["iou"] = self._iou(preds, target.int())
        output_weights.append(1)

        # Precision - ratio of true positives to predicted positives
        output_metrics["precision"] = self._precision(preds, target.int())
        output_weights.append(1)

        # Recall - ratio of true positives to actual positives
        output_metrics["recall"] = self._recall(preds, target.int())
        output_weights.append(1)

        # F1 Score - harmonic mean of precision and recall
        output_metrics["f1"] = self._f1(preds, target.int())
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
