import torch
# import torchmetrics as tm  # Lots of possible metrics in here https://lightning.ai/docs/torchmetrics/stable/all-metrics.html
import torch.nn.functional as F

"""
Template metrics to be used for flare forecasting.  Within the FlareMetrics class,
different methods are defined to calculate metrics for training loss, as well as evaluation
metrics to report during training, and validation. The __call__ method allows for easy selection
of the appropriate metric set based on the provided mode.

The loss names used in the dictionary keys are propagated during the logging.
"""

class FlareMetrics:
    def __init__(self, mode: str):
        self.mode = mode
 
    # CROSS METRICS
    def train_loss(
        self, preds: torch.Tensor, target: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], list[float]]:
        
        ce = F.cross_entropy(preds, target)

        output_metrics = {"cross_entropy": ce}
        output_weights = [1]

        return output_metrics, output_weights

    # METRICS micro F1

    def train_metrics(
        self, preds: torch.Tensor, target: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], list[float]]:
        
        pred_classes = preds.argmax(dim=1)

        accuracy = (pred_classes == target).float().mean()

        tp = (pred_classes * target).sum()
        fp = (pred_classes != target).sum() 
        fn = fp      

        # Micro precision, recall, F1
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        micro_f1 = 2 * precision * recall / (precision + recall + 1e-8)

        output_metrics = {
            "accuracy": accuracy,
            "micro_f1": micro_f1
        }
        output_weights = [1, 1]

        return output_metrics, output_weights

    def val_metrics(
        self, preds: torch.Tensor, target: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], list[float]]:
        
        output_metrics, output_weights = self.train_metrics(preds, target)

        ce = F.cross_entropy(preds, target)
        output_metrics["cross_entropy"] = ce
        output_weights.append(1)

        return output_metrics, output_weights

    def __call__(
        self, preds: torch.Tensor, target: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], list[float]]:
        
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
