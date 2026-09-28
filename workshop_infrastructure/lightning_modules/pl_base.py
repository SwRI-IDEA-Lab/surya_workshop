"""
The Lightning modules every downstream app shares.

Two classes, differing only in ``configure_optimizers``:

  ``SuryaLightningModule``          plain Adam over everything trainable. What a linear
                                    baseline wants.
  ``SuryaFinetuneLightningModule``  two parameter groups, so a randomly-initialized head
                                    and near-identity LoRA adapters can train at
                                    different rates. What fine-tuning Surya wants.

Everything else -- the batch contract, the loss/metric dicts, the logging -- is the same
for every task, which is why it lives here instead of being copied into each app.

The one thing that is *not* the same is the shape of the target, so it is the one thing
an app must supply: ``target_fn`` maps the batch dict to the tensor the metrics compare
predictions against. There is no default, because a silently wrong target shape does not
raise -- it broadcasts, and trains against nonsense. The two shipped examples:

    scalar label per sample   lambda b: b["forecast"].unsqueeze(1).float()   (B, 1)
    image target             lambda b: b["forecast"].squeeze(2).float()      (B, C, H, W)

Key batch contract:
  - batch["ts"]       : torch.Tensor input stack, (B, C, T, H, W)
  - batch["forecast"] : the target, in whatever shape the dataset produces; ``target_fn``
                        is what turns it into the shape the metrics expect

Optional preprocessing:
  - If ``preprocess_fn`` is provided, it is called on the batch dict before every model
    call. This is the hook for input transformations (such as inverse-normalizing SDO
    channels) that should not live inside the model.

Key metrics contract (the `metrics` dict passed to __init__):
  - metrics["train_loss"]    : callable(output, target) -> (loss_dict, weight_list)
        Backpropagated. Logged as "train_loss".
  - metrics["val_loss"]      : callable(output, target) -> (loss_dict, weight_list)
        Optional. Logged as "val_loss" and therefore what ModelCheckpoint monitors.
        Falls back to metrics["train_loss"] when absent.
  - metrics["train_metrics"] : callable(output, target) -> (metric_dict, weight_list)
  - metrics["val_metrics"]   : callable(output, target) -> (metric_dict, weight_list)
        Reported only. These do NOT affect checkpoint selection -- "val_loss" does.

Where:
  - loss_dict / metric_dict map string names -> torch scalar tensors
  - weight_list is a list-like of floats aligned with the dict iteration order, used to
    form a weighted sum loss.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Optional, Tuple

import lightning as L
import torch


# Type aliases for clarity in documentation / teaching.
LossDict = Mapping[str, torch.Tensor]
MetricDict = Mapping[str, torch.Tensor]
Weights = Any  # often a list[float] or list[torch.Tensor]


class SuryaLightningModule(L.LightningModule):
    """
    PyTorch LightningModule for a Surya downstream task.

    This class wraps:
      (1) a user-provided PyTorch model (nn.Module-like) and
      (2) a set of loss/metric callables packaged in the `metrics` dictionary.

    Parameters
    ----------
    model:
        A callable model (typically torch.nn.Module) that accepts the batch input tensor
        `x = batch["ts"]` and returns predictions `output`.

    metrics:
        Dictionary containing the training loss function and metric functions.

        Required keys:
          - "train_loss": callable(output, target) -> (losses, weights)
              losses: dict[str, torch.Tensor] scalar losses
              weights: list-like aligned with iteration order of losses.keys()
          - "train_metrics": callable(output, target) -> (metrics, weights)
          - "val_metrics": callable(output, target) -> (metrics, weights)

        Optional key:
          - "val_loss": callable(output, target) -> (losses, weights)
              The validation objective. Defaults to "train_loss" when not supplied, so
              older metrics dicts keep working unchanged.

        The module uses:
          - train_loss in training_step, backpropagated and logged as "train_loss"
          - val_loss in validation_step, logged as "val_loss" — the quantity
            ModelCheckpoint monitors
          - train_metrics logged during training_step (if weights is non-empty)
          - val_metrics logged during validation_step (if weights is non-empty).
            Reported only; they do not influence checkpoint selection.

    lr:
        Learning rate for the Adam optimizer.

    batch_size:
        Optional batch size passed to Lightning's `self.log(..., batch_size=...)`.
        This improves correct averaging behavior when using distributed settings
        or variable batch sizes.

    target_fn:
        Callable mapping the batch dict to the target tensor, e.g.
        ``lambda b: b["forecast"].squeeze(2).float()``. Required: see the module
        docstring for why there is no default.

    preprocess_fn:
        Optional callable applied to the batch dict before every model call.
        Signature: ``(batch: dict) -> dict``. Use this to apply input
        transformations (e.g., ``destandardize_channels``) without
        embedding them in the model itself.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        metrics: Dict[str, Callable[..., Tuple[Dict[str, torch.Tensor], Weights]]],
        lr: float,
        target_fn: Callable[[Dict], torch.Tensor],
        batch_size: Optional[int] = None,
        preprocess_fn: Optional[Callable[[Dict], Dict]] = None,
    ):
        super().__init__()
        self.target_fn = target_fn
        self.batch_size = batch_size
        self.model = model
        self.preprocess_fn = preprocess_fn

        # Loss callables: return (loss_dict, weight_list)
        self.training_loss = metrics["train_loss"]
        # "val_loss" is optional: falling back to train_loss keeps a metrics dict written
        # before this key existed working, with identical behavior.
        self.validation_loss = metrics.get("val_loss", metrics["train_loss"])

        # Metric callables: return (metric_dict, weight_list)
        self.training_evaluation = metrics["train_metrics"]
        self.validation_evaluation = metrics["val_metrics"]

        self.lr = lr

    @staticmethod
    def _combine_losses(loss_dict: LossDict, weights: Weights) -> torch.Tensor:
        """Return a weighted sum of the losses in ``loss_dict``.

        ``weights`` must be aligned with ``loss_dict.keys()`` iteration order.
        Raises ``ValueError`` if ``loss_dict`` is empty.
        """
        loss = None
        for n, key in enumerate(loss_dict.keys()):
            component = loss_dict[key] * weights[n]
            loss = component if loss is None else (loss + component)
        if loss is None:
            raise ValueError("loss_dict is empty; cannot compute a scalar loss.")
        return loss

    def forward(self, batch: dict) -> torch.Tensor:
        """
        Forward pass used by Lightning and by explicit calls in steps.

        Parameters
        ----------
        batch:
            Batch dict (at minimum contains ``"ts"`` and ``"forecast"``).

        Returns
        -------
        torch.Tensor
            Model predictions for the batch.
        """
        return self.model(batch)

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """
        Runs one training step on a single batch.

        Workflow
        --------
        1) Extract inputs and targets from the batch:
              x = batch["ts"]
              target = batch["forecast"]
        2) Compute model output:
              output = self(x)
        3) Compute per-component losses and combine via provided weights:
              training_losses, training_loss_weights = training_loss(output, target)
        4) Log:
              - total weighted loss as "train_loss" (progress bar)
              - each component loss as "train_loss_<name>"
              - training metrics as "train_metric_<name>" (if any)

        Notes
        -----
        - The target comes from ``target_fn``, which the app supplies, because its
          shape is the one part of this loop that is task-specific.
        - The loss combination depends on dict iteration order; ensure loss dict
          insertion order is consistent if that matters.

        Returns
        -------
        torch.Tensor
            The scalar training loss used for backpropagation.
        """
        target = self.target_fn(batch)

        if self.preprocess_fn is not None:
            batch = self.preprocess_fn(batch)
        output = self(batch)
        training_losses, training_loss_weights = self.training_loss(output, target)
        loss = self._combine_losses(training_losses, training_loss_weights)

        # Log aggregate loss and component losses.
        self.log("train_loss", loss, prog_bar=True, batch_size=self.batch_size, sync_dist=True)
        for key in training_losses.keys():
            self.log(f"train_loss_{key}", training_losses[key], prog_bar=False, batch_size=self.batch_size, sync_dist=True)

        # Log evaluation metrics (optional).
        training_evaluation_metrics, training_evaluation_weights = self.training_evaluation(output, target)
        if len(training_evaluation_weights) > 0:
            for key in training_evaluation_metrics.keys():
                self.log(f"train_metric_{key}", training_evaluation_metrics[key], prog_bar=False, batch_size=self.batch_size, sync_dist=True)

        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        """
        Runs one validation step on a single batch.

        Workflow
        --------
        1) Extract inputs and targets
        2) Compute output
        3) Compute validation losses and combine via weights
        4) Log:
              - total weighted loss as "val_loss" (progress bar)
              - each component loss as "val_loss_<name>"
              - validation metrics as "val_metric_<name>" (if any)

        Notes
        -----
        - The loss is computed with `self.validation_loss`, which comes from
          metrics["val_loss"] and falls back to metrics["train_loss"] when that key is
          absent. Supply a distinct "val_loss" callable to monitor something other than
          the training objective.
        - "val_loss" is what ModelCheckpoint monitors. The `val_metrics` logged at the end
          of this method are reported only and do not affect checkpoint selection.
        - No value is returned (Lightning uses logs for validation tracking).
        """
        target = self.target_fn(batch)

        if self.preprocess_fn is not None:
            batch = self.preprocess_fn(batch)
        output = self(batch)
        val_losses, val_loss_weights = self.validation_loss(output, target)
        loss = self._combine_losses(val_losses, val_loss_weights)

        # Log aggregate loss and component losses.
        self.log("val_loss", loss, prog_bar=True, batch_size=self.batch_size, sync_dist=True)
        for key in val_losses.keys():
            self.log(f"val_loss_{key}", val_losses[key], prog_bar=False, batch_size=self.batch_size, sync_dist=True)

        # Log evaluation metrics (optional).
        val_evaluation_metrics, val_evaluation_weights = self.validation_evaluation(output, target)
        if len(val_evaluation_weights) > 0:
            for key in val_evaluation_metrics.keys():
                self.log(f"val_metric_{key}", val_evaluation_metrics[key], prog_bar=False, batch_size=self.batch_size, sync_dist=True)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Configure the optimizer used by Lightning.

        Returns
        -------
        torch.optim.Optimizer
            Adam optimizer over all module parameters with learning rate `self.lr`.
        """
        return torch.optim.Adam(self.parameters(), lr=self.lr)


class SuryaFinetuneLightningModule(SuryaLightningModule):
    """``SuryaLightningModule`` with a two-group optimizer for LoRA fine-tuning.

    Under LoRA the model holds two populations of trainable parameters:

      * adapters (``lora_A`` / ``lora_B``), which start as a near-identity perturbation
        of a backbone that already works, and
      * the head, which starts *random* and has to learn the task from nothing.

    One learning rate has to serve both. Set it low enough for the adapters and the head
    crawls; set it high enough for the head and the adapters shove the pretrained
    features around in the first few steps -- on a few hundred samples, that is most of
    your budget spent undoing the pretraining. Giving the head a larger rate is the
    standard fix, and ``head_lr_multiplier`` is how you dial it.

    Weight decay is applied to the adapters only. Decaying the head pulls a freshly
    initialized read-out toward zero while it is still finding a scale; decaying the
    adapters pulls them toward zero, which means toward the *pretrained* weights -- a
    sensible prior when you have few samples.

    A backbone layer kept trainable via ``model.trainable_backbone_modules`` -- a
    tokenizer rebuilt for a different channel count, say -- lands in the backbone group,
    because ``_is_head_parameter`` matches the ``head_`` prefix and such a layer does not
    carry it. That is the right group: it starts from pretrained weights, so it wants the
    base rate rather than the multiplied one.

    Set ``head_lr_multiplier=1.0`` and ``weight_decay=0.0`` to recover plain Adam over
    everything, i.e. exactly what ``SuryaLightningModule`` does.

    Additional Args (everything else is inherited):
        head_lr_multiplier: The head trains at ``lr * head_lr_multiplier``. 1.0 disables
            the split.
        weight_decay: Applied to non-head trainable parameters (the LoRA adapters, or the
            whole backbone under full fine-tuning). Never applied to the head.
    """

    def __init__(self, *args, head_lr_multiplier: float = 10.0, weight_decay: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.head_lr_multiplier = head_lr_multiplier
        self.weight_decay = weight_decay

    @staticmethod
    def _is_head_parameter(name: str) -> bool:
        """True for parameters belonging to a ``head_*`` module.

        The name is matched on a dot boundary, because under PEFT the head is reached
        through wrapper attributes and the path looks like
        ``model.base_model.model.head_unembed.modules_to_save.default.weight``. A bare
        substring test would be nearly as accurate and would quietly also match a
        backbone module that happened to end in ``head_``.
        """
        return any(part.startswith("head_") for part in name.split("."))

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Adam over two parameter groups: the head, and everything else trainable.

        Only parameters with ``requires_grad`` are handed to the optimizer. Under LoRA
        that is the adapters and the head; under a linear probe it is the head alone, and
        the backbone group comes out empty (which Adam accepts).
        """
        head, backbone = [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            (head if self._is_head_parameter(name) else backbone).append(param)

        if not head and not backbone:
            raise RuntimeError(
                "No trainable parameters. Check model.use_lora / model.freeze_backbone in "
                "the config, and that every head layer is named head_* (see "
                "discover_head_modules in workshop_infrastructure/utils.py)."
            )

        head_lr = self.lr * self.head_lr_multiplier
        print(
            f"[OPTIM] Adam | head: {sum(p.numel() for p in head):,} params @ lr={head_lr:g} | "
            f"backbone/adapters: {sum(p.numel() for p in backbone):,} params @ lr={self.lr:g}, "
            f"weight_decay={self.weight_decay:g}"
        )
        return torch.optim.Adam(
            [
                {"params": head, "lr": head_lr, "weight_decay": 0.0},
                {"params": backbone, "lr": self.lr, "weight_decay": self.weight_decay},
            ]
        )
