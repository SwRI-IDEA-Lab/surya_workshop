"""
The LightningModule for the fine-tuning path.

``FlareLightningModule`` (in ``pl_simple_baseline.py``) owns the training mechanics that
are the same for every model here: the batch contract, the loss/metric dicts, the
logging. This subclass changes exactly one thing — ``configure_optimizers`` — because
fine-tuning a pretrained backbone and training a linear baseline want different
optimizers, and that difference is worth seeing rather than hiding behind a flag.

**The problem it solves.** Under LoRA the model holds two populations of parameters:

  * adapters (``lora_A`` / ``lora_B``), which start at a near-identity perturbation of a
    backbone that already works, and
  * the head, which starts *random* and has to learn the task from nothing.

One learning rate has to serve both. Set it low enough for the adapters and the head
crawls; set it high enough for the head and the adapters shove the pretrained features
around in the first few steps — on a few hundred samples, that is most of your budget
spent undoing the pretraining. Giving the head a larger rate is the standard fix, and
``head_lr_multiplier`` is how you dial it.

Weight decay is applied to the adapters only. Decaying the head pulls a freshly
initialized read-out toward zero while it is still trying to find a scale; decaying the
adapters pulls them toward zero, which means toward the *pretrained* weights — a
sensible prior when you have few samples.

Set ``head_lr_multiplier=1.0`` and ``weight_decay=0.0`` to recover plain Adam over
everything, i.e. exactly what the baseline module does.

**A trainable backbone layer lands in the backbone group, not the head group.** An app
that keeps part of the backbone trainable via ``model.trainable_backbone_modules`` -- a
tokenizer rebuilt for a different channel count, say -- gets it at the base learning
rate with weight decay, because ``_is_head_parameter`` matches on the ``head_`` prefix
and such a layer does not carry it. That is the right group for it: it starts from
pretrained weights, so it wants the rate the rest of the pretrained model gets, not the
10x rate a randomly-initialized read-out needs.
"""

from __future__ import annotations

import torch

from downstream_apps.template.lightning_modules.pl_simple_baseline import FlareLightningModule


class FlareFinetuneLightningModule(FlareLightningModule):
    """``FlareLightningModule`` with a two-group optimizer for LoRA fine-tuning.

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
