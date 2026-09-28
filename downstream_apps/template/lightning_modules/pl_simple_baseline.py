"""
The flare template's Lightning modules.

The training mechanics -- the batch contract, the loss and metric dicts, the logging, the
optimizer groups -- are the same for every downstream task, so they live in
``workshop_infrastructure/lightning_modules/pl_base.py`` and are imported, not copied.

What is task-specific is the shape of the target, and that is all these two subclasses
supply. The flare label is one scalar per sample, so it becomes ``(B, 1)``.

  ``FlareLightningModule``          plain Adam. Used by the linear baseline.
  ``FlareFinetuneLightningModule``  two parameter groups, so the randomly-initialized
                                    head and the near-identity LoRA adapters train at
                                    different rates. Used for fine-tuning; see the base
                                    class for why that split matters.
"""

from __future__ import annotations

from typing import Dict

import torch

from workshop_infrastructure.lightning_modules.pl_base import (
    SuryaFinetuneLightningModule,
    SuryaLightningModule,
)


def flare_target(batch: Dict) -> torch.Tensor:
    """The flare label as ``(B, 1)``.

    ``FlareDSDataset`` puts one scalar per sample in ``batch["forecast"]``, so the batch
    arrives as ``(B,)``. The metrics flatten both sides with ``reshape(-1)``, but the
    unsqueeze is kept because it is the shape the linear baseline's output matches.
    """
    return batch["forecast"].unsqueeze(1).float()


class FlareLightningModule(SuryaLightningModule):
    """``SuryaLightningModule`` with the flare target shape filled in."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("target_fn", flare_target)
        super().__init__(*args, **kwargs)


class FlareFinetuneLightningModule(SuryaFinetuneLightningModule):
    """``SuryaFinetuneLightningModule`` with the flare target shape filled in."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("target_fn", flare_target)
        super().__init__(*args, **kwargs)
