"""
The EUV2MAG Lightning modules.

The training mechanics live in ``workshop_infrastructure/lightning_modules/pl_base.py``
and are imported, not copied. All these subclasses supply is the target shape.

  ``Euv2MagLightningModule``          plain Adam. Used by the Conv1x1 baseline.
  ``Euv2MagFinetuneLightningModule``  two parameter groups, so the randomly-initialized
                                      decoder head and the near-identity LoRA adapters
                                      train at different rates. Used for fine-tuning.

The trainable tokenizer (see ``models/finetune_model.py``) is not a ``head_*`` module, so
it lands in the backbone group at the base learning rate — which is what it wants, having
started from pretrained weights.
"""

from __future__ import annotations

from typing import Dict

import torch

from workshop_infrastructure.lightning_modules.pl_base import (
    SuryaFinetuneLightningModule,
    SuryaLightningModule,
)


def euv2mag_target(batch: Dict) -> torch.Tensor:
    """The magnetogram target as ``(B, C, H, W)``.

    ``Euv2MagDataset`` returns ``forecast`` as ``(B, C, L, H, W)``, where ``L`` is the lead
    time axis. This task predicts a single frame, so ``L == 1`` and squeezing axis 2 is
    exact. ``squeeze(2)`` rather than a bare ``squeeze()``: the latter would also collapse
    a single-channel or single-sample axis, silently changing the shape for
    ``target_channels`` of length 1 or a final batch of one.

    Predicting several lead times would mean folding ``L`` into the channel axis here and
    widening the decoder to match, not squeezing it away.
    """
    return batch["forecast"].squeeze(2).float()


class Euv2MagLightningModule(SuryaLightningModule):
    """``SuryaLightningModule`` with the image target shape filled in."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("target_fn", euv2mag_target)
        super().__init__(*args, **kwargs)


class Euv2MagFinetuneLightningModule(SuryaFinetuneLightningModule):
    """``SuryaFinetuneLightningModule`` with the image target shape filled in."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("target_fn", euv2mag_target)
        super().__init__(*args, **kwargs)
