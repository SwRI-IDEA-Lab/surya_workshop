"""
Task-specific configuration for the EUV2MAG image-translation app.

Everything generic -- paths, channels, temporal sampling, S3 settings, the model and LoRA
configs, the training and logging sections, and ``load_config()`` itself -- lives in
``workshop_infrastructure/configs.py``. This file holds only what is specific to *this*
task: which channels are inputs, which are targets, and the channel order the pretrained
checkpoint was trained with.

**Three channel lists, and mixing them up is silent.** They answer different questions:

    channels                   what the dataset reads out of each NetCDF file
    input_channels             which of those the model takes as input
    target_channels            which of those the model is asked to predict
    pretrained_channel_order   the 13 channels the checkpoint was pretrained with

``channels`` is the union of inputs and targets, and is deliberately shorter than 13 --
there is no reason to decode nine channels per sample and throw them away.

``pretrained_channel_order`` is the odd one out: it is not about this app's data at all.
It is the order the checkpoint's patch-embedding convolution was trained in, and it is
what the tokenizer slice indexes into (see ``adapt_patch_embed_weight``). Using
``channels`` there instead would load some other channel's pretrained tokenizer, with no
error and no sign of it in the logs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from typing import ClassVar

from workshop_infrastructure.configs import (  # re-exported for convenience
    DataConfig,
    LoraAdapterConfig,
    ModelConfig,
    OutputConfig,
    TimeEmbeddingConfig,
    TrainingConfig,
    load_config,
)

# The channel order the Surya 366M checkpoint was pretrained with. This is a property of
# the checkpoint, not of any app, which is why it has a default and rarely needs setting.
#
# Note it is NOT the key order in assets/scalers.yaml, where hmi_bx/by/bz precede hmi_m.
# The scalers are looked up by name so their order does not matter; the tokenizer slice
# is positional, so this one does.
SURYA_PRETRAINED_CHANNEL_ORDER = [
    "aia94", "aia131", "aia171", "aia193", "aia211", "aia304", "aia335", "aia1600",
    "hmi_m", "hmi_bx", "hmi_by", "hmi_bz", "hmi_v",
]


@dataclass
class Euv2MagDataConfig(DataConfig):
    """DataConfig plus the input/target channel split used by ``Euv2MagDataset``."""

    # EUV channels fed to the model, in the order the model's tokenizer expects them.
    input_channels: list[str] = field(
        default_factory=lambda: ["aia304", "aia193", "aia171"]
    )
    # What the model predicts. One channel for a line-of-sight magnetogram; could be
    # ["hmi_bx", "hmi_by", "hmi_bz"] for a vector field.
    target_channels: list[str] = field(default_factory=lambda: ["hmi_m"])
    # The pretraining channel order the tokenizer slice indexes into. Only change this if
    # you are loading a checkpoint pretrained on a different channel set.
    pretrained_channel_order: list[str] = field(
        default_factory=lambda: list(SURYA_PRETRAINED_CHANNEL_ORDER)
    )

    # No new path fields, so PATH_FIELDS is inherited unchanged. Declared explicitly so a
    # future path field has an obvious place to go.
    PATH_FIELDS: ClassVar[tuple[str, ...]] = DataConfig.PATH_FIELDS

    def __post_init__(self) -> None:
        super().__post_init__()

        overlap = sorted(set(self.input_channels) & set(self.target_channels))
        if overlap:
            raise ValueError(
                f"data.input_channels and data.target_channels both contain {overlap}. "
                "The model would be given the answer as an input, and the validation "
                "loss would look excellent for the wrong reason."
            )
        for name, values in (
            ("input_channels", self.input_channels),
            ("target_channels", self.target_channels),
        ):
            if not values:
                raise ValueError(f"data.{name} is empty; it must name at least one channel.")
            duplicates = sorted({c for c in values if values.count(c) > 1})
            if duplicates:
                raise ValueError(f"data.{name} repeats {duplicates}.")
            missing = [c for c in values if c not in self.channels]
            if missing:
                raise ValueError(
                    f"data.{name} names {missing}, which data.channels does not contain, "
                    "so the dataset never reads them.\n"
                    f"data.channels is currently {self.channels}. It must be the union of "
                    "input_channels and target_channels."
                )

        # Channels read but never used are almost always a leftover from editing the
        # lists, and each one costs decode time and memory on every sample.
        unused = [c for c in self.channels
                  if c not in self.input_channels and c not in self.target_channels]
        if unused:
            raise ValueError(
                f"data.channels contains {unused}, which are neither inputs nor targets. "
                "Every channel listed there is decoded for every sample, so remove them "
                "or move them into input_channels / target_channels."
            )

        missing_pretrained = [
            c for c in self.input_channels if c not in self.pretrained_channel_order
        ]
        if missing_pretrained:
            raise ValueError(
                f"data.input_channels names {missing_pretrained}, which is not in "
                "data.pretrained_channel_order, so there are no pretrained tokenizer "
                "weights for it.\n"
                f"The pretraining order is {self.pretrained_channel_order}. Either the "
                "channel name is misspelt, or this checkpoint was not pretrained on it."
            )

    def validate_with_model(self, model) -> None:
        """The tokenizer must be built for exactly the channels the dataset delivers.

        A mismatch here is not caught by a shape error at startup: the model would be
        constructed for one channel count and handed another, and the failure surfaces as
        an opaque convolution error in the first forward pass -- or, if the counts happen
        to be compatible, as silently wrong channel-to-filter assignment.
        """
        if model.in_channels != len(self.input_channels):
            raise ValueError(
                f"model.in_channels ({model.in_channels}) does not match "
                f"len(data.input_channels) ({len(self.input_channels)}: "
                f"{self.input_channels}).\n"
                f"Set model.in_channels: {len(self.input_channels)}. It is the number of "
                "channels the tokenizer is built for, which must be the number the "
                "dataset hands it -- not the 13 the checkpoint was pretrained with."
            )

    @property
    def input_channel_indices(self) -> list[int]:
        """Positions of ``input_channels`` in the pretraining channel order.

        This is what ``load_pretrained_weights(channel_indices=...)`` wants: it selects
        these channels' planes out of the checkpoint's patch-embedding weight so the
        tokenizer starts as the pretrained tokenizer restricted to them.
        """
        return [self.pretrained_channel_order.index(c) for c in self.input_channels]


load_euv2mag_config = partial(load_config, data_cls=Euv2MagDataConfig)


__all__ = [
    "Euv2MagDataConfig",
    "load_euv2mag_config",
    "SURYA_PRETRAINED_CHANNEL_ORDER",
    # Re-exports so app code can import everything config-related from one place.
    "DataConfig",
    "OutputConfig",
    "TrainingConfig",
    "ModelConfig",
    "LoraAdapterConfig",
    "TimeEmbeddingConfig",
    "load_config",
]
