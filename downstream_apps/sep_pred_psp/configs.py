"""
Task-specific configuration for the SEP-from-PSP prediction app.

Everything generic — paths, channels, temporal sampling, S3 settings, the model and
LoRA configs, the training and logging sections, and ``load_config()`` itself — lives in
``workshop_infrastructure/configs.py``. This file holds only what is specific to *this*
task: the SHARP-to-PSP catalog and how its events are aligned to the Surya index.

**This is the pattern to copy when you fork the template.** Subclass ``DataConfig`` with
your task's fields, then bind ``load_config`` to it. You never maintain a copy of the
base config.
"""

from __future__ import annotations

from dataclasses import dataclass
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


@dataclass
class SepPspDSDataConfig(DataConfig):
    """DataConfig plus the SHARP-to-PSP catalog alignment settings used by ``SepPspDSDataset``.

    These keys are what makes this app's ``data:`` section different from any other
    downstream task's. Swap them for your own when you fork.
    """
    # Path to the label catalog (relative paths resolve against the config file's dir).
    sep_psp_index_path: str = ""
    # PSP/ISOIS SEP event list; decides which samples are events vs. non-events.
    event_list_path: str = ""
    # Held-out Surya index for final evaluation (not used during training).
    test_data_path: str = ""
    # Minimum PSP-time gap between a non-event sample and any event window.
    non_event_buffer: str = "1d"
    # Ceiling on how many Surya frames a single SEP event may contribute to a split.
    # 1 keeps every event sample independent, but then a split can never hold more event
    # samples than it has events (95 train / 11 val here, so 190 / 22 balanced samples).
    # Raising it lets a draw grow past that by taking a second, third, ... frame from each
    # event window, spread as far apart in PSP time as the window allows. Those frames are
    # further views of an event already in the set, not new events.
    max_frames_per_event: int = 1
    # Catalog column used as the training label. "Jlinlin" is smoothed in time
    # (~0.02 dex hour to hour); "Jlinlin_raw" is unsmoothed (~0.12 dex). Both span ~6
    # decades, so they are log10-z-scored by datasets/label_transform.py before training.
    label_column: str = "Jlinlin"
    # Column in the catalog holding the event timestamp.
    ds_time_column: str = "start_time"
    # Max allowed gap when matching catalog events to Surya timesteps.
    ds_time_tolerance: str = "4d"
    # "forward" uses the solar state *before* the event (causal prediction).
    ds_match_direction: str = "forward"

    # These are paths, so they must join the base class's list to get the same
    # relative-to-the-config-file resolution. Extend this whenever you add a path field.
    PATH_FIELDS: ClassVar[tuple[str, ...]] = DataConfig.PATH_FIELDS + (
        "sep_psp_index_path",
        "event_list_path",
        "test_data_path",
    )


# The app's entry point. Identical to load_config() except that the data: section is
# parsed into SepPspDSDataConfig, so the keys above are recognized instead of rejected.
load_sep_psp_ds_config = partial(load_config, data_cls=SepPspDSDataConfig)


__all__ = [
    "SepPspDSDataConfig",
    "load_sep_psp_ds_config",
    # Re-exports so app code can import everything config-related from one place.
    "DataConfig",
    "OutputConfig",
    "TrainingConfig",
    "ModelConfig",
    "LoraAdapterConfig",
    "TimeEmbeddingConfig",
    "load_config",
]
