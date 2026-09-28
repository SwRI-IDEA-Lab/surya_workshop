"""
The EUV2MAG dataset: a channel split over the Surya stack.

``HelioNetCDFDataset`` does all the work -- NetCDF loading from local disk or S3, per
channel signum-log + z-score normalization, input frame sampling, validity filtering and
seeded subsampling. This subclass adds one thing: it splits the channels the base class
returns into the ones the model sees and the one it is asked to predict.

The split is by name, not by position, because ``data.channels`` is the app's own list and
its order is free. The indices are computed once in ``__init__`` rather than per item.

Nothing here validates the channel lists -- ``Euv2MagDataConfig.__post_init__`` does that
at config-load time, so a bad list fails before the first NetCDF file is opened rather
than on every ``__getitem__``.
"""

from __future__ import annotations

from workshop_infrastructure.datasets.helio import HelioNetCDFDataset


class Euv2MagDataset(HelioNetCDFDataset):
    """Surya stack split into EUV inputs and a magnetogram target.

    ``load_forecast_frames`` is left at its default ``True``, unlike the flare template
    which sets it ``False``. The flare label comes from an external catalog, so that app
    has no use for the target frame; here the target *is* the forecast frame, which is the
    whole task.

    Additional Args (everything else goes to ``HelioNetCDFDataset``):
        input_channels: Channels fed to the model, in the order its tokenizer expects.
        target_channels: Channels the model predicts.
        return_surya_stack: If False, return only the target. Useful for inspecting label
            statistics without paying for the input frames.

    Shapes, for C_in input channels, C_out target channels and T input frames:
        ts        (C_in, T, H, W)
        forecast  (C_out, L, H, W)   L = 1 + rollout_steps
    """

    def __init__(
        self,
        input_channels: list[str],
        target_channels: list[str],
        return_surya_stack: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.input_channels = list(input_channels)
        self.target_channels = list(target_channels)
        self.return_surya_stack = return_surya_stack

        # Resolved once. self.channels is the order the base class stacks channels in, so
        # these index the channel axis of what __getitem__ receives.
        position = {name: i for i, name in enumerate(self.channels)}
        unknown = [c for c in self.input_channels + self.target_channels if c not in position]
        if unknown:
            # Reachable only by constructing the dataset directly; via the config the
            # check in Euv2MagDataConfig fires first, with a fuller message.
            raise ValueError(
                f"{unknown} are not in this dataset's channels ({self.channels}). "
                "input_channels and target_channels must both be subsets of data.channels."
            )
        self._input_indices = [position[c] for c in self.input_channels]
        self._target_indices = [position[c] for c in self.target_channels]

    def __getitem__(self, idx: int) -> dict:
        """Return the base sample with ``ts`` and ``forecast`` restricted to their channels.

        The base class's other keys (``time_delta_input``, ``lead_time_delta``, the
        latitudes) are passed through untouched -- the model's time embedding reads them.
        """
        sample = super().__getitem__(idx=idx)

        forecast = sample["forecast"][self._target_indices, ...]
        if not self.return_surya_stack:
            return {"forecast": forecast}

        sample["ts"] = sample["ts"][self._input_indices, ...]
        sample["forecast"] = forecast
        return sample
