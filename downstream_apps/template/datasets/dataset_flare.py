import numpy as np
import pandas as pd
from typing import Literal, Optional
from workshop_infrastructure.datasets.helio_aws import HelioNetCDFDatasetAWS


class SolarFlareDataset(HelioNetCDFDatasetAWS):
    """
    The solar flare index data (flare_index_path) should be of the form

    timestamp,max_goes_class,cumulative_index,label_max,label_cum
    2011-01-01 00:00:00,B8.3,0.0,0,0
    2011-01-01 01:00:00,B8.3,0.0,0,0
    2011-01-01 02:00:00,B8.3,0.0,0,0
    2011-01-01 03:00:00,B8.3,0.0,0,0
    2011-01-01 04:00:00,B8.3,0.0,0,0
    2011-01-01 05:00:00,B8.3,0.0,0,0
    2011-01-01 06:00:00,B8.3,0.0,0,0
    """

    def __init__(
        self,
        index_path: str,
        time_delta_input_minutes: list[int],
        time_delta_target_minutes: int,
        n_input_timestamps: int,
        rollout_steps: int,
        scalers=None,
        num_mask_aia_channels=0,
        drop_hmi_probability=0,
        use_latitude_in_learned_flow=False,
        channels: list[str] | None = None,
        phase="train",
        pooling: int | None = None,
        random_vert_flip: bool = False,
        s3_use_simplecache: bool = True,
        s3_cache_dir: str = "/tmp/helio_s3_cache",
        #### Put your donwnstream (DS) specific parameters below this line
        label_type: str = "label_max",
        return_surya_stack: bool = True,
        max_number_of_samples: int | None = None,
        flare_index_path: str = None,
    ):
        self.label_type = label_type
        self.return_surya_stack = return_surya_stack
        self.flare_index = pd.read_csv(flare_index_path)
        self.flare_index["timestamp"] = pd.to_datetime(
            self.flare_index["timestamp"]
        ).values.astype("datetime64[ns]")
        self.flare_index.set_index("timestamp", inplace=True)
        self.flare_index.sort_index(inplace=True)

        super().__init__(
            index_path=index_path,
            time_delta_input_minutes=time_delta_input_minutes,
            time_delta_target_minutes=time_delta_target_minutes,
            n_input_timestamps=n_input_timestamps,
            rollout_steps=rollout_steps,
            scalers=scalers,
            num_mask_aia_channels=num_mask_aia_channels,
            drop_hmi_probability=drop_hmi_probability,
            use_latitude_in_learned_flow=use_latitude_in_learned_flow,
            channels=channels,
            phase=phase,
            pooling=pooling,
            random_vert_flip=random_vert_flip,
            s3_use_simplecache=s3_use_simplecache,
            s3_cache_dir=s3_cache_dir,
        )

        if max_number_of_samples is not None:
            self.adjusted_length = min(self.adjusted_length, max_number_of_samples)

        self.valid_indices = self.filter_valid_indices()

    def filter_valid_indices(self) -> list:
        valid_indices = super().filter_valid_indices()

        valid_indices = [t for t in valid_indices if t in self.flare_index.index]

        return valid_indices

    # def _get_index_data(self, idx: int) -> tuple[dict, dict]:
    #     data, metadata = super()._get_index_data(idx)

    #     reference_timestamp = self.valid_indices[idx]
    #     data["label"] = self.index.loc[reference_timestamp, "label_max"]

    #     return data, metadata

    def __getitem__(self, idx: int) -> dict:
        """
        Args:
            idx: Index of sample to load. (Pytorch standard.)
        Returns:
            Dictionary with following keys. The values are tensors with shape as follows:
                # Surya keys--------------------------------
                ts (torch.Tensor):                C, T, H, W
                time_delta_input (torch.Tensor):  T
                input_latitude (torch.Tensor):    T
                lead_time_delta (torch.Tensor):   L
                forecast_latitude (torch.Tensor): L
                # Surya keys--------------------------------
                forecast
            C - Channels, T - Input times, H - Image height, W - Image width, L - Lead time.
        """

        base_dictionary = {}
        if self.return_surya_stack:
            # This lines assembles the dictionary that Surya's dataset returns (defined above)
            base_dictionary = super().__getitem__(idx=idx)

        reference_timestamp = self.valid_indices[idx]
        base_dictionary["label"] = self.flare_index.loc[
            reference_timestamp, self.label_type
        ]
        # save timestamp with int (nanoseconds)
        base_dictionary["flare_timestamp"] = reference_timestamp.value
        # base_dictionary["flare_timestamp"] = reference_timestamp.value // 10**9

        return base_dictionary
