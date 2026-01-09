import numpy as np
import pandas as pd
from Surya.surya.datasets.helio import HelioNetCDFDataset


class FlareDSDataset(HelioNetCDFDataset):
    """
    Template child class of HelioNetCDFDataset to show an example of how to create a
    dataset for donwstream applications. It includes both the necessary parameters
    to initialize the parent class, as well as those of the child

    HelioFM Parameters
    ------------------
    index_path : str
        Path to HelioFM index
    time_delta_input_minutes : list[int]
        Input delta times to define the input stack in minutes from the present
    time_delta_target_minutes : int
        Target delta time to define the output stack on rollout in minutes from the present
    n_input_timestamps : int
        Number of input timestamps
    rollout_steps : int
        Number of rollout steps
    scalers : optional
        scalers used to perform input data normalization, by default None
    num_mask_aia_channels : int, optional
        Number of aia channels to mask during training, by default 0
    drop_hmi_probablity : int, optional
        Probability of removing hmi during training, by default 0
    use_latitude_in_learned_flow : bool, optional
        Switch to provide heliographic latitude for each datapoint, by default False
    channels : list[str] | None, optional
        Input channels to use, by default None
    phase : str, optional
        Descriptor of the phase used for this database, by default "train"

    Downstream (DS) Parameters
    --------------------------
    ds_flare_index_path : str, optional
        DS index.  In this example a flare dataset, by default None
    ds_time_column : str, optional
        Name of the column to use as datestamp to compare with HelioFM's index, by default None
    ds_time_tolerance : str, optional
        How much time difference is tolerated when finding matches between HelioFM and the DS, by default None
    ds_match_direction : str, optional
        Direction used to find matches using pd.merge_asof possible values are "forward", "backward",
        or "nearest".  For causal relationships is better to use "forward", by default "forward"

    Raises
    ------
    ValueError
        Error is raised if there is not overlap between the HelioFM and DS indices
        given a tolerance
    """

    def __init__(
        self,
        #### All these lines are required by the parent HelioNetCDFDataset class
        index_path: str,
        time_delta_input_minutes: list[int],
        time_delta_target_minutes: int,
        n_input_timestamps: int,
        rollout_steps: int,
        scalers=None,
        num_mask_aia_channels=0,
        drop_hmi_probablity=0,
        use_latitude_in_learned_flow=False,
        channels: list[str] | None = None,
        phase="train",
        #### Put your donwnstream (DS) specific parameters below this line
        ds_flare_index_path: str = None,
        ds_time_column: str = None,
        ds_time_tolerance: str = None,
        ds_match_direction: str = "forward",
    ):

        ## Initialize parent class
        super().__init__(
            index_path=index_path,
            time_delta_input_minutes=time_delta_input_minutes,
            time_delta_target_minutes=time_delta_target_minutes,
            n_input_timestamps=n_input_timestamps,
            rollout_steps=rollout_steps,
            scalers=scalers,
            num_mask_aia_channels=num_mask_aia_channels,
            drop_hmi_probablity=drop_hmi_probablity,
            use_latitude_in_learned_flow=use_latitude_in_learned_flow,
            channels=channels,
            phase=phase,
        )

        # Load ds index and find intersection with HelioFM index
        self.ds_index = pd.read_csv(ds_flare_index_path)
        self.ds_index["ds_index"] = pd.to_datetime(
            self.ds_index[ds_time_column]
        ).values.astype("datetime64[ns]")
        self.ds_index.sort_values("ds_index", inplace=True)

        # Implement normalization.  This is going to be DS application specific, no two will look the same
        self.ds_index["normalized_intensity"] = np.log10(self.ds_index["intensity"])
        self.ds_index["normalized_intensity"] = self.ds_index[
            "normalized_intensity"
        ] - np.min(self.ds_index["normalized_intensity"])
        self.ds_index["normalized_intensity"] = self.ds_index[
            "normalized_intensity"
        ] / (2 * np.std(self.ds_index["normalized_intensity"]))

        # Create HelioFM valid indices and find closest match to DS index
        self.df_valid_indices = pd.DataFrame(
            {"valid_indices": self.valid_indices}
        ).sort_values("valid_indices")
        self.df_valid_indices = pd.merge_asof(
            self.df_valid_indices,
            self.ds_index,
            right_on="ds_index",
            left_on="valid_indices",
            direction=ds_match_direction,
        )
        # Remove duplicates keeping closest match
        self.df_valid_indices["index_delta"] = np.abs(
            self.df_valid_indices["valid_indices"] - self.df_valid_indices["ds_index"]
        )
        self.df_valid_indices = self.df_valid_indices.sort_values(
            ["ds_index", "index_delta"]
        )
        self.df_valid_indices.drop_duplicates(
            subset="ds_index", keep="first", inplace=True
        )
        # Enforce a maximum time tolerance for matches
        if ds_time_tolerance is not None:
            self.df_valid_indices = self.df_valid_indices.loc[
                self.df_valid_indices["index_delta"] <= pd.Timedelta(ds_time_tolerance),
                :,
            ]
            if len(self.df_valid_indices) == 0:
                raise ValueError("No intersection between HelioFM and DS indices")

        # Override valid indices variables to reflect matches between HelioFM and DS
        self.valid_indices = [
            pd.Timestamp(date) for date in self.df_valid_indices["valid_indices"]
        ]
        self.adjusted_length = len(self.valid_indices)
        self.df_valid_indices.set_index("valid_indices", inplace=True)

    def __len__(self):
        return self.adjusted_length

    def __getitem__(self, idx: int) -> dict:
        """
        Args:
            idx: Index of sample to load. (Pytorch standard.)
        Returns:
            Dictionary with following keys. The values are tensors with shape as follows:
                # HelioFM keys--------------------------------
                ts (torch.Tensor):                C, T, H, W
                time_delta_input (torch.Tensor):  T
                input_latitude (torch.Tensor):    T
                lead_time_delta (torch.Tensor):   L
                forecast_latitude (torch.Tensor): L
                # HelioFM keys--------------------------------
                forecast
            C - Channels, T - Input times, H - Image height, W - Image width, L - Lead time.
        """

        # This lines assembles the dictionary that HelioFM's dataset returns (defined above)
        base_dictionary, metadata = super().__getitem__(idx=idx)

        # We now add the flare intensity label
        base_dictionary["forecast"] = self.df_valid_indices.iloc[idx][
            "normalized_intensity"
        ].astype(np.float32)

        return base_dictionary, metadata
