from typing import Optional

from workshop_infrastructure.datasets.helio_aws import HelioNetCDFDatasetAWS


class EUV2MAGDataset(HelioNetCDFDatasetAWS):
    """
    Dataset for translating EUV AIA channels to HMI magnetogram targets. It includes only the necessary parameters
    to initialize the parent class, but not the specific downstream (DS) parameters because this downstream application 
    requires only the data from the Surya dataset itself.

    Surya Parameters
    ------------------
    index_path : str
        Path to Surya index
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
    s3_use_simplecache : bool, optional
        If True (default), use fsspec's simplecache to keep a local read-through
        cache of objects.
    s3_cache_dir : str, optional
        Directory used by simplecache. Default: /tmp/helio_s3_cache

    Downstream (DS) Parameters
    --------------------------
    return_surya_stack : bool, optional
        If True (default), the dataset will return the full Surya stack
        otherlwise only the flare intensity label is returned
    input_channels : list[str] | None, optional
        AIA EUV input channels, e.g. ["aia304", "aia193", "aia171"].
    target_channels : list[str] | None, optional
        HMI magnetogram target channels, e.g. ["hmi_m"] or ["hmi_bx", "hmi_by", "hmi_bz"].


    Raises
    ------
    ValueError
        Error is raised if there is not overlap between the Surya and DS indices
        given a tolerance

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
        s3_use_simplecache: bool = True,
        s3_cache_dir: str = "/tmp/helio_s3_cache",
        #### Put your donwnstream (DS) specific parameters below this line
        input_channels: Optional[list[str]] = None,
        target_channels: Optional[list[str]] = None,
        return_surya_stack: bool = True,
    ):
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
            s3_use_simplecache=s3_use_simplecache,
            s3_cache_dir=s3_cache_dir,
        )

        self.input_channels = input_channels or ["aia304", "aia193", "aia171"]
        self.target_channels = target_channels or ["hmi_m"]
        self.return_surya_stack = return_surya_stack

    def __len__(self):
        return self.adjusted_length

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
        base_dictionary = super().__getitem__(idx=idx)

        overlap = set(self.input_channels) & set(self.target_channels)
        if overlap:
            raise ValueError(f"target_channels must not overlap input_channels: {sorted(overlap)}")

        channel_to_index = {name: i for i, name in enumerate(self.channels)}
        missing = set(self.input_channels + self.target_channels) - set(channel_to_index.keys())
        if missing:
            raise ValueError(f"Missing channels in dataset: {sorted(missing)}")

        input_indices = [channel_to_index[name] for name in self.input_channels]
        target_indices = [channel_to_index[name] for name in self.target_channels]

        ts = base_dictionary["ts"][input_indices, ...]
        forecast = base_dictionary["forecast"][target_indices, ...]

        if not self.return_surya_stack:
            return {"forecast": forecast}

        base_dictionary["ts"] = ts
        base_dictionary["forecast"] = forecast

        return base_dictionary
