from typing import Optional
from workshop_infrastructure.datasets.helio_aws import HelioNetCDFDatasetAWS


class EUVToMagnetogramDataset(HelioNetCDFDatasetAWS):
    """
    Dataset for translating EUV AIA channels to HMI magnetogram targets.

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
    input_channels : list[str]
        AIA EUV input channels, e.g. ["aia304", "aia193", "aia171"].
    target_channel : str
        HMI magnetogram target channel, e.g. "hmi_m".

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
        drop_hmi_probability=0,
        use_latitude_in_learned_flow=False,
        channels: list[str] | None = None,
        phase="train",
        s3_use_simplecache: bool = True,
        s3_cache_dir: str = "/tmp/helio_s3_cache",
        #### Put your donwnstream (DS) specific parameters below this line
        input_channels: Optional[list[str]] = None,
        target_channel: str = "hmi_m",
    ):
        if input_channels is None:
            input_channels = ["aia304", "aia193", "aia171"]
        if target_channel in input_channels:
            raise ValueError("target_channel must not be included in input_channels")

        self.input_channels = input_channels
        self.target_channel = target_channel
        all_channels = input_channels + [target_channel]

        if channels is not None:
            missing = set(input_channels + [target_channel]) - set(channels)
            if missing:
                raise ValueError(f"channels must include input/target channels: {sorted(missing)}")

        ## Initialize parent class
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
            channels=all_channels if channels is None else channels,
            phase=phase,
            s3_use_simplecache=s3_use_simplecache,
            s3_cache_dir=s3_cache_dir,
        )

    def __getitem__(self, idx: int) -> dict:
        """
        Args:
            idx: Index of sample to load. (Pytorch standard.)
        Returns:
            Dictionary with following keys. The values are tensors with shape as follows:
                # Surya keys--------------------------------
                ts (torch.Tensor):                C_in, T, H, W
                time_delta_input (torch.Tensor):  T
                lead_time_delta (torch.Tensor):   L
                # Surya keys--------------------------------
                forecast (torch.Tensor):          C_out, L, H, W
            C_in - Input channels, C_out - Output channels.
        """
        base_dictionary = super().__getitem__(idx=idx)

        channel_to_index = {name: i for i, name in enumerate(self.channels)}
        input_indices = [channel_to_index[name] for name in self.input_channels]
        target_index = channel_to_index[self.target_channel]

        base_dictionary["ts"] = base_dictionary["ts"][input_indices, ...]
        base_dictionary["forecast"] = base_dictionary["forecast"][[target_index], ...]

        return base_dictionary
