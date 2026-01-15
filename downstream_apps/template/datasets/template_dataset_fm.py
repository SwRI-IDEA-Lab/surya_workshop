import numpy as np
import pandas as pd
from typing import Literal, Optional
from workshop_infrastructure.datasets.helio_aws import HelioNetCDFDatasetAWS
from astropy.io import fits
from datetime import datetime
from glob import glob


class FlareDSDataset(HelioNetCDFDatasetAWS):
    """
    Template child class of HelioNetCDFDatasetAWS to show an example of how to create a
    dataset for donwstream applications. It includes both the necessary parameters
    to initialize the parent class, as well as those of the child

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
    max_number_of_samples : int | None, optional
        If provided, limits the maximum number of samples in the dataset, by default None
    ds_flare_index_path : str, optional
        DS index.  In this example a flare dataset, by default None
    ds_time_column : str, optional
        Name of the column to use as datestamp to compare with Surya's index, by default None
    ds_time_tolerance : str, optional
        How much time difference is tolerated when finding matches between Surya and the DS, by default None
    ds_match_direction : str, optional
        Direction used to find matches using pd.merge_asof possible values are "forward", "backward",
        or "nearest".  For causal relationships is better to use "forward", by default "forward"

    Raises
    ------
    ValueError
        Error is raised if there is not overlap between the Surya and DS indices
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
        drop_hmi_probability=0,
        use_latitude_in_learned_flow=False,
        channels: list[str] | None = None,
        phase="train",
        s3_use_simplecache: bool = True,
        s3_cache_dir: str = "/tmp/helio_s3_cache",
        #### Put your donwnstream (DS) specific parameters below this line
        return_surya_stack: bool = True,
        max_number_of_samples: int | None = None,
        ds_flare_index_path: str | None = None,
        ds_time_column: str | None = None,
        ds_time_tolerance: str | None = None,
        ds_match_direction: Literal["forward", "backward", "nearest"] = "forward",
    ):
        print("Hello")
        if ds_match_direction not in ["forward", "backward", "nearest"]:
            raise ValueError("ds_match_direction must be one of 'forward', 'backward', or 'nearest'")

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
            channels=channels,
            phase=phase,
            s3_use_simplecache=s3_use_simplecache,
            s3_cache_dir=s3_cache_dir,
        )

        self.return_surya_stack = return_surya_stack

        # Load ds index and find intersection with Surya index
        if ds_flare_index_path is not None:
            self.ds_index = pd.read_csv(ds_flare_index_path)
        else:
            raise ValueError("ds_flare_index_path must be provided for FlareDSDataset")

        self.ds_index["ds_index"] = pd.to_datetime(
            self.ds_index[ds_time_column]
        ).values.astype("datetime64[ns]")
        self.ds_index.sort_values("ds_index", inplace=True)
        


        
        self.ds_time_column = ds_time_column
    
        
        # Remove duplicates keeping closest match
        '''
        self.df_valid_indices["index_delta"] = np.abs(
            self.df_valid_indices["valid_indices"] - self.df_valid_indices["ds_index"]
        )
        self.df_valid_indices = self.df_valid_indices.sort_values(
            ["ds_index", "index_delta"]
        )
        self.df_valid_indices.drop_duplicates(
            subset="ds_index", keep="first", inplace=True
        )
        '''
       

        if max_number_of_samples is not None:
            self.adjusted_length = min(self.adjusted_length, max_number_of_samples)

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
        base_dictionary = {}
        if self.return_surya_stack:
            # This lines assembles the dictionary that Surya's dataset returns (defined above)
            base_dictionary= super().__getitem__(idx=idx)
        
        # ---- Load filament mask (once) ----
        fits_time = self.ds_index.iloc[idx][self.ds_time_column]
        fits_time = datetime.strptime(fits_time, "%Y-%m-%d %H:%M:%S")
        mask_fits_path = f"/shared/huggingface_data/filaments/AIA171_masked_{fits_time.strftime('%Y%m%d_%H')}00*.fits"

        with fits.open(glob(mask_fits_path)[0]) as hdul:
            mask = np.array(hdul[0].data)

        # Ensure mask is float32 and binary
        self.mask = np.flipud((mask > 0).astype(np.float32)).copy()

        # Optional sanity check
        assert self.mask.ndim == 2, "Mask must be 2D (H, W)"
        
        base_dictionary["mask"] = self.mask
        base_dictionary["ds_index"] = self.ds_index.iloc[idx][self.ds_time_column]
        print(self.ds_index.iloc[idx][self.ds_time_column])
        
        
        #print(base_dictionary["ds_index"])
        #print(type(base_dictionary["ds_index"]))
        print(self.mask.shape)
        print(base_dictionary)
        return base_dictionary
