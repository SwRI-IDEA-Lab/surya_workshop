import numpy as np
import pandas as pd
import datetime
from typing import Literal, Optional
from workshop_infrastructure.datasets.helio_aws import HelioNetCDFDatasetAWS

#def crop_ts(row):
#    ts = row["ts"]  # shape (13, 1, 4096, 4096)#
#
#    x0 = int(row["bbox_ll_x_pix"])
#    x1 = int(row["bbox_ur_x_pix"])
#    y0 = int(row["bbox_ll_y_pix"])
#    y1 = int(row["bbox_ur_y_pix"])##

    # Crop last two dimensions (y, x)
#    return ts[..., y0:y1, x0:x1]

class SolarJetDataset(HelioNetCDFDatasetAWS):
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
        ds_fov_size_pix: float | None = None,
    ):

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

        # Load time from jet catalogue to make ds index, add two minutes for the jet to appear 
        self.ds_index["ds_index"] = (
            pd.to_datetime(self.ds_index[ds_time_column]) + pd.Timedelta(minutes=2))
        self.ds_index.sort_values("ds_index", inplace=True)

        if ds_fov_size_pix is None:
            raise ValueError("ds_fov_size_pix must be provided for SolarJetDataset")

        # Compute center coordinates of the bounding box (in arcsec)
        self.ds_index["bbox_center_x_arcsec"] = (self.ds_index["jet_bbox_min_x"] + self.ds_index["jet_bbox_max_x"]) / 2
        self.ds_index["bbox_center_y_arcsec"] = (self.ds_index["jet_bbox_min_y"] + self.ds_index["jet_bbox_max_y"]) / 2
        # Transform center coordinates to pixel
        pixel_size = 0.6
        npixels = 4096
        img_center = npixels/2
        self.ds_index["bbox_center_x_pixel"] = (self.ds_index["bbox_center_x_arcsec"] / pixel_size) + img_center
        self.ds_index["bbox_center_y_pixel"] = (self.ds_index["bbox_center_y_arcsec"] / pixel_size) + img_center
        # Get FOV size in pixel
        half_fov_size_pixel = ds_fov_size_pix / 2
        # compute new bounding box coordinates in pixels
        self.ds_index["bbox_ll_x_pixel"] = self.ds_index["bbox_center_x_pixel"] - half_fov_size_pixel
        self.ds_index["bbox_ur_x_pixel"] = self.ds_index["bbox_center_x_pixel"] + half_fov_size_pixel
        self.ds_index["bbox_ll_y_pixel"] = self.ds_index["bbox_center_y_pixel"] - half_fov_size_pixel
        self.ds_index["bbox_ur_y_pixel"] = self.ds_index["bbox_center_y_pixel"] + half_fov_size_pixel
        # Now correct for bounding box going "outside" of the image (center was too close to the border)
        ### calculating necessary shift in x
        shift_x = np.where(self.ds_index["bbox_ll_x_pixel"] < 0, 0 - self.ds_index["bbox_ll_x_pixel"], 0)
        shift_x = np.where(self.ds_index["bbox_ur_x_pixel"] + shift_x > (npixels-1), 
                           (npixels-1) - self.ds_index["bbox_ur_x_pixel"], shift_x)
        ### calculating necessary shift in y
        shift_y = np.where(self.ds_index["bbox_ll_y_pixel"] < 0, 0 - self.ds_index["bbox_ll_y_pixel"], 0)
        shift_y = np.where(self.ds_index["bbox_ur_y_pixel"] + shift_y > (npixels-1), 
                           (npixels-1) - self.ds_index["bbox_ur_y_pixel"], shift_y)
        ### apply shifts (overwrite)
        self.ds_index["bbox_ll_x_pixel"] = self.ds_index["bbox_ll_x_pixel"] + shift_x
        self.ds_index["bbox_ur_x_pixel"] = self.ds_index["bbox_ur_x_pixel"] + shift_x
        self.ds_index["bbox_ll_y_pixel"] = self.ds_index["bbox_ll_y_pixel"] + shift_y
        self.ds_index["bbox_ur_y_pixel"] = self.ds_index["bbox_ur_y_pixel"] + shift_y

        # Implement normalization.  This is going to be DS application specific, no two will look the same
        #self.ds_index["normalized_intensity"] = np.log10(self.ds_index["intensity"])
        #self.ds_index["normalized_intensity"] = self.ds_index[
        #    "normalized_intensity"
        #] - np.min(self.ds_index["normalized_intensity"])
        #self.ds_index["normalized_intensity"] = self.ds_index[
        #    "normalized_intensity"
        #] / (2 * np.std(self.ds_index["normalized_intensity"]))

        # Create Surya valid indices and find closest match to DS index
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
                raise ValueError("No intersection between Surya and DS indices")

        # Override valid indices variables to reflect matches between Surya and DS
        self.valid_indices = [
            pd.Timestamp(date) for date in self.df_valid_indices["valid_indices"]
        ]
        self.adjusted_length = len(self.valid_indices)
        self.df_valid_indices.set_index("valid_indices", inplace=True)

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
            base_dictionary = super().__getitem__(idx=idx)
            # crop according to FOV defined above
            x0 = int(self.df_valid_indices["bbox_ll_x_pixel"].iloc[idx])
            x1 = int(self.df_valid_indices["bbox_ur_x_pixel"].iloc[idx])
            y0 = int(self.df_valid_indices["bbox_ll_y_pixel"].iloc[idx])
            y1 = int(self.df_valid_indices["bbox_ur_y_pixel"].iloc[idx])
            base_dictionary["ts"] = base_dictionary["ts"][..., y0:y1, x0:x1]

        # We now add the flare intensity label
        #base_dictionary["forecast"] = self.df_valid_indices.iloc[idx][
        #    "normalized_intensity"
        #].astype(np.float32)

        # And the timestamp of the auxiliary index
        base_dictionary["ds_index"] = self.df_valid_indices["ds_index"].iloc[idx].isoformat()

        # Add bounding box coordinates
        #base_dictionary["bbox_ll_x_arcsec"] = self.df_valid_indices["bbox_ll_x_arcsec"].iloc[idx]
        #base_dictionary["bbox_ur_x_arcsec"] = self.df_valid_indices["bbox_ur_x_arcsec"].iloc[idx]
        #base_dictionary["bbox_ll_y_arcsec"] = self.df_valid_indices["bbox_ll_y_arcsec"].iloc[idx]
        #base_dictionary["bbox_ur_y_arcsec"] = self.df_valid_indices["bbox_ur_y_arcsec"].iloc[idx]
        print("Reloaded!5")
        return base_dictionary
