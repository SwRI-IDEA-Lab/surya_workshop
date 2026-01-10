import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd
import re
import s3fs
from tqdm import tqdm  # Import tqdm for the progress bar


def fetch_nc_files(directory, start_year, start_month, end_year, end_month):
    # Define the regex pattern for matching the filenames
    pattern = re.compile(r"(\d{8})_(\d{4})\.nc")

    # Check if the directory is an S3 path
    if directory.startswith("s3://"):
        fs = s3fs.S3FileSystem()
        all_files = ['s3://' + f for f in fs.glob(os.path.join(directory, "**/*.nc"))]  # Get all nc files recursively
        # Construct a more specific glob pattern to limit the search
        # glob_pattern = os.path.join(
        #     directory,
        #     *[str(year) for year in range(start_year, end_year + 1)],
        #     *[f"{month:02}" for month in range(start_month, end_month + 1)],
        #     "*.nc",
        # )
        

        # all_files = ['s3://' + f for f in fs.glob(glob_pattern)]  # Get all nc files recursively
    else:
        all_files = sorted(Path(directory).rglob("*.nc"))

    # List to store matching files
    matching_files = []

    # Iterate over files
    for filepath in sorted(all_files):
        filename = os.path.basename(filepath)
        match = pattern.match(filename)
        # if match:
        # breakpoint()
        if int(match.group(1)[4:6]) >= start_month:
            # Extract the date part from the filename (YYYYMMDD)
            date_str = match.group(1)
            if start_month and end_month:
                year = int(date_str[:4])  # Get the year from the date string
                month = int(date_str[4:6])  # Get the month from the date string

                # Check if the file is within the given year and month range
                if (
                    (start_year < year < end_year)
                    or (
                        year == start_year
                        and month >= start_month
                        and month <= end_month
                    )
                    or (year == end_year and month <= end_month)
                ):
                    matching_files.append(str(filepath))

            else:
                year = int(date_str[:4])  # Get the year from the date string
                # Check if the year is within the given range
                if start_year <= year <= end_year:
                    matching_files.append(filepath)

    return matching_files


def create_csv_index(
    dirpath,
    start_year,
    start_month,
    end_year,
    end_month,
    csv_output,
    all_possible_intervals,
):
    #    nc_files = sorted(Path(dirpath).rglob("*.nc"))
    nc_files = fetch_nc_files(dirpath, start_year, start_month, end_year, end_month)

    nc_files = set(nc_files)
    print(f"Number of nc files: {len(nc_files)}")
    records = []

    # Wrap the loop with tqdm for the progress bar
    for filepath, time_val in tqdm(all_possible_intervals, desc="Processing files"):
        if filepath in nc_files:
            present = 1
        else:
            present = 0

        records.append(
            {
                "path": filepath,
                "timestep": time_val,  # numpy.datetime64[ns]
                "present": present,
            }
        )
    # Create a DataFrame from the records and save it as a CSV file
    df = pd.DataFrame(records)
    df.reset_index()
    df.to_csv(csv_output)
    print(f"Index file created at {csv_output}")


def generate_time_intervals(dirpath, start_year, start_month, end_year, end_month):
    # Define the start time using numpy.datetime64
    start_time = np.datetime64(f"{start_year}-{start_month:02d}-01 00:00:00")

    # Use pandas to get the last day of the end month
    end_time = (
        pd.to_datetime(f"{end_year}-{end_month:02d}-01")
        + pd.DateOffset(months=1)
        - pd.Timedelta(days=1)
    )
    end_time = np.datetime64(end_time)  # Convert it back to numpy.datetime64

    # Create time intervals (12-minute increments)
    time_intervals = pd.date_range(start=start_time, end=end_time, freq="12T")

    # Initialize a list to store the file paths and corresponding datetime
    result = []

        # Check if the directory is an S3 path
    is_s3 = dirpath.startswith("s3://")

        # Iterate over the generated time intervals
    for time in time_intervals:
        # Generate the filename based on the time
        date_str = time.strftime("%Y%m%d")  # Extract date as YYYYMMDD
        time_str = time.strftime("%H%M")  # Extract time as HHMM

        filename = os.path.join(
            dirpath, f"{time.year}", f"{time.month:02d}", f"{date_str}_{time_str}.nc"
        )   
        if is_s3:
            filename =  f"{dirpath}/{time.year}/{time.month:02d}/{date_str}_{time_str}.nc"

        formatted_time = time.strftime("%Y-%m-%d %H:%M:%S")
      
        result.append([filename, formatted_time])

    return result


def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(
        description="Generate a CSV index of netCDF files."
    )
    parser.add_argument(
        "--dirpath",
        default="/nobackupnfs1/sroy14/processed_data/Helio/lz4",
        type=str,
        help="Root Directory path to search for netCDF files. Do NOT include year and month",
    )
    parser.add_argument(
        "--start_year", type=int, default=2010, help="Start year of data"
    )
    parser.add_argument(
        "--start_month", type=int, default=1, help="starting month  of data"
    )
    parser.add_argument(
        "--end_year", default=2015, type=int, help="Ending year of data"
    )
    parser.add_argument(
        "--end_month", type=int, default=6, help="Ending month  of data"
    )

    parser.add_argument(
        "--csv_output", default="lz4_csv/train_201101_201106.csv", type=str, help="Output CSV file path."
    )
    args = parser.parse_args()

    all_possible_intervals = generate_time_intervals(
        args.dirpath, args.start_year, args.start_month, args.end_year, args.end_month
    )
    # Generate the CSV index
    create_csv_index(
        args.dirpath,
        args.start_year,
        args.start_month,
        args.end_year,
        args.end_month,
        args.csv_output,
        all_possible_intervals,
    )
    # create_csv_index('/lustre/fs0/scratch/shared/data/', 2011, 2, 2011, 2, 'temp.csv')


if __name__ == "__main__":
    main()
