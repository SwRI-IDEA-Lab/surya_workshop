#!/usr/bin/env python3
"""
Build train/validation/test CSV indices from the output of create_csv_index.py,
with user-configurable day-of-year windows.

The input CSV is expected to contain at least:
    - path: str
    - timestep: str or datetime-like
    - present: int (0 or 1)

This script:
  1) Parses `timestep` as pandas datetime.
  2) Adds `dayofyear = timestep.dt.dayofyear`.
  3) Optionally filters to `present == 1`.
  4) Optionally excludes whole years (default: 2012, 2022).
  5) Splits samples by day-of-year into four windows:
        - first padding window  (default: 1..15)   -> excluded from all splits
        - validation window     (default: 16..30)  -> validation split
        - test window           (default: 31..45)  -> test split
        - second padding window (default: 46..60)  -> excluded from all splits
     Training is defined as the remainder (i.e., all samples NOT in any of the
     above windows), within the allowed years / presence filter.

Important:
  - The windows are inclusive on both ends.
  - Windows must not overlap.
  - By default, this mirrors your requested defaults; it does not attempt to
    reproduce the earlier notebook’s “>= 61” rule unless you set windows to do so.

Example:
  python build_splits_from_index.py \
    --input_csv lz4_csv/index.csv \
    --out_dir lz4_csv/splits \
    --only_present \
    --val_window 16 30 \
    --test_window 31 45 \
    --pad1_window 1 15 \
    --pad2_window 46 60
"""

from __future__ import annotations

import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, List

import pandas as pd


DEFAULT_EXCLUDE_YEARS = (2012, 2022)


@dataclass(frozen=True)
class DayOfYearWindow:
    """
    Inclusive day-of-year window [start, end].

    Parameters
    ----------
    start : int
        Inclusive start day-of-year (1..366).
    end : int
        Inclusive end day-of-year (1..366).

    Notes
    -----
    - This class does not enforce any business rules about overlap with other
      windows. That is handled by validation utilities.
    """
    start: int
    end: int


def _coerce_datetime(df: pd.DataFrame, col: str = "timestep") -> pd.DataFrame:
    """
    Ensure that a dataframe column is parsed as pandas datetime.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe containing a timestamp column.
    col : str, optional
        Name of the column to coerce to datetime. Default is "timestep".

    Returns
    -------
    pandas.DataFrame
        DataFrame with `col` converted to pandas datetime dtype.

    Raises
    ------
    ValueError
        If the specified column does not exist or cannot be parsed as datetime.
    """
    if col not in df.columns:
        raise ValueError(
            f"Missing required column '{col}'. Found columns: {list(df.columns)}"
        )
    df[col] = pd.to_datetime(df[col], errors="raise", utc=False)
    return df


def add_dayofyear(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a `dayofyear` column derived from the `timestep` column.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing a datetime-like `timestep` column.

    Returns
    -------
    pandas.DataFrame
        Same dataframe with an additional integer column `dayofyear` in [1, 366].
    """
    df["dayofyear"] = df["timestep"].dt.dayofyear
    return df


def _validate_window_bounds(window: DayOfYearWindow, name: str) -> None:
    """
    Validate that a window lies within [1, 366] and start <= end.

    Parameters
    ----------
    window : DayOfYearWindow
        Window to validate.
    name : str
        Human-readable name used in error messages.

    Raises
    ------
    ValueError
        If bounds are invalid.
    """
    if window.start < 1 or window.end < 1 or window.start > 366 or window.end > 366:
        raise ValueError(
            f"{name} must be within dayofyear 1..366. Got [{window.start}, {window.end}]."
        )
    if window.start > window.end:
        raise ValueError(
            f"{name} start must be <= end. Got [{window.start}, {window.end}]."
        )


def _validate_no_overlap(windows: List[Tuple[str, DayOfYearWindow]]) -> None:
    """
    Validate that multiple inclusive windows do not overlap.

    Parameters
    ----------
    windows : list of (str, DayOfYearWindow)
        Named windows to check.

    Raises
    ------
    ValueError
        If any pair overlaps.
    """
    # Convert to list of (name, start, end) and sort by start then end.
    spans = sorted([(n, w.start, w.end) for n, w in windows], key=lambda x: (x[1], x[2]))
    for i in range(len(spans) - 1):
        name_a, a_start, a_end = spans[i]
        name_b, b_start, b_end = spans[i + 1]
        # Inclusive overlap check: overlap exists if next.start <= current.end
        if b_start <= a_end:
            raise ValueError(
                f"Day-of-year windows overlap: {name_a} [{a_start}, {a_end}] "
                f"and {name_b} [{b_start}, {b_end}]. Windows must be disjoint."
            )


def _window_mask(dayofyear: pd.Series, window: DayOfYearWindow) -> pd.Series:
    """
    Build a boolean mask selecting rows whose dayofyear falls in a window.

    Parameters
    ----------
    dayofyear : pandas.Series
        Series of integers (day-of-year).
    window : DayOfYearWindow
        Inclusive window [start, end].

    Returns
    -------
    pandas.Series
        Boolean mask where True indicates membership in the window.
    """
    return (dayofyear >= window.start) & (dayofyear <= window.end)


def split_indices_by_windows(
    df: pd.DataFrame,
    pad1_window: DayOfYearWindow,
    val_window: DayOfYearWindow,
    test_window: DayOfYearWindow,
    pad2_window: DayOfYearWindow,
    exclude_years: tuple[int, ...] = DEFAULT_EXCLUDE_YEARS,
    only_present: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split a dataframe into training, validation, and test indices using
    explicit day-of-year windows.

    Definitions (all inclusive):
      - pad1_window: excluded from all splits
      - val_window : validation split
      - test_window: test split
      - pad2_window: excluded from all splits
      - training   : remainder (not in any of the above windows)

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe containing at least `timestep` and optionally `present`
        and `dayofyear`.
    pad1_window : DayOfYearWindow
        First padding window (excluded).
    val_window : DayOfYearWindow
        Validation window.
    test_window : DayOfYearWindow
        Test window.
    pad2_window : DayOfYearWindow
        Second padding window (excluded).
    exclude_years : tuple[int, ...], optional
        Years to remove entirely from all splits (default: 2012, 2022).
    only_present : bool, optional
        If True, keep only rows where `present == 1`.

    Returns
    -------
    (train_df, val_df, test_df) : tuple of pandas.DataFrame
        Train, validation, and test splits. Each is sorted by (timestep, path)
        and has a reset integer index.

    Raises
    ------
    ValueError
        If windows are invalid, overlap, or required columns are missing.
    """
    # Validate window bounds and overlap
    _validate_window_bounds(pad1_window, "pad1_window")
    _validate_window_bounds(val_window, "val_window")
    _validate_window_bounds(test_window, "test_window")
    _validate_window_bounds(pad2_window, "pad2_window")
    _validate_no_overlap(
        [
            ("pad1_window", pad1_window),
            ("val_window", val_window),
            ("test_window", test_window),
            ("pad2_window", pad2_window),
        ]
    )

    # Optionally restrict to samples that exist on disk
    if only_present:
        if "present" not in df.columns:
            raise ValueError("only_present=True requires a 'present' column in the input CSV.")
        df = df.loc[df["present"] == 1].copy()

    # Remove excluded years
    for year in exclude_years:
        df = df.loc[df["timestep"].dt.year != year].copy()

    # Ensure dayofyear exists
    if "dayofyear" not in df.columns:
        df = add_dayofyear(df)

    doy = df["dayofyear"]

    pad1_mask = _window_mask(doy, pad1_window)
    val_mask = _window_mask(doy, val_window)
    test_mask = _window_mask(doy, test_window)
    pad2_mask = _window_mask(doy, pad2_window)

    excluded_mask = pad1_mask | pad2_mask
    assigned_mask = val_mask | test_mask | excluded_mask

    df_val = df.loc[val_mask].copy()
    df_test = df.loc[test_mask].copy()
    df_train = df.loc[~assigned_mask].copy()

    # Stable ordering
    df_train = df_train.sort_values(["timestep", "path"]).reset_index(drop=True)
    df_val = df_val.sort_values(["timestep", "path"]).reset_index(drop=True)
    df_test = df_test.sort_values(["timestep", "path"]).reset_index(drop=True)

    return df_train, df_val, df_test


def _parse_window_arg(values: List[int], name: str) -> DayOfYearWindow:
    """
    Parse a CLI argument of the form: <start> <end> into a DayOfYearWindow.

    Parameters
    ----------
    values : list[int]
        Two-element list containing [start, end].
    name : str
        Window name used in error messages.

    Returns
    -------
    DayOfYearWindow
        Parsed window.

    Raises
    ------
    ValueError
        If the list does not contain exactly two integers.
    """
    if len(values) != 2:
        raise ValueError(f"{name} must be provided as two integers: START END.")
    return DayOfYearWindow(start=int(values[0]), end=int(values[1]))


def main() -> None:
    """
    Command-line entry point.

    Loads the CSV index, adds `dayofyear`, performs the split using
    explicit user-defined windows, and writes train/val/test CSVs to disk.
    """
    parser = argparse.ArgumentParser(
        description="Add dayofyear and build train/val/test CSV indices using explicit windows."
    )
    parser.add_argument(
        "--input_csv",
        required=True,
        type=str,
        help="CSV produced by create_csv_index.py",
    )
    parser.add_argument(
        "--out_dir",
        required=True,
        type=str,
        help="Directory where split CSVs will be written",
    )
    parser.add_argument(
        "--only_present",
        action="store_true",
        help="Keep only rows where present == 1",
    )
    parser.add_argument(
        "--exclude_years",
        nargs="*",
        type=int,
        default=list(DEFAULT_EXCLUDE_YEARS),
        help="Years to exclude entirely from all splits",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="",
        help="Optional filename prefix (e.g. 'lz4_')",
    )

    # User-defined windows with requested defaults
    parser.add_argument(
        "--pad1_window",
        nargs=2,
        type=int,
        default=[1, 15],
        metavar=("START", "END"),
        help="First padding window (excluded). Default: 1 15",
    )
    parser.add_argument(
        "--val_window",
        nargs=2,
        type=int,
        default=[16, 30],
        metavar=("START", "END"),
        help="Validation window. Default: 16 30",
    )
    parser.add_argument(
        "--test_window",
        nargs=2,
        type=int,
        default=[31, 45],
        metavar=("START", "END"),
        help="Test window. Default: 31 45",
    )
    parser.add_argument(
        "--pad2_window",
        nargs=2,
        type=int,
        default=[46, 60],
        metavar=("START", "END"),
        help="Second padding window (excluded). Default: 46 60",
    )

    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pad1_window = _parse_window_arg(args.pad1_window, "--pad1_window")
    val_window = _parse_window_arg(args.val_window, "--val_window")
    test_window = _parse_window_arg(args.test_window, "--test_window")
    pad2_window = _parse_window_arg(args.pad2_window, "--pad2_window")

    # Load CSV
    df = pd.read_csv(input_csv)

    # Drop common index artifact if present
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    # Normalize timestamps and enrich dataframe
    df = _coerce_datetime(df, "timestep")
    df = add_dayofyear(df)

    # Perform split
    df_train, df_val, df_test = split_indices_by_windows(
        df=df,
        pad1_window=pad1_window,
        val_window=val_window,
        test_window=test_window,
        pad2_window=pad2_window,
        exclude_years=tuple(args.exclude_years),
        only_present=args.only_present,
    )

    # Write outputs
    train_path = out_dir / f"{args.prefix}train.csv"
    val_path = out_dir / f"{args.prefix}val.csv"
    test_path = out_dir / f"{args.prefix}test.csv"

    df_train.to_csv(train_path, index=False)
    df_val.to_csv(val_path, index=False)
    df_test.to_csv(test_path, index=False)

    print("Wrote splits:")
    print(f"  train: {train_path} ({len(df_train):,} rows)")
    print(f"  val:   {val_path} ({len(df_val):,} rows)")
    print(f"  test:  {test_path} ({len(df_test):,} rows)")
    print("Windows (inclusive):")
    print(f"  pad1: {pad1_window.start}..{pad1_window.end} (excluded)")
    print(f"  val:  {val_window.start}..{val_window.end}")
    print(f"  test: {test_window.start}..{test_window.end}")
    print(f"  pad2: {pad2_window.start}..{pad2_window.end} (excluded)")


if __name__ == "__main__":
    main()
