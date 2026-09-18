"""
Build the train / val / test Surya indices for the SEP-from-PSP task.

Surya's own val and test indices only cover Jan 16-30 and Jan 31-Feb 14 of each year, and
no PSP/ISOIS event falls in them. Instead, the splits are carved from Surya's full index
by date, restricted to the PSP era:

    train: 2020, 2021, 2023, 2024 (every date, including January)
    val:   2022-01-01 .. 2022-07-31
    test:  2022-08-01 .. 2022-12-31

2022 is held out of Surya's own splits entirely, so it is unseen by the backbone too.

Run from the repo root:
    python -m downstream_apps.sep_pred_psp.data.make_split_indices
"""

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
FULL_INDEX = REPO_ROOT / "data" / "indices" / "surya_aws_s3_full_index.csv"
OUT_DIR = Path(__file__).parent / "indices"

TRAIN_YEARS = [2020, 2021, 2023, 2024]
VAL_RANGE = ("2022-01-01", "2022-08-01")   # [start, end)
TEST_RANGE = ("2022-08-01", "2023-01-01")  # [start, end)


def main() -> None:
    index = pd.read_csv(FULL_INDEX)[["path", "timestep", "present"]]
    t = pd.to_datetime(index["timestep"])

    splits = {
        "train": t.dt.year.isin(TRAIN_YEARS),
        "val": (t >= VAL_RANGE[0]) & (t < VAL_RANGE[1]),
        "test": (t >= TEST_RANGE[0]) & (t < TEST_RANGE[1]),
    }

    OUT_DIR.mkdir(exist_ok=True)
    for name, mask in splits.items():
        out = OUT_DIR / f"sep_psp_{name}.csv"
        index[mask].to_csv(out, index=False)
        print(f"{name}: {int(mask.sum())} timesteps "
              f"({t[mask].min()} -> {t[mask].max()}) -> {out}")


if __name__ == "__main__":
    main()
