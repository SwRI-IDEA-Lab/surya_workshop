"""Plot the SEP intensity series in SHARP_to_PSP_times.csv, start to end.

Diagnostic only -- it reads one CSV, writes a PNG, and changes nothing. No event list, no
sampling, no model: just the label column against PSP time, so the series the task learns
from can be looked at directly.

The CSV holds one row per (PSP hour, active region), so the label repeats across every
SHARP row of the same hour; the plot uses one point per hour.

Usage (from the repo root):

    python -m downstream_apps.sep_pred_psp.check_data
    python -m downstream_apps.sep_pred_psp.check_data --column both
    python -m downstream_apps.sep_pred_psp.check_data --start 2022-01-01 --end 2022-08-01
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display on a training node; must precede pyplot

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_CSV = Path(__file__).parent / "data" / "SHARP_to_PSP_times.csv"
TIME_COLUMN = "time"  # PSP time; "SHARP time" is the solar-surface counterpart

COLORS = {"Jlinlin": "#2a78d6", "Jlinlin_raw": "#eb6834"}

# Threshold colors. Green marks the high tail (an event-sized intensity), red the quiet
# floor; everything between keeps the neutral series color.
HIGH_COLOR = "#2e8b3d"
LOW_COLOR = "#d62728"
LINE_COLOR = "#b9b8b4"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="SHARP-to-PSP CSV to read.")
    parser.add_argument(
        "--column", default="Jlinlin", choices=["Jlinlin", "Jlinlin_raw", "both"],
        help="Which intensity column to plot (default: Jlinlin).",
    )
    parser.add_argument("--start", type=pd.Timestamp, default=None, help="Optional start of the window to plot.")
    parser.add_argument("--end", type=pd.Timestamp, default=None, help="Optional end of the window to plot.")
    parser.add_argument("--high", type=float, default=1.0, help="Values above this are green (default: 1).")
    parser.add_argument("--low", type=float, default=0.1, help="Values below this are red (default: 0.1).")
    parser.add_argument(
        "--linear", action="store_true",
        help="Linear y axis. The default is log: the series spans ~6 decades.",
    )
    parser.add_argument(
        "--out", type=Path, default=Path(__file__).parent / "check_data_series.png",
        help="Where to write the figure.",
    )
    return parser.parse_args()


def load_series(csv: Path, columns: list[str], start, end) -> pd.DataFrame:
    """One row per PSP hour, sorted, optionally restricted to [start, end]."""
    df = pd.read_csv(csv, usecols=[TIME_COLUMN] + columns)
    df[TIME_COLUMN] = pd.to_datetime(df[TIME_COLUMN], format="mixed")
    # One row per hour: the label is a property of the hour, repeated across that hour's
    # active regions (~9 rows, up to 22).
    df = df.drop_duplicates(TIME_COLUMN).sort_values(TIME_COLUMN).reset_index(drop=True)
    if start is not None:
        df = df[df[TIME_COLUMN] >= start]
    if end is not None:
        df = df[df[TIME_COLUMN] <= end]
    if df.empty:
        raise SystemExit("No rows in that time window.")
    return df


def describe(df: pd.DataFrame, columns: list[str]) -> None:
    """Print what the plot cannot show: coverage, gaps, and the value distribution."""
    t = df[TIME_COLUMN]
    step = t.diff().dt.total_seconds().div(3600)
    print(f"rows (unique PSP hours): {len(df):,}")
    print(f"span: {t.min()}  ->  {t.max()}  ({(t.max() - t.min()).days:,} days)")
    print(f"cadence (hours): median {step.median():.2f}, "
          f"{(step > 1.01).sum():,} gaps > 1 h, largest {step.max():,.0f} h")
    for c in columns:
        x = df[c]
        print(f"\n{c}: min {x.min():.3g}  median {x.median():.3g}  max {x.max():.3g}  "
              f"| non-positive {int((x <= 0).sum())}  NaN {int(x.isna().sum())}")
        q = x.quantile([0.1, 0.5, 0.9, 0.99])
        print("  percentiles  " + "  ".join(f"p{int(p*100)} {v:.3g}" for p, v in q.items()))


def plot_series(
    df: pd.DataFrame, columns: list[str], out: Path, log_y: bool,
    high: float | None = None, low: float | None = None,
) -> Path:
    """Plot the series; with one column, points are colored by the two thresholds.

    The line stays neutral and the coloring goes on the points: at 17k hourly samples a
    colored line segment would be decided by whichever endpoint drew last, while a point
    says exactly which hour crossed a threshold.
    """
    fig, ax = plt.subplots(figsize=(16, 5))
    color_by_threshold = len(columns) == 1 and high is not None and low is not None

    for c in columns:
        ax.plot(df[TIME_COLUMN], df[c], linewidth=0.4,
                color=LINE_COLOR if color_by_threshold else COLORS.get(c, "#52514e"),
                label=None if color_by_threshold else c,
                alpha=0.9 if len(columns) == 1 else 0.75, zorder=1)

    if color_by_threshold:
        c = columns[0]
        t, v = df[TIME_COLUMN], df[c]
        for mask, color, label in [
            (v > high, HIGH_COLOR, f"{c} > {high:g}"),
            (v < low, LOW_COLOR, f"{c} < {low:g}"),
        ]:
            ax.scatter(t[mask], v[mask], s=3, color=color, label=f"{label}  (n={int(mask.sum()):,})",
                       zorder=3, linewidths=0)
        for level in (high, low):
            ax.axhline(level, color="#8a8984", linestyle="--", linewidth=0.8, zorder=2)
        ax.legend(frameon=False, markerscale=3, loc="upper left")
        between = int(((v >= low) & (v <= high)).sum())
        print(f"\n{c}: {int((v > high).sum()):,} hours above {high:g} (green), "
              f"{int((v < low).sum()):,} below {low:g} (red), {between:,} between.")

    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel("PSP time")
    ax.set_ylabel(" / ".join(columns) + (" (log)" if log_y else ""))
    ax.grid(alpha=0.3, linewidth=0.6)
    ax.margins(x=0.01)
    t = df[TIME_COLUMN]
    ax.set_title(
        f"{' and '.join(columns)} — {t.min():%Y-%m-%d} to {t.max():%Y-%m-%d} "
        f"({len(df):,} unique PSP hours)",
        fontsize=13,
    )
    if len(columns) > 1:
        ax.legend(frameon=False)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, facecolor="white")
    plt.close(fig)
    print(f"\n[out] {out}")
    return out


def main() -> None:
    args = parse_args()
    columns = ["Jlinlin", "Jlinlin_raw"] if args.column == "both" else [args.column]
    df = load_series(args.csv, columns, args.start, args.end)
    describe(df, columns)
    plot_series(df, columns, args.out, log_y=not args.linear, high=args.high, low=args.low)


if __name__ == "__main__":
    main()
