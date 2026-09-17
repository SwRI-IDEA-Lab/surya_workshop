"""Label normalization for the SEP/PSP task.

Jlinlin spans ~6 decades (2e-4 to 671 for ``Jlinlin_raw``) with a quiet-time peak near
0.02 and a long event tail. Trained directly on those values, MSE is dominated by a
handful of samples -- in a 50-sample training draw the three largest carry 94% of the
total squared deviation -- so the gradient is almost entirely "chase the biggest event"
and the other 47 samples contribute nothing. Hence: log10, then z-score.

Two properties this module exists to guarantee:

- **The statistics come from the training split only.** Computing them over the whole
  catalog would let the validation and test periods set the target scale.
- **One transform object is shared by every split and every run of a learning curve.**
  Build it once (see ``build_datasets()``) and pass it to both datasets. Statistics
  re-derived per split -- or per subsample of a 50/100/200/500 ladder -- would put each
  run's val_loss on a different scale and make the curve meaningless.

``LogStandardizer`` is invertible, so predictions can be returned to physical units:
``standardizer.inverse(pred)`` gives Jlinlin back.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LogStandardizer:
    """log10 followed by a z-score, with the statistics fixed at construction.

    Attributes:
        mean: Mean of log10(label) over the training split.
        std: Standard deviation of log10(label) over the training split.
        column: Catalog column the statistics were computed from, for the record.
        n_rows: How many catalog rows went into them, for the record.
    """

    mean: float
    std: float
    column: str = ""
    n_rows: int = 0

    def __call__(self, values: pd.Series) -> pd.Series:
        """Transform a label column. This is the ``label_transform`` callable."""
        return (np.log10(values) - self.mean) / self.std

    def inverse(self, values):
        """Map standardized values back to the label's own units."""
        return np.power(10.0, np.asarray(values, dtype=float) * self.std + self.mean)

    def __str__(self) -> str:
        return (
            f"log10({self.column or 'label'}) z-scored with mean={self.mean:.4f}, "
            f"std={self.std:.4f} (from {self.n_rows:,} training-split catalog rows)"
        )


def build_log_standardizer(
    catalog_path: str,
    train_index_path: str,
    column: str = "Jlinlin",
    time_column: str = "SHARP time",
) -> LogStandardizer:
    """Fit a :class:`LogStandardizer` on the training split's catalog rows.

    The catalog covers every year; the training index covers only the training years. A
    catalog row belongs to the training split when its ``time_column`` falls on a date the
    training index contains -- the same day-level correspondence the Surya matching uses,
    which is what keeps the validation and test periods out of the statistics.

    Args:
        catalog_path: The SHARP-to-PSP CSV holding the labels.
        train_index_path: The training split's Surya index CSV.
        column: Label column to fit on (``Jlinlin`` or ``Jlinlin_raw``).
        time_column: Catalog column holding the SDO/SHARP timestamp.

    Returns:
        A fitted ``LogStandardizer``.

    Raises:
        ValueError: If the column is missing, holds non-positive values (log10 is
            undefined there), or no catalog row falls inside the training split.
    """
    catalog = pd.read_csv(catalog_path)
    if column not in catalog.columns:
        raise ValueError(
            f"Label column {column!r} is not in {catalog_path}. "
            f"Available columns: {sorted(catalog.columns)}"
        )

    train_days = pd.to_datetime(
        pd.read_csv(train_index_path, usecols=["timestep"])["timestep"], format="mixed"
    ).dt.normalize().unique()

    catalog_days = pd.to_datetime(catalog[time_column], format="mixed").dt.normalize()
    values = catalog.loc[catalog_days.isin(train_days), column]

    if len(values) == 0:
        raise ValueError(
            f"No row of {catalog_path} falls on a date present in {train_index_path}; "
            "cannot fit label statistics."
        )
    if (values <= 0).any():
        raise ValueError(
            f"{column!r} holds {(values <= 0).sum()} non-positive value(s) in the training "
            "split; log10 is undefined there. Clip or drop them before fitting."
        )

    log_values = np.log10(values)
    return LogStandardizer(
        mean=float(log_values.mean()),
        std=float(log_values.std()),
        column=column,
        n_rows=len(values),
    )
