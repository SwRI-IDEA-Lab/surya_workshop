"""
Per-epoch validation predictions, and a best-epoch figure for WandB.

Two jobs, deliberately in one callback because they share the same buffer:

1. **Every validation epoch**, every per-sample prediction is written to
   ``val_predictions.csv`` (one row per sample per epoch, keyed by the Surya frame). That
   file is the raw material for any plot you want to make afterwards, and it survives the
   run whether or not WandB is enabled.
2. **For the best epoch only** — the epoch that minimized ``monitor``, i.e. the epoch whose
   weights ``ModelCheckpoint`` kept — a two-panel figure of per-sample true vs. predicted
   bars, non-event samples on the left and event samples on the right, is written to disk
   and logged to WandB. RMSE is reported per panel and overall, inside the image.

The figure is rendered once at ``on_fit_end`` rather than on every improvement, so a run
logs exactly one image and it always corresponds to the saved checkpoint.

Why a callback rather than code in the LightningModule: ``SepPspLightningModule`` is
generic (it knows about losses and metrics, not about SEP events), while the event /
non-event split plotted here is specific to this task. The module's only contribution is
that ``validation_step`` returns its predictions.

DDP: predictions are all-gathered before anything is written, so the CSV and the figure
cover the whole validation set and not one rank's shard. Rows are de-duplicated on the
Surya timestamp, which removes the samples DistributedSampler repeats to even out ranks.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Optional

import matplotlib

matplotlib.use("Agg")  # a training run has no display; must be set before pyplot is imported

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from lightning.pytorch.callbacks import Callback

# Slots 1 and 2 of the workshop categorical palette. Validated as a pair against the light
# surface: CVD ΔE 24.7 (protan), normal-vision ΔE 33.6, both ≥3:1 contrast — so the two
# series stay distinguishable in print, in grayscale and for colorblind readers.
TRUE_COLOR = "#2a78d6"
PRED_COLOR = "#eb6834"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#52514e"

# An MSE-trained regressor is not constrained to the label's sign or range, so with a log
# y axis (log_y=True, i.e. a label still in its own positive units) anything <= 0 has to be
# clipped to a floor and reported, rather than silently dropped by the log.
_LOG_FLOOR = 1e-6


def _rmse(target: np.ndarray, pred: np.ndarray) -> float:
    """Root-mean-square error in whatever units the label is in."""
    if target.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((target - pred) ** 2)))


def plot_event_predictions(
    df: pd.DataFrame,
    *,
    label_name: str = "Jlinlin",
    subtitle: str = "",
    log_y: bool = False,
) -> plt.Figure:
    """Two panels of per-sample true-vs-predicted bars, non-events left and events right.

    One pair of bars per validation sample: the true label and the model's prediction,
    side by side, so each sample can be read off individually. Samples are sorted by true
    value, which turns the true series into a monotone staircase -- any departure of the
    predicted bar from it is the error on that sample, and a model that ignores its input
    shows up immediately as a flat orange line under a rising blue one.

    Args:
        df: One row per validation sample, with ``is_event`` (0/1), ``target`` and ``pred``.
        label_name: Name of the predicted quantity, used on the y axis.
        subtitle: Line under the title — the caller puts the epoch and its score here.
        log_y: True when the label is in its own positive units (raw Jlinlin), which needs
            a log y axis; values <= 0 are then clipped to a floor and counted in the panel.
            False when the label is already log-scaled (this app's log10-z-scored target),
            where values are legitimately negative and the axis stays linear.

    Returns:
        The figure. The caller owns it and is responsible for closing it.

    Bars are grouped rather than overlaid with transparency: an alpha-blended overlap
    invents a third color and hides which series is taller.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    fig.patch.set_facecolor(SURFACE)

    finite = np.concatenate([df["target"].to_numpy(float), df["pred"].to_numpy(float)])
    finite = finite[np.isfinite(finite)]
    if log_y:
        floor = max(_LOG_FLOOR, float(finite[finite > 0].min()) / 2) if (finite > 0).any() else _LOG_FLOOR
        top = float(finite.max()) if finite.size else 1.0
    else:
        lo = float(finite.min()) if finite.size else -1.0
        hi = float(finite.max()) if finite.size else 1.0
        pad = 0.08 * max(hi - lo, 1e-9)

    for ax, (flag, name) in zip(axes, [(0, "Non-event"), (1, "Event")]):
        # Sorted by true value: the blue series becomes a staircase the orange one can be
        # read against. Sorting is per panel, so the x axis is a rank, not a sample id.
        sub = df[df["is_event"] == flag].sort_values("target").reset_index(drop=True)
        target = sub["target"].to_numpy(float)
        pred = sub["pred"].to_numpy(float)
        x = np.arange(len(sub))

        n_nonpositive = 0
        if log_y:
            n_nonpositive = int((pred <= 0).sum())
            target = np.clip(target, floor, None)
            pred = np.clip(pred, floor, None)
            base = floor  # bars grow from the floor, not from 0, which a log axis cannot show
        else:
            base = 0.0

        ax.bar(x - 0.21, target, width=0.40, bottom=0 if log_y else None,
               color=TRUE_COLOR, label=f"True {label_name}")
        ax.bar(x + 0.21, pred, width=0.40, bottom=0 if log_y else None,
               color=PRED_COLOR, label="Predicted")
        if log_y:
            ax.set_yscale("log")
            ax.set_ylim(floor, top * 1.35)
        else:
            ax.set_ylim(lo - pad, hi + pad * 2.2)
            if lo < 0 < hi:
                ax.axhline(0, color="#c9c8c3", linewidth=0.9, zorder=0)

        ax.set_title(
            f"{name}  (n={len(sub)})    RMSE {_rmse(sub['target'].to_numpy(float), sub['pred'].to_numpy(float)):.4g}",
            color=INK, fontsize=12, pad=10,
        )
        ax.set_xlabel("validation samples, sorted by true value", color=INK_MUTED, fontsize=10)
        ax.set_xlim(-0.8, max(len(sub) - 0.2, 0.8))
        # One tick every 5 samples, numbered from 1: 25 individual labels would collide.
        ticks = np.arange(0, len(sub), 5)
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(int(t) + 1) for t in ticks])
        ax.set_facecolor(SURFACE)
        ax.grid(axis="y", color="#e4e3df", linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#d6d5d0")
        ax.tick_params(colors=INK_MUTED, labelsize=9)

        if n_nonpositive:
            ax.text(
                0.02, 0.97, f"{n_nonpositive} prediction(s) <= 0, clipped to the axis floor",
                transform=ax.transAxes, va="top", ha="left",
                fontsize=8, color=INK_MUTED,
                bbox=dict(facecolor=SURFACE, edgecolor="none", pad=2.0),
            )

    axes[0].set_ylabel(label_name, color=INK_MUTED, fontsize=10)

    overall = _rmse(df["target"].to_numpy(float), df["pred"].to_numpy(float))
    fig.suptitle(
        f"Validation predictions — overall RMSE {overall:.4g}",
        color=INK, fontsize=14, y=0.99,
    )
    if subtitle:
        fig.text(0.5, 0.905, subtitle, ha="center", color=INK_MUTED, fontsize=10)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", ncol=2, frameon=False,
        fontsize=10, labelcolor=INK_MUTED, bbox_to_anchor=(0.5, -0.01),
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.88))
    return fig


class ValidationPredictionLogger(Callback):
    """Save validation predictions every epoch; plot and log the best epoch's.

    Args:
        output_dir: Where ``val_predictions.csv`` and ``val_predictions_best.png`` go.
            Defaults to ``<trainer.default_root_dir>/val_predictions``.
        monitor: The logged metric that defines "best". Use the same key
            ``ModelCheckpoint`` monitors, or the figure will describe a different epoch
            from the one whose weights were kept.
        mode: ``"min"`` or ``"max"``, matching ``monitor``.
        label_name: Name of the predicted quantity, for axis labels.
        log_y: True for a positive linear label (raw Jlinlin), False when the label is
            already log-scaled (this app's log10-z-scored target). See
            ``plot_event_predictions``.
        image_key: WandB key the figure is logged under.
    """

    def __init__(
        self,
        output_dir: Optional[str | Path] = None,
        monitor: str = "val_loss",
        mode: str = "min",
        label_name: str = "Jlinlin",
        log_y: bool = False,
        image_key: str = "val_prediction_histograms",
    ):
        super().__init__()
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got {mode!r}")
        self.output_dir = Path(output_dir) if output_dir is not None else None
        self.monitor = monitor
        self.mode = mode
        self.label_name = label_name
        self.log_y = log_y
        self.image_key = image_key

        self._batches: list[dict[str, torch.Tensor]] = []
        self._best_score: Optional[float] = None
        self._best_epoch: Optional[int] = None
        self._best_df: Optional[pd.DataFrame] = None
        self._csv_path: Optional[Path] = None

    # -- paths ---------------------------------------------------------------------

    def _resolve_dir(self, trainer) -> Path:
        base = self.output_dir or Path(trainer.default_root_dir) / "val_predictions"
        base.mkdir(parents=True, exist_ok=True)
        return base

    def on_fit_start(self, trainer, pl_module) -> None:
        """Start a fresh CSV, so a rerun does not append to the previous run's rows."""
        self._best_score = self._best_epoch = self._best_df = None
        if trainer.is_global_zero:
            self._csv_path = self._resolve_dir(trainer) / "val_predictions.csv"
            self._csv_path.unlink(missing_ok=True)

    # -- collection ----------------------------------------------------------------

    def on_validation_epoch_start(self, trainer, pl_module) -> None:
        self._batches = []

    def on_validation_batch_end(
        self, trainer, pl_module, outputs: Any, batch: dict, batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if trainer.sanity_checking:
            return
        preds = outputs["preds"] if isinstance(outputs, dict) else outputs
        if preds is None:
            return

        device = pl_module.device
        # Timestamps travel as int64 nanoseconds: all_gather moves tensors, not strings.
        stamps = pd.to_datetime(list(batch["valid_index"])).to_numpy().astype("int64")
        self._batches.append({
            "pred": preds.detach().reshape(-1).float().to(device),
            "target": batch["forecast"].detach().reshape(-1).float().to(device),
            "is_event": batch["is_event"].detach().reshape(-1).long().to(device),
            "stamp_ns": torch.as_tensor(stamps, dtype=torch.long, device=device),
        })

    def _gather(self, pl_module, key: str) -> np.ndarray:
        """Concatenate this rank's batches, then all-gather across ranks."""
        local = torch.cat([b[key] for b in self._batches])
        # all_gather is a no-op on a single device and must run on every rank, so it is
        # called before any is_global_zero guard.
        return pl_module.all_gather(local).reshape(-1).cpu().numpy()

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if trainer.sanity_checking or not self._batches:
            return

        frame = pd.DataFrame({
            "epoch": trainer.current_epoch,
            "valid_index": pd.to_datetime(self._gather(pl_module, "stamp_ns")),
            "is_event": self._gather(pl_module, "is_event").astype(int),
            "target": self._gather(pl_module, "target").astype(float),
            "pred": self._gather(pl_module, "pred").astype(float),
        })
        # DistributedSampler pads the last batch by repeating samples so every rank gets
        # the same count; those repeats are identical rows and must not be counted twice.
        frame = frame.drop_duplicates("valid_index").sort_values("valid_index")
        self._batches = []

        if not trainer.is_global_zero:
            return

        if self._csv_path is not None:
            frame.to_csv(
                self._csv_path, mode="a", index=False, header=not self._csv_path.exists()
            )

        score = trainer.callback_metrics.get(self.monitor)
        if score is None:
            warnings.warn(
                f"ValidationPredictionLogger: {self.monitor!r} is not in callback_metrics "
                f"({sorted(trainer.callback_metrics)}); cannot pick a best epoch."
            )
            return
        score = float(score)

        better = (
            self._best_score is None
            or (score < self._best_score if self.mode == "min" else score > self._best_score)
        )
        if better:
            self._best_score, self._best_epoch, self._best_df = score, trainer.current_epoch, frame

    # -- reporting -----------------------------------------------------------------

    def on_fit_end(self, trainer, pl_module) -> None:
        if not trainer.is_global_zero:
            return
        if self._best_df is None:
            warnings.warn("ValidationPredictionLogger: no validation epoch recorded; no figure.")
            return

        fig = plot_event_predictions(
            self._best_df,
            label_name=self.label_name,
            log_y=self.log_y,
            subtitle=(
                f"best epoch {self._best_epoch} of {trainer.current_epoch}"
                f"  ·  {self.monitor} {self._best_score:.6g}"
                f"  ·  {len(self._best_df)} validation samples"
            ),
        )
        try:
            png = self._resolve_dir(trainer) / "val_predictions_best.png"
            fig.savefig(png, dpi=150, facecolor=SURFACE, bbox_inches="tight")
            print(f"[VAL] Best-epoch histogram: {png}")
            if self._csv_path is not None:
                print(f"[VAL] Per-epoch predictions: {self._csv_path}")
            self._log_to_wandb(trainer, fig)
        finally:
            plt.close(fig)

    def _log_to_wandb(self, trainer, fig) -> None:
        """Log the figure to every WandB logger attached to the trainer, if any."""
        from lightning.pytorch.loggers import WandbLogger

        wandb_loggers = [lg for lg in trainer.loggers if isinstance(lg, WandbLogger)]
        if not wandb_loggers:
            return
        import wandb

        for logger in wandb_loggers:
            logger.experiment.log({
                self.image_key: wandb.Image(fig),
                f"{self.image_key}/best_epoch": self._best_epoch,
                f"{self.image_key}/rmse": _rmse(
                    self._best_df["target"].to_numpy(float),
                    self._best_df["pred"].to_numpy(float),
                ),
            })
