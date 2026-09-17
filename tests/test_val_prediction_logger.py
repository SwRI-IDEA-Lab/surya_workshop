"""Tests for per-epoch validation prediction saving and the best-epoch figure.

Runs a real (tiny, CPU-only) Lightning fit so the callback is exercised through the same
hooks it will see in training, rather than by calling its methods by hand.
"""

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

import lightning as L

from downstream_apps.sep_pred_psp.lightning_modules.val_prediction_logger import (
    ValidationPredictionLogger,
    plot_event_predictions,
)

N_PER_CLASS = 25


class TinyValDataset(Dataset):
    """25 non-event + 25 event samples, shaped like the batch contract of the real app."""

    def __init__(self):
        rng = np.random.default_rng(0)
        self.is_event = np.array([0] * N_PER_CLASS + [1] * N_PER_CLASS)
        self.target = np.where(self.is_event == 1, 10 ** rng.normal(0.3, 0.5, 2 * N_PER_CLASS),
                               10 ** rng.normal(-1.7, 0.4, 2 * N_PER_CLASS))
        self.stamps = pd.date_range("2022-01-01", periods=2 * N_PER_CLASS, freq="12min")

    def __len__(self):
        return len(self.target)

    def __getitem__(self, i):
        return {
            "ts": torch.tensor([float(self.target[i])], dtype=torch.float32),
            "forecast": np.float32(self.target[i]),
            "is_event": int(self.is_event[i]),
            "valid_index": self.stamps[i].isoformat(),
        }


class TinyModule(L.LightningModule):
    """Mirrors the real module's contract: logs val_loss and returns {"preds": ...}."""

    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(1, 1)

    def _step(self, batch):
        out = self.layer(batch["ts"])
        return out, torch.nn.functional.mse_loss(out.reshape(-1), batch["forecast"].reshape(-1))

    def training_step(self, batch, batch_idx):
        return self._step(batch)[1]

    def validation_step(self, batch, batch_idx):
        out, loss = self._step(batch)
        self.log("val_loss", loss, batch_size=len(batch["forecast"]))
        return {"preds": out.detach()}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-2)


def run_fit(tmp_path, max_epochs=3):
    loader = DataLoader(TinyValDataset(), batch_size=8, drop_last=False)
    callback = ValidationPredictionLogger(output_dir=tmp_path, monitor="val_loss", mode="min")
    trainer = L.Trainer(
        max_epochs=max_epochs, accelerator="cpu", devices=1, logger=False,
        enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False,
        num_sanity_val_steps=0, callbacks=[callback],
    )
    trainer.fit(TinyModule(), loader, loader)
    return callback


def test_every_epoch_is_saved_with_every_validation_sample(tmp_path):
    run_fit(tmp_path, max_epochs=3)
    saved = pd.read_csv(tmp_path / "val_predictions.csv")
    assert sorted(saved["epoch"].unique()) == [0, 1, 2]
    for epoch, rows in saved.groupby("epoch"):
        assert len(rows) == 2 * N_PER_CLASS, f"epoch {epoch} lost samples"
        assert (rows["is_event"] == 1).sum() == N_PER_CLASS
        assert (rows["is_event"] == 0).sum() == N_PER_CLASS
        assert rows["valid_index"].is_unique  # keyed by Surya frame


def test_figure_is_written_for_the_best_epoch_only(tmp_path):
    callback = run_fit(tmp_path, max_epochs=3)
    saved = pd.read_csv(tmp_path / "val_predictions.csv")

    assert (tmp_path / "val_predictions_best.png").exists()
    # The recorded best epoch must be the one that actually minimized val_loss, and the
    # kept frame must be that epoch's rows -- not the last epoch's.
    assert callback._best_epoch is not None
    best_rows = saved[saved["epoch"] == callback._best_epoch].reset_index(drop=True)
    np.testing.assert_allclose(
        sorted(callback._best_df["pred"]), sorted(best_rows["pred"]), rtol=1e-5
    )


def test_a_rerun_does_not_append_to_the_previous_runs_rows(tmp_path):
    run_fit(tmp_path, max_epochs=2)
    run_fit(tmp_path, max_epochs=2)
    saved = pd.read_csv(tmp_path / "val_predictions.csv")
    assert len(saved) == 2 * 2 * N_PER_CLASS


def test_missing_monitor_key_warns_rather_than_crashing_the_run(tmp_path):
    loader = DataLoader(TinyValDataset(), batch_size=8, drop_last=False)
    callback = ValidationPredictionLogger(output_dir=tmp_path, monitor="not_a_metric")
    trainer = L.Trainer(
        max_epochs=1, accelerator="cpu", devices=1, logger=False, enable_checkpointing=False,
        enable_progress_bar=False, enable_model_summary=False, num_sanity_val_steps=0,
        callbacks=[callback],
    )
    with pytest.warns(UserWarning, match="not_a_metric"):
        trainer.fit(TinyModule(), loader, loader)
    # Predictions are still saved: only the figure needs a monitored metric.
    assert (tmp_path / "val_predictions.csv").exists()


def test_mode_is_validated():
    with pytest.raises(ValueError, match="mode must be"):
        ValidationPredictionLogger(mode="lowest")


# --- the figure itself ------------------------------------------------------------------


def make_df():
    rng = np.random.default_rng(1)
    target = np.concatenate([10 ** rng.normal(-1.7, 0.4, N_PER_CLASS),
                             10 ** rng.normal(0.3, 0.5, N_PER_CLASS)])
    return pd.DataFrame({
        "is_event": [0] * N_PER_CLASS + [1] * N_PER_CLASS,
        "target": target,
        "pred": target * 10 ** rng.normal(0, 0.3, 2 * N_PER_CLASS),
    })


def test_figure_has_two_panels_and_reports_rmse_in_the_image():
    fig = plot_event_predictions(make_df())
    texts = [fig._suptitle.get_text()] + [ax.get_title() for ax in fig.axes]
    assert sum("RMSE" in t for t in texts) == 3  # overall + one per panel
    assert any("Non-event  (n=25)" in t for t in texts)
    assert any("Event  (n=25)" in t for t in texts)


def test_one_pair_of_bars_per_sample():
    # The figure is per-sample, not binned: each panel holds exactly one true bar and one
    # predicted bar for every sample it covers.
    fig = plot_event_predictions(make_df())
    panels = [ax for ax in fig.axes if ax.patches]
    assert len(panels) == 2
    for ax in panels:
        assert len(ax.patches) == 2 * N_PER_CLASS


def test_true_bars_are_sorted_so_the_predicted_series_can_be_read_against_them():
    # Sorting by true value is what makes a flat prediction visible at a glance.
    fig = plot_event_predictions(make_df())
    for ax in (a for a in fig.axes if a.patches):
        true_heights = [p.get_height() for p in ax.patches[:N_PER_CLASS]]
        assert true_heights == sorted(true_heights)


def test_negative_values_stay_visible_on_a_linear_axis():
    # The log10-z-scored target is legitimately negative; a linear axis must cover it
    # rather than clip it away.
    df = make_df()
    df["target"] = np.log10(df["target"])
    df["pred"] = np.log10(df["pred"]) - 0.4
    fig = plot_event_predictions(df, log_y=False)
    lowest = min(df["target"].min(), df["pred"].min())
    for ax in (a for a in fig.axes if a.patches):
        assert ax.get_ylim()[0] <= lowest, "a negative bar falls outside the axis"


def test_negative_predictions_are_reported_not_silently_dropped_on_a_log_axis():
    # With log_y a non-positive prediction cannot be drawn; the count has to surface in
    # the image instead of vanishing.
    df = make_df()
    df.loc[[0, 1, 2], "pred"] = -0.5
    fig = plot_event_predictions(df, log_y=True)
    notes = [t.get_text() for ax in fig.axes for t in ax.texts]
    assert any("3 prediction(s) <= 0" in n for n in notes)
