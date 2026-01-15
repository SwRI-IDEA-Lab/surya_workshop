#!/usr/bin/env python3
"""
test_eve13p5_checkpoint_plot.py

Test-only script:
- Loads HelioSpectformer1D model + FlareLightningModule wrapper
- Loads weights from a Lightning checkpoint (.ckpt)
- Runs inference on a test CSV via FlareDSDataset
- Computes MAPE + PCC in physical units
- Plots GT vs Predicted (parity plot)
k
Example:
python test_eve13p5_checkpoint_plot.py \
  --config ./configs/config.yaml \
  --test_csv ./data/caiik_2011_2013_EVE_13.5_sample_75_aws_testing.csv \
  --checkpoint /home/haodijiang/checkpoints/EVE_13p5-epoch=00-val_loss=1.1171.ckpt \
  --batch_size 2 \
  --num_workers 2
"""
# EVE_model_local.ckpt
'''
python test_checkpoint_eval.py \
  --config ./configs/config.yaml \
  --test_csv ./data/caiik_2011_2013_EVE_13.5_sample_75_aws_testing.csv \
  --checkpoint /home/haodijiang/checkpoints/EVE_model_local.ckpt \
  --batch_size 2 \
  --num_workers 2 \
  --device cpu
'''

import argparse
import os
import sys

import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from torch.utils.data import DataLoader


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True, help="Path to config.yaml used in training.")
    p.add_argument("--test_csv", type=str, required=True, help="Test CSV file (your EVE 13.5 test index).")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to Lightning .ckpt file.")
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--device", type=str, default="cuda", help="cuda or cpu")

    # IMPORTANT:
    # These MUST match the constants used during training normalization.
    # If you do not pass them, defaults are set to the values you previously shared.
    p.add_argument("--eve_log10_min", type=float, default=-5.078961027729049)
    p.add_argument("--eve_log10_scale", type=float, default=0.12238584732591484)

    # repo paths (same idea as notebook)
    p.add_argument("--base_path", type=str, default="../../", help="Path that contains workshop_infrastructure/")
    p.add_argument("--surya_path", type=str, default="../../Surya", help="Path that contains surya/")

    p.add_argument("--save_plot", type=str, default="gt_vs_pred_eve13p5.png")
    p.add_argument("--save_npz", type=str, default="pred_gt_eve13p5_phys.npz")
    return p.parse_args()


def mape_percent(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-12) -> float:
    gt_safe = np.clip(np.abs(gt), eps, None)
    return float(np.mean(np.abs(pred - gt) / gt_safe) * 100.0)


def main() -> None:
    args = parse_args()

    # ---------------------------------------------------------------------
    # Paths (match notebook behavior)
    # ---------------------------------------------------------------------
    sys.path.append(args.base_path)
    sys.path.append(args.surya_path)

    # Imports that rely on your repo structure
    from surya.utils.data import build_scalers
    from workshop_infrastructure.models.finetune_models import HelioSpectformer1D
    from downstream_apps.haodi.datasets.template_dataset_haodi import FlareDSDataset
    from downstream_apps.haodi.metrics.template_metrics import FlareMetrics
    from downstream_apps.haodi.lightning_modules.pl_simple_baseline import FlareLightningModule

    # ---------------------------------------------------------------------
    # Load config
    # ---------------------------------------------------------------------
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # scalers (Surya stack normalization)
    from pathlib import Path

    config_path = Path(args.config).resolve()
    config_dir = config_path.parent

    # Try common locations for a "scalers" spec inside the config
    scalers_spec = None
    if isinstance(config, dict):
        if "scalers" in config:
            scalers_spec = config["scalers"]
        elif "data" in config and isinstance(config["data"], dict) and "scalers" in config["data"]:
            scalers_spec = config["data"]["scalers"]

    # If scalers_spec is a path string, load that yaml and inject it back
    if isinstance(scalers_spec, str):
        scalers_path = (config_dir / scalers_spec).resolve() if not Path(scalers_spec).is_absolute() else Path(scalers_spec)
        with open(scalers_path, "r") as f:
            scalers_cfg = yaml.safe_load(f)
        # build_scalers expects the scalers entries to live in config["scalers"] (most common),
        # so put it there.
        config["scalers"] = scalers_cfg

    # If scalers_spec is a list of strings, load each and concatenate into one list
    elif isinstance(scalers_spec, list) and all(isinstance(x, str) for x in scalers_spec):
        merged = []
        for item in scalers_spec:
            p = (config_dir / item).resolve() if not Path(item).is_absolute() else Path(item)
            with open(p, "r") as f:
                part = yaml.safe_load(f)
            if isinstance(part, list):
                merged.extend(part)
            else:
                merged.append(part)
        config["scalers"] = merged

    # Now this should work
    scalers = build_scalers(config)
    # ---------------------------------------------------------------------
    # Dataset + Loader (TEST ONLY)
    # ---------------------------------------------------------------------
    test_dataset = FlareDSDataset(
        index_path=config["data"]["train_data_path"],  # Surya index path (same key used before)
        time_delta_input_minutes=config["data"]["time_delta_input_minutes"],
        time_delta_target_minutes=config["data"]["time_delta_target_minutes"],
        n_input_timestamps=config["model"]["time_embedding"]["time_dim"],
        rollout_steps=config["rollout_steps"],
        channels=config["data"]["channels"],
        drop_hmi_probability=0,
        use_latitude_in_learned_flow=config["use_latitude_in_learned_flow"],
        scalers=scalers,
        phase="test",
        s3_use_simplecache=False,
        s3_cache_dir="/tmp/helio_s3_cache",
        return_surya_stack=True,
        max_number_of_samples=None,
        ds_flare_index_path=args.test_csv,
        ds_time_column="timestep",
        ds_time_tolerance="6h",
        ds_match_direction="nearest",
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(args.device == "cuda"),
    )

    # ---------------------------------------------------------------------
    # Build the SAME model architecture
    # ---------------------------------------------------------------------
    model = HelioSpectformer1D(
        img_size=config["model"]["img_size"],
        patch_size=config["model"]["patch_size"],
        in_chans=config["model"]["in_channels"],
        embed_dim=config["model"]["embed_dim"],
        time_embedding=config["model"]["time_embedding"],
        depth=config["model"]["depth"],
        num_heads=config["model"]["num_heads"],
        mlp_ratio=config["model"]["mlp_ratio"],
        drop_rate=config["model"]["drop_rate"],
        dtype=config["dtype"],
        window_size=config["model"]["window_size"],
        dp_rank=config["model"]["dp_rank"],
        learned_flow=config["model"]["learned_flow"],
        use_latitude_in_learned_flow=config["use_latitude_in_learned_flow"],
        init_weights=config["model"]["init_weights"],
        checkpoint_layers=config["model"]["checkpoint_layers"],
        n_spectral_blocks=config["model"]["spectral_blocks"],
        rpe=config["model"]["rpe"],
        ensemble=config["model"]["ensemble"],
        finetune=config["model"]["finetune"],
        nglo=config["model"]["nglo"],
        dropout=config["model"]["dropout"],
        num_penultimate_transformer_layers=0,
        num_penultimate_heads=0,
        num_outputs=1,
        config=config,
    )

    # Metrics are required by FlareLightningModule __init__, but not used in this script.
    train_loss_metrics = FlareMetrics("train_loss")
    train_eval_metrics = FlareMetrics("train_metrics")
    val_eval_metrics = FlareMetrics("val_metrics")
    metrics = {
        "train_loss": train_loss_metrics,
        "train_metrics": train_eval_metrics,
        "val_metrics": val_eval_metrics,
    }

    lit_model = FlareLightningModule(
        model=model,
        metrics=metrics,
        lr=1e-4,  # not used for inference
        batch_size=args.batch_size,
        eve_log10_min=args.eve_log10_min,
        eve_log10_scale=args.eve_log10_scale,
    )

    # ---------------------------------------------------------------------
    # Load checkpoint weights (safe path: load state_dict)
    # ---------------------------------------------------------------------
    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)

    if "state_dict" not in ckpt:
        raise ValueError("Checkpoint does not contain 'state_dict'. Is this a Lightning .ckpt file?")

    missing, unexpected = lit_model.load_state_dict(ckpt["state_dict"], strict=False)
    if missing:
        print("WARNING: Missing keys when loading state_dict:")
        for k in missing:
            print("  ", k)
    if unexpected:
        print("WARNING: Unexpected keys when loading state_dict:")
        for k in unexpected:
            print("  ", k)

    lit_model.to(device)
    lit_model.eval()
    lit_model.freeze()

    # ---------------------------------------------------------------------
    # Inference loop: collect physical preds and GT
    # Dataset provides eve_13p5_raw (physical target) :contentReference[oaicite:2]{index=2}
    # Inverse transform implemented in LightningModule :contentReference[oaicite:3]{index=3}
    # ---------------------------------------------------------------------
    all_pred_phys = []
    all_gt_phys = []

    with torch.no_grad():
        for batch in test_loader:
            # move tensors to device
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

            out = lit_model(batch)               # normalized prediction, shape ~ (B,1)
            pred_norm = out.squeeze(-1)
            pred_phys = lit_model._inv_norm_to_phys(pred_norm)

            gt_phys = batch["eve_13p5_raw"].float()
            if gt_phys.ndim > 1:
                gt_phys = gt_phys.squeeze(-1)

            all_pred_phys.append(pred_phys.detach().cpu())
            all_gt_phys.append(gt_phys.detach().cpu())

    preds = torch.cat(all_pred_phys).numpy()
    gt = torch.cat(all_gt_phys).numpy()

    # ---------------------------------------------------------------------
    # Metrics: MAPE + PCC
    # ---------------------------------------------------------------------
    mape = mape_percent(preds, gt)
    pcc, _ = pearsonr(preds, gt)

    print(f"Test samples: {len(gt)}")
    print(f"Test MAPE (physical units): {mape:.2f}%")
    print(f"Test PCC  (physical units): {pcc:.4f}")

    # Save arrays for later analysis
    np.savez(args.save_npz, preds=preds, gt=gt)
    print(f"Saved preds/gt to: {args.save_npz}")

    # ---------------------------------------------------------------------
    # Plot: GT vs Predicted (parity)
    # ---------------------------------------------------------------------
    plt.figure(figsize=(6, 6))
    plt.scatter(gt, preds, s=12, alpha=0.6, edgecolors="none")

    min_val = float(min(gt.min(), preds.min()))
    max_val = float(max(gt.max(), preds.max()))
    plt.plot([min_val, max_val], [min_val, max_val], "k--", linewidth=1)

    plt.xlabel("Ground Truth EVE 13.5 nm")
    plt.ylabel("Predicted EVE 13.5 nm")
    plt.title(f"GT vs Predicted (MAPE={mape:.2f}%, PCC={pcc:.3f})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.save_plot, dpi=300)
    print(f"Saved plot to: {args.save_plot}")
    plt.show()


if __name__ == "__main__":
    main()
