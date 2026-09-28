#!/usr/bin/env python3
"""
Measure what restricting the tokenizer to a channel subset does to the token distribution.

This is the diagnostic behind ``model.trainable_backbone_modules: [embedding.patch_embed]``
in the config. The argument for training the tokenizer is that the sliced tokenizer is not
the function the backbone was trained to consume; this script is how that stops being an
argument and becomes a number.

What it compares, on real normalized data:

  * the token standard deviation from the full 13-channel pretrained tokenizer, and
  * the same from the slice of it this app uses (``data.input_channels`` only),

both against ``pos_embed.std()``. ``LinearEmbedding`` is ``patch_embed(x) + pos_embed``
with no normalization in between, and the blocks are pre-norm -- LayerNorm only ever sees a
branch input, never the residual stream -- so the ratio between those two terms is a real
property of what the backbone receives, not something a later layer normalizes away.
``pos_embed`` is a fixed-amplitude Fourier buffer, so it does not shrink with the content.

Usage (from the repo root):

    python -m downstream_apps.Imagetranslation.tools.measure_token_scale
    python -m downstream_apps.Imagetranslation.tools.measure_token_scale --index my.csv -n 10

The default index is the app's validation index, which points at S3 -- pass ``--index`` with
a CSV of local paths to avoid the download, and ``--s3-cache-dir`` otherwise.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from downstream_apps.Imagetranslation.configs import load_euv2mag_config
from workshop_infrastructure.datasets.helio import HelioNetCDFDataset
from workshop_infrastructure.models.embedding import LinearEmbedding
from workshop_infrastructure.utils import build_scalers

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "config_script.yaml"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--index", default=None,
                   help="CSV index to sample from (default: data.valid_data_path).")
    p.add_argument("-n", "--num-samples", type=int, default=3)
    p.add_argument("--s3-cache-dir", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_euv2mag_config(args.config)
    if args.s3_cache_dir:
        cfg.data.s3_cache_dir = args.s3_cache_dir

    order = cfg.data.pretrained_channel_order
    subset = [order.index(c) for c in cfg.data.input_channels]

    # All 13 channels, so the two tokenizers see the same data. load_forecast_frames=False:
    # this measures the input side only, and target frames would just be downloads.
    dataset = HelioNetCDFDataset(
        index_path=args.index or cfg.data.valid_data_path,
        time_delta_input_minutes=cfg.data.time_delta_input_minutes,
        time_delta_target_minutes=cfg.data.time_delta_target_minutes,
        n_input_timestamps=cfg.model.time_embedding.time_dim,
        rollout_steps=0,
        scalers=build_scalers(info=cfg.data.scalers_path),
        channels=order,
        phase="val",
        pooling=cfg.data.pooling,
        load_forecast_frames=False,
        s3_mode=cfg.data.s3_mode,
        s3_storage_options={"anon": cfg.data.s3_anon},
        s3_cache_dir=cfg.data.s3_cache_dir,
    )

    checkpoint = torch.load(cfg.model.pretrained_path, weights_only=True,
                            map_location="cpu", mmap=True)
    weight = checkpoint["embedding.patch_embed.proj.weight"]
    bias = checkpoint["embedding.patch_embed.proj.bias"]
    # (embed_dim, C * T, p, p) indexed c * T + t; keep the trailing frame, as the model does.
    frames = weight.shape[1] // len(order)
    full = torch.stack([weight[:, c * frames + frames - 1] for c in range(len(order))], dim=1)
    sliced = full[:, subset]

    pos_std = float(
        LinearEmbedding(
            img_size=cfg.model.img_size, patch_size=cfg.model.patch_size,
            in_chans=cfg.model.in_channels, embed_dim=cfg.model.embed_dim, time_dim=1,
        ).pos_embed.std()
    )

    stride = cfg.model.patch_size
    n = min(args.num_samples, len(dataset))
    print(f"{'+'.join(cfg.data.input_channels)} out of {len(order)} pretrained channels, "
          f"{cfg.model.img_size}px (data.pooling={cfg.data.pooling})\n")
    print(f"{'sample':<8}{'std all':>10}{'std subset':>12}{'ratio':>8}"
          f"{'all:pos':>10}{'subset:pos':>12}")
    ratios = []
    for i in range(n):
        ts = torch.from_numpy(np.asarray(dataset[i]["ts"])).unsqueeze(0).float()
        s_all = float(F.conv2d(ts[:, :, -1], full, bias, stride=stride).std())
        s_sub = float(F.conv2d(ts[:, subset, -1], sliced, bias, stride=stride).std())
        ratios.append(s_sub / s_all)
        print(f"{i:<8}{s_all:>10.3f}{s_sub:>12.3f}{s_sub / s_all:>8.3f}"
              f"{s_all / pos_std:>10.2f}{s_sub / pos_std:>12.2f}")

    mean = float(np.mean(ratios))
    print(f"\nn={n}  ratio mean {mean:.3f}  min {min(ratios):.3f}  max {max(ratios):.3f}")
    print(f"pos_embed std {pos_std:.3f}")
    print(f"Position's share of each token grows by {1 / mean:.2f}x when the tokenizer is "
          f"restricted.")
    print(
        "\nThat shift is what the tokenizer has to absorb, and only the tokenizer can: it is\n"
        "added to pos_embed before any normalization, and a LoRA adapter acts after\n"
        "tokenization and cannot change a per-channel linear map. Hence\n"
        "model.trainable_backbone_modules: [embedding.patch_embed].\n"
        f"\nFor reference, an independence argument -- every channel z-scored and "
        f"uncorrelated --\npredicts sqrt({len(subset)}/{len(order)}) = "
        f"{(len(subset) / len(order)) ** 0.5:.3f}. The measured value differs because the\n"
        "channels are correlated and these are not average channels."
    )


if __name__ == "__main__":
    main()
