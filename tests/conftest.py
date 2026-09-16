"""Shared tiny-model helpers for the config/model test suite.

Small enough to be fast, large enough to exercise both backbone block types.
Spectral gating derives its grid from img_size // patch_size, so these must stay consistent.
"""

import torch

from workshop_infrastructure.models.finetune_models import HelioSpectformer1D

IMG_SIZE = 64
PATCH_SIZE = 16
IN_CHANS = 13
EMBED_DIM = 32
DEPTH = 3
N_SPECTRAL_BLOCKS = 1  # -> 1 spectral block, 2 attention blocks
N_ATTENTION_BLOCKS = DEPTH - N_SPECTRAL_BLOCKS


def make_model(pooling="class_token", penultimate_linear_layer=True):
    return HelioSpectformer1D(
        img_size=IMG_SIZE,
        patch_size=PATCH_SIZE,
        in_chans=IN_CHANS,
        embed_dim=EMBED_DIM,
        time_embedding={"type": "linear", "time_dim": 1},
        depth=DEPTH,
        n_spectral_blocks=N_SPECTRAL_BLOCKS,
        num_heads=2,
        mlp_ratio=4,
        drop_rate=0.0,
        window_size=2,
        dp_rank=2,
        dtype=torch.float32,
        pooling=pooling,
        penultimate_linear_layer=penultimate_linear_layer,
        num_outputs=1,
    )


def make_batch(batch_size=2):
    return {
        "ts": torch.randn(batch_size, IN_CHANS, 1, IMG_SIZE, IMG_SIZE),
        "time_delta_input": torch.zeros(batch_size, 1),
    }
