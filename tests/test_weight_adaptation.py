"""Tests for adapting pretrained Surya weights to a differently-shaped model.

These cover the two conversions that stand between "fine-tuning Surya" and "fine-tuning a
partly-random model that looks exactly the same in the logs": the patch-embedding
selection and the spectral-filter truncation. They also pin the behaviour that surfaces
the problem — ``load_pretrained_weights`` raising on an unconvertible tensor.
"""

import pytest
import torch

from workshop_infrastructure.models.weight_adaptation import (
    adapt_patch_embed_weight,
    truncate_spectral_filter,
)
from workshop_infrastructure.utils import load_pretrained_weights

from conftest import EMBED_DIM, IN_CHANS, PATCH_SIZE, make_batch, make_model


# ---------------------------------------------------------------------------
# Patch embedding: in_chans * time_dim -> in_chans * fewer_frames
# ---------------------------------------------------------------------------

def test_patch_embed_keeps_the_most_recent_frame_of_every_channel():
    # PatchEmbed3D flattens (B, C, T, H, W) -> (B, C*T, H, W), so index = c * T + t.
    # Going from T=2 to T=1 must keep t=1 (the reference timestep, which the dataset
    # always places last), i.e. the odd indices.
    ckpt = torch.arange(1 * 26 * 1 * 1, dtype=torch.float32).reshape(1, 26, 1, 1)
    out = adapt_patch_embed_weight(ckpt, (1, 13, 1, 1), in_chans=13)
    assert out.shape == (1, 13, 1, 1)
    assert out[0, :, 0, 0].tolist() == [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25]


def test_patch_embed_is_a_noop_when_the_frame_count_already_matches():
    ckpt = torch.randn(4, 26, 2, 2)
    out = adapt_patch_embed_weight(ckpt, ckpt.shape, in_chans=13)
    assert torch.equal(out, ckpt)


def test_patch_embed_refuses_to_invent_frames_it_has_no_weights_for():
    ckpt = torch.randn(4, 26, 2, 2)  # 2 frames
    with pytest.raises(ValueError, match="no pretrained weights|pretrained with"):
        adapt_patch_embed_weight(ckpt, (4, 39, 2, 2), in_chans=13)  # asks for 3


def test_patch_embed_refuses_an_embed_dim_change():
    ckpt = torch.randn(4, 26, 2, 2)
    with pytest.raises(ValueError, match="embed_dim or patch size"):
        adapt_patch_embed_weight(ckpt, (8, 13, 2, 2), in_chans=13)


# ---------------------------------------------------------------------------
# Spectral filter: NxN token grid -> smaller token grid
# ---------------------------------------------------------------------------

def test_spectral_truncation_selects_the_low_frequency_block():
    # Build a filter that is 1.0 exactly on the modes a 16x16 grid can represent and 0
    # elsewhere; truncating to 16 must therefore return all ones.
    n_ckpt, n_target, dim = 64, 16, 3
    half = n_target // 2
    f = torch.zeros(n_ckpt, n_ckpt // 2 + 1, dim, 2)
    rows = list(range(0, half + 1)) + list(range(n_ckpt - (half - 1), n_ckpt))
    for r in rows:
        f[r, : half + 1] = 1.0

    out = truncate_spectral_filter(f, (n_target, n_target // 2 + 1, dim, 2))
    assert out.shape == (16, 9, dim, 2)
    assert (out == 1.0).all()


def test_spectral_truncation_preserves_the_filter_applied_to_shared_frequencies():
    """The physical check: mode m means the same spatial frequency at either grid size.

    A signal band-limited below the small grid's Nyquist limit must come out of the
    truncated filter with the same spectrum it gets from the full one. If the row
    selection were wrong (e.g. naive ``[:n]`` slicing, which drops the negative
    frequencies) this is the assertion that fails.
    """
    torch.manual_seed(0)
    n_ckpt, n_target = 64, 16
    f = torch.randn(n_ckpt, n_ckpt // 2 + 1, 1, 2) * 0.02
    small_f = truncate_spectral_filter(f, (n_target, n_target // 2 + 1, 1, 2))

    # Band-limit to |k| <= 5, well inside what the 16x16 grid represents.
    band, half = 6, n_target // 2
    big_spec = torch.zeros(n_ckpt, n_ckpt // 2 + 1, dtype=torch.complex64)
    small_spec = torch.zeros(n_target, n_target // 2 + 1, dtype=torch.complex64)
    for k in list(range(0, band)) + list(range(-band + 1, 0)):
        values = torch.randn(band, dtype=torch.complex64)
        big_spec[k % n_ckpt, :band] = values
        small_spec[k % n_target, :band] = values

    def apply(spec, n, filt):
        x = torch.fft.irfft2(spec, s=(n, n), dim=(0, 1), norm="ortho")
        X = torch.fft.rfft2(x, dim=(0, 1), norm="ortho")
        return X * torch.view_as_complex(filt)[:, :, 0]

    big_out = apply(big_spec, n_ckpt, f)
    small_out = apply(small_spec, n_target, small_f)

    rows = torch.cat([torch.arange(0, half + 1), torch.arange(n_ckpt - (half - 1), n_ckpt)])
    shared = big_out.index_select(0, rows)[:, : half + 1]
    assert torch.allclose(shared, small_out, atol=1e-5)


def test_spectral_truncation_refuses_to_upsample():
    f = torch.randn(16, 9, 2, 2)
    with pytest.raises(ValueError, match="no pretrained values|token grid is"):
        truncate_spectral_filter(f, (64, 33, 2, 2))


# ---------------------------------------------------------------------------
# load_pretrained_weights
# ---------------------------------------------------------------------------

def _checkpoint_for(model, **shape_overrides):
    """A flat (un-prefixed) checkpoint matching ``model``, with optional shape changes."""
    state = {
        k[len("backbone."):] if k.startswith("backbone.") else k: torch.randn_like(v).float()
        for k, v in model.state_dict().items()
        if not k.startswith("head_")
    }
    state.update(shape_overrides)
    return state


def test_load_raises_rather_than_silently_dropping_an_unconvertible_tensor(tmp_path):
    model = make_model(pooling="global_average")
    ckpt = _checkpoint_for(model)
    # An embed_dim change has no defined conversion; it used to be dropped in silence.
    key = next(k for k in ckpt if k.endswith("blocks_attention.0.mlp.fc1.weight"))
    ckpt[key] = torch.randn(ckpt[key].shape[0] * 2, ckpt[key].shape[1])
    path = tmp_path / "ckpt.pt"
    torch.save(ckpt, path)

    with pytest.raises(ValueError, match="no defined conversion"):
        load_pretrained_weights(model, str(path))


def test_strict_shapes_false_restores_the_old_drop_and_continue(tmp_path):
    model = make_model(pooling="global_average")
    ckpt = _checkpoint_for(model)
    key = next(k for k in ckpt if k.endswith("blocks_attention.0.mlp.fc1.weight"))
    ckpt[key] = torch.randn(ckpt[key].shape[0] * 2, ckpt[key].shape[1])
    path = tmp_path / "ckpt.pt"
    torch.save(ckpt, path)

    load_pretrained_weights(model, str(path), strict_shapes=False)  # must not raise


def test_load_adapts_a_two_frame_patch_embedding_into_a_one_frame_model(tmp_path):
    """The template's real case: the checkpoint was pretrained with 2 frames, the config
    asks for 1, and the tokenizer used to be left at its random initialization."""
    model = make_model(pooling="global_average")  # time_dim=1 -> 13 input channels
    ckpt = _checkpoint_for(model)
    two_frame = torch.randn(EMBED_DIM, IN_CHANS * 2, PATCH_SIZE, PATCH_SIZE)
    ckpt["embedding.patch_embed.proj.weight"] = two_frame
    path = tmp_path / "ckpt.pt"
    torch.save(ckpt, path)

    load_pretrained_weights(model, str(path))

    loaded = model.backbone.embedding.patch_embed.proj.weight
    expected = two_frame[:, [2 * c + 1 for c in range(IN_CHANS)]]
    assert torch.allclose(loaded, expected)
    # And the model still runs.
    model(make_batch(batch_size=2))
