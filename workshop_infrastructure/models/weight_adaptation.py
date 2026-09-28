"""
Adapting pretrained Surya weights to a differently-shaped fine-tuning model.

The 366M checkpoint was pretrained at one specific resolution (4096x4096) with one
specific number of input frames (2). A downstream app that changes either of those ends
up with two tensors whose shapes no longer match the checkpoint:

    embedding.patch_embed.proj.weight          (embed_dim, in_chans * time_dim, p, p)
    blocks_spectral_gating.*.filter.complex_weight   (grid, grid // 2 + 1, embed_dim, 2)

Together they are ~170M of the 366M pretrained parameters. Dropping them means the run
starts from a random tokenizer and random spectral filters — it is not Surya any more,
and nothing about the loss curve makes that obvious. The two functions here convert the
pretrained tensors instead of discarding them, and ``load_pretrained_weights()`` refuses
to silently drop anything they cannot handle.

Both conversions are exact in the sense that matters:

* ``adapt_patch_embed_weight`` **selects** the pretrained per-frame weights rather than
  averaging or re-initializing them, so a 1-frame model inherits the tokenizer the
  pretrained model applied to its most recent frame.
* ``truncate_spectral_filter`` **truncates in the frequency plane**. The token grid always
  spans the same field of view, so token-grid mode ``m`` is the same *physical* spatial
  frequency (period = field / m) at every resolution. A 64x64 token grid therefore wants
  exactly the |k| <= 32 block of the pretrained 256x256 filter, in rfft2 layout. This is
  a restriction of the learned filter to the frequencies the smaller grid can represent,
  not an interpolation or a guess.
"""

from __future__ import annotations

import torch


def adapt_patch_embed_weight(
    ckpt_weight: torch.Tensor,
    target_shape: torch.Size | tuple[int, ...],
    in_chans: int,
) -> torch.Tensor:
    """Adapt a pretrained patch-embedding conv weight to a different number of frames.

    ``PatchEmbed3D`` flattens ``(B, C, T, H, W)`` to ``(B, C * T, H, W)`` before the
    convolution, so the input-channel axis is ordered ``c * T + t``: all of channel 0's
    frames, then all of channel 1's, and so on. Reducing ``T`` is therefore a strided
    selection, not a reshape — and picking the wrong stride silently scrambles channels
    into each other.

    The frames kept are the **most recent** ones (the tail of the time axis), because
    ``HelioNetCDFDataset`` always places the reference timestep last:
    ``time_delta_input_minutes`` is sorted ascending and ends at offset 0.

    Args:
        ckpt_weight: Pretrained weight, shape ``(embed_dim, in_chans * T_ckpt, p, p)``.
        target_shape: Shape the model wants, ``(embed_dim, in_chans * T_model, p, p)``.
        in_chans: Number of physical input channels (13 for the Surya stack). Shared by
            both shapes — this function changes ``T``, never ``C``.

    Returns:
        A tensor of shape ``target_shape`` holding the selected pretrained weights.

    Raises:
        ValueError: If the shapes differ in any way this selection cannot express
            (different embed_dim or patch size, a channel-count change, a non-integer
            frame count, or a target that asks for more frames than the checkpoint has).
    """
    target_shape = tuple(target_shape)
    if ckpt_weight.shape[0] != target_shape[0] or ckpt_weight.shape[2:] != target_shape[2:]:
        raise ValueError(
            "Patch-embedding weights differ in embed_dim or patch size "
            f"(checkpoint {tuple(ckpt_weight.shape)} vs model {target_shape}); only the "
            "number of input frames can be adapted."
        )

    ckpt_in, model_in = ckpt_weight.shape[1], target_shape[1]
    if ckpt_in % in_chans or model_in % in_chans:
        raise ValueError(
            f"Patch-embedding input channels ({ckpt_in} in the checkpoint, {model_in} in "
            f"the model) must both be multiples of in_chans={in_chans}."
        )

    t_ckpt, t_model = ckpt_in // in_chans, model_in // in_chans
    if t_model > t_ckpt:
        raise ValueError(
            f"The model asks for {t_model} input frames but the checkpoint was pretrained "
            f"with {t_ckpt}. There are no pretrained weights for the extra frames; reduce "
            "model.time_embedding.time_dim, or set model.pretrained_path: null to train "
            "the tokenizer from scratch."
        )

    # Keep the last t_model frames of every channel: index c * t_ckpt + t for
    # t in [t_ckpt - t_model, t_ckpt).
    keep = torch.tensor(
        [c * t_ckpt + t for c in range(in_chans) for t in range(t_ckpt - t_model, t_ckpt)],
        dtype=torch.long,
    )
    return ckpt_weight.index_select(1, keep).clone()


def truncate_spectral_filter(
    ckpt_weight: torch.Tensor,
    target_shape: torch.Size | tuple[int, ...],
) -> torch.Tensor:
    """Restrict a pretrained spectral-gating filter to a smaller token grid.

    ``SpectralGatingNetwork`` holds one learnable complex gain per rfft2 mode of the token
    grid, stored as ``(h, w, embed_dim, 2)`` with ``h = grid`` and ``w = grid // 2 + 1``.

    A token grid of ``N x N`` tokens always covers the whole solar disk, so mode index
    ``m`` means "one cycle per field / m" regardless of ``N``. Halving the image
    resolution halves ``N`` and simply removes the modes above the new Nyquist limit — the
    surviving modes keep their indices. The correct conversion is therefore to take the
    low-|k| block in rfft2 layout:

        dim 0 (length N, wrapped): frequencies [0, 1, ..., N/2, -N/2+1, ..., -1]
            -> keep rows [0 : Nt/2 + 1] and [-(Nt/2 - 1) :]
        dim 1 (length N/2 + 1, one-sided): frequencies [0, 1, ..., N/2]
            -> keep cols [0 : Nt/2 + 1]

    ``norm="ortho"`` scaling cancels between the forward and inverse transforms, and the
    filter is a dimensionless per-mode gain, so no rescaling is needed.

    Args:
        ckpt_weight: Pretrained filter, shape ``(h_c, w_c, embed_dim, 2)``.
        target_shape: Shape the model wants, ``(h_t, w_t, embed_dim, 2)``.

    Returns:
        A tensor of shape ``target_shape`` holding the low-frequency block.

    Raises:
        ValueError: If the target grid is larger than the checkpoint's (there are no
            pretrained gains for the extra high frequencies), or if either shape is not
            a valid rfft2 filter layout.
    """
    target_shape = tuple(target_shape)
    if ckpt_weight.shape[2:] != target_shape[2:]:
        raise ValueError(
            "Spectral filters differ in embed_dim "
            f"(checkpoint {tuple(ckpt_weight.shape)} vs model {target_shape}); only the "
            "token-grid size can be adapted."
        )

    n_ckpt, n_target = ckpt_weight.shape[0], target_shape[0]
    for name, n, w in (("checkpoint", n_ckpt, ckpt_weight.shape[1]), ("model", n_target, target_shape[1])):
        if w != n // 2 + 1 or n % 2:
            raise ValueError(
                f"The {name} spectral filter has shape {tuple(ckpt_weight.shape if name == 'checkpoint' else target_shape)}, "
                f"which is not an rfft2 layout for an even NxN token grid (expected "
                f"(N, N // 2 + 1, embed_dim, 2))."
            )

    if n_target > n_ckpt:
        raise ValueError(
            f"The model's token grid is {n_target}x{n_target} but the checkpoint was "
            f"pretrained on {n_ckpt}x{n_ckpt}. The extra high-frequency modes have no "
            "pretrained values. Reduce model.img_size (or increase data.pooling), or set "
            "model.pretrained_path: null to train the spectral filters from scratch."
        )
    if n_target == n_ckpt:
        return ckpt_weight.clone()

    half = n_target // 2
    # Positive frequencies 0..half, then the wrapped negative ones -(half-1)..-1.
    rows = torch.cat([
        torch.arange(0, half + 1),
        torch.arange(n_ckpt - (half - 1), n_ckpt),
    ])
    out = ckpt_weight.index_select(0, rows)[:, : half + 1].clone()
    assert tuple(out.shape) == target_shape, (tuple(out.shape), target_shape)
    return out
