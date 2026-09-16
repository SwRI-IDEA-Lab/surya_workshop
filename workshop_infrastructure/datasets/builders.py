"""
Dataset and DataLoader construction shared by every Surya downstream app.

``HelioNetCDFDataset`` takes ~20 constructor arguments, all of which come straight from
the ``data:`` and ``training:`` sections of the config. Spelling that block out at every
entry point means the same 20 lines get copied into every notebook and every training
script, and they drift: one copy forgets ``phase="val"``, another hardcodes a flag the
YAML was supposed to control.

These builders own that block once. A downstream app supplies its dataset subclass and
whatever task-specific keyword arguments that subclass adds::

    train_loader, val_loader = build_helio_dataloaders(
        cfg,
        FlareDSDataset,
        ds_flare_index_path=cfg.data.flare_index_path,
        label_transform=my_label_transform,
    )
"""

from __future__ import annotations

import random
from functools import partial
from typing import Any, Tuple, Type

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from workshop_infrastructure.utils import build_scalers


def _seed_worker(worker_id: int, base_seed: int) -> None:
    """Seed every RNG a DataLoader worker might draw from.

    Seeding torch alone is not enough: ``RandomChannelMaskerTransform`` in
    ``workshop_infrastructure/datasets/helio.py`` draws from Python's ``random``
    (``random.sample`` / ``random.random``), and downstream datasets commonly use NumPy.

    Supplying this deliberately displaces Lightning's auto-injected
    ``pl_worker_init_function``, which derives worker seeds from ``torch.initial_seed()``
    — i.e. from the ambient global RNG state, which is exactly the coupling we are
    removing here.
    """
    seed = (base_seed + worker_id) % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _base_dataset_kwargs(cfg, scalers) -> dict:
    """Map a TrainingConfig onto the HelioNetCDFDataset constructor arguments.

    This is the single place where config field names meet dataset parameter names.
    """
    return dict(
        # Temporal sampling
        time_delta_input_minutes=cfg.data.time_delta_input_minutes,
        time_delta_target_minutes=cfg.data.time_delta_target_minutes,
        n_input_timestamps=cfg.model.time_embedding.time_dim,
        rollout_steps=cfg.rollout_steps,
        # Channels and normalization
        channels=cfg.data.channels,
        scalers=scalers,
        # Augmentation
        drop_hmi_probability=cfg.drop_hmi_probability,
        use_latitude_in_learned_flow=cfg.use_latitude_in_learned_flow,
        # Storage: local root, and how s3:// paths in the index are read
        sdo_data_root_path=cfg.data.sdo_data_root_path,
        s3_mode=cfg.data.s3_mode,
        s3_storage_options={"anon": cfg.data.s3_anon},
        s3_cache_dir=cfg.data.s3_cache_dir,
        s3_boto3_max_concurrency=cfg.data.s3_boto3_max_concurrency,
        s3_boto3_part_size_mb=cfg.data.s3_boto3_part_size_mb,
    )


def build_helio_datasets(
    cfg,
    dataset_cls: Type[Dataset],
    scalers: Any = None,
    **task_kwargs,
) -> Tuple[Dataset, Dataset]:
    """Build the train and validation datasets described by ``cfg``.

    Args:
        cfg: A ``TrainingConfig`` from ``load_config()``.
        dataset_cls: ``HelioNetCDFDataset`` or a subclass of it.
        scalers: Normalization statistics. Built from ``cfg.data.scalers_path`` if omitted;
            pass an existing dict to avoid re-reading the YAML.
        **task_kwargs: Extra keyword arguments forwarded to ``dataset_cls`` — the
            task-specific parameters your subclass adds.

    Returns:
        ``(train_dataset, val_dataset)``. They differ only in the index they read and in
        ``phase``: the validation set uses ``phase="val"``, which disables the random
        channel masking and vertical flips that are applied during training.
    """
    if scalers is None:
        scalers = build_scalers(info=cfg.data.scalers_path)

    common = {**_base_dataset_kwargs(cfg, scalers), **task_kwargs}

    train_dataset = dataset_cls(index_path=cfg.data.train_data_path, phase="train", **common)
    val_dataset = dataset_cls(index_path=cfg.data.valid_data_path, phase="val", **common)
    return train_dataset, val_dataset


def build_helio_dataloaders(
    cfg,
    dataset_cls: Type[Dataset],
    scalers: Any = None,
    num_workers: int | None = None,
    seed: int | None = None,
    **task_kwargs,
) -> Tuple[DataLoader, DataLoader]:
    """Build the train and validation DataLoaders described by ``cfg``.

    Args:
        cfg: A ``TrainingConfig`` from ``load_config()``.
        dataset_cls: ``HelioNetCDFDataset`` or a subclass of it.
        scalers: Normalization statistics; built from the config if omitted.
        num_workers: Override for ``cfg.num_workers`` (useful in notebooks, where fewer
            workers start faster).
        seed: Seed for the shuffle order and the per-worker RNGs. Defaults to ``cfg.seed``.
            Pinning it explicitly is what makes the epoch order depend only on the config:
            a bare ``shuffle=True`` seeds its sampler from whatever the global torch RNG
            state happens to be when the iterator is created, so anything that consumes
            RNG earlier in the program silently reshuffles the data.
        **task_kwargs: Extra keyword arguments forwarded to ``dataset_cls``.

    Returns:
        ``(train_loader, val_loader)``. Only the training loader shuffles.
    """
    train_dataset, val_dataset = build_helio_datasets(
        cfg, dataset_cls, scalers=scalers, **task_kwargs
    )

    workers = cfg.num_workers if num_workers is None else num_workers
    base_seed = cfg.seed if seed is None else seed

    loader_kwargs = dict(
        batch_size=cfg.batch_size,
        num_workers=workers,
        pin_memory=True,
        drop_last=True,
    )
    if workers > 0:
        # "spawn": the dataset holds an s3fs/boto3 handle that does not survive fork.
        # Both of these are rejected outright when num_workers == 0.
        loader_kwargs["multiprocessing_context"] = "spawn"
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["worker_init_fn"] = partial(_seed_worker, base_seed=base_seed)

    # An explicit generator makes the shuffle a function of the seed alone, rather than
    # of the global RNG state at the moment the iterator happens to be created.
    shuffle_generator = torch.Generator()
    shuffle_generator.manual_seed(base_seed)

    train_loader = DataLoader(
        train_dataset, shuffle=True, generator=shuffle_generator, **loader_kwargs
    )
    # No generator for validation: it is not shuffled, so there is nothing to seed.
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    return train_loader, val_loader
