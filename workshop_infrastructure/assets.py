"""
Downloading the Surya assets (normalization scalers and pretrained weights).

Both files are fetched from HuggingFace on demand. This module is the single
implementation: the training script, the notebooks and the two ``download_*.sh``
wrappers all route through ``ensure_assets``.

Typical use, from a config::

    from workshop_infrastructure.assets import ensure_assets
    ensure_assets(cfg)                        # scalers + weights
    ensure_assets(cfg, which=["scalers"])     # scalers only (no backbone needed)

Or standalone, with no config::

    python -m workshop_infrastructure.assets --scalers --dest downstream_apps/template/assets
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


@dataclass(frozen=True)
class AssetSpec:
    """Where one asset lives on HuggingFace."""
    repo_id: str
    repo_type: str
    filename: str


# The two assets every downstream app needs. Keys are the names accepted by `which`.
SURYA_ASSETS: dict[str, AssetSpec] = {
    "scalers": AssetSpec("nasa-ibm-ai4science/core-sdo", "dataset", "scalers.yaml"),
    "weights": AssetSpec("nasa-ibm-ai4science/Surya-1.0", "model", "surya.366m.v1.pt"),
}


def _hf_hub_download():
    """Import hf_hub_download lazily, with an actionable message when it is missing."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise RuntimeError(
            "huggingface_hub is required to download assets. "
            "Install it with: pip install huggingface_hub"
        ) from e
    return hf_hub_download


def download_asset(spec: AssetSpec, local_path: str | Path) -> Path:
    """Fetch one asset to ``local_path`` unless it is already there.

    Returns the local path, whether or not anything was downloaded.
    """
    local_path = Path(local_path)
    if local_path.exists():
        return local_path

    print(f"[assets] {local_path.name} not found — downloading from {spec.repo_id} ...")
    local_path.parent.mkdir(parents=True, exist_ok=True)
    _hf_hub_download()(
        repo_id=spec.repo_id,
        repo_type=spec.repo_type,
        filename=spec.filename,
        local_dir=str(local_path.parent),
    )
    print(f"[assets] Saved to {local_path}")
    return local_path


def ensure_assets(cfg, which: Iterable[str] = ("scalers", "weights")) -> None:
    """Download whichever of the configured assets are missing.

    Only fetches what is absent, so running the notebooks first and the script second
    (or re-running either) costs nothing.

    Args:
        cfg: A ``TrainingConfig``. ``cfg.data.scalers_path`` and
            ``cfg.model.pretrained_path`` say where the files belong; both are already
            resolved to absolute paths by ``load_config()``.
        which: Which assets to consider. Pass ``["scalers"]`` for workflows that never
            touch the backbone, such as the dataset and linear-baseline notebooks.
    """
    destinations = {
        "scalers": getattr(cfg.data, "scalers_path", None),
        "weights": getattr(cfg.model, "pretrained_path", None),
    }

    unknown = sorted(set(which) - set(SURYA_ASSETS))
    if unknown:
        raise ValueError(
            f"Unknown asset name(s): {', '.join(unknown)}. "
            f"Valid names: {', '.join(sorted(SURYA_ASSETS))}."
        )

    for name in which:
        local_path = destinations[name]
        if not local_path:
            # e.g. pretrained_path is null for a run that trains from scratch.
            continue
        download_asset(SURYA_ASSETS[name], local_path)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI used by the download_*.sh wrappers, for use without a config."""
    parser = argparse.ArgumentParser(
        description="Download Surya assets (scalers and/or pretrained weights) from HuggingFace."
    )
    parser.add_argument("--scalers", action="store_true", help="Download scalers.yaml.")
    parser.add_argument("--weights", action="store_true", help="Download the pretrained backbone.")
    parser.add_argument("--dest", required=True, help="Directory to place the files in.")
    args = parser.parse_args(argv)

    selected = [n for n, on in (("scalers", args.scalers), ("weights", args.weights)) if on]
    if not selected:
        parser.error("Nothing to do: pass --scalers and/or --weights.")

    dest = Path(args.dest)
    for name in selected:
        spec = SURYA_ASSETS[name]
        download_asset(spec, dest / spec.filename)
    return 0


if __name__ == "__main__":
    sys.exit(main())
