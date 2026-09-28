"""Kept as an import path only.

``FlareFinetuneLightningModule`` now lives next to ``FlareLightningModule`` in
``pl_simple_baseline.py``, since both are three lines over a shared base in
``workshop_infrastructure/lightning_modules/pl_base.py``. Importing it from here still
works so older notebooks and scripts do not break.
"""

from downstream_apps.template.lightning_modules.pl_simple_baseline import (  # noqa: F401
    FlareFinetuneLightningModule,
    FlareLightningModule,
    flare_target,
)

__all__ = ["FlareFinetuneLightningModule", "FlareLightningModule", "flare_target"]
