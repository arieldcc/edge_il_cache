from __future__ import annotations

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config.experiment_config_all_ds import (
    WIKI2018 as DS_CFG,
    ILConfig,
    CacheConfig,
)
from src.experiments.run_il_cache_all_ds_template import run_experiment


def main() -> None:
    run_experiment(ds_cfg=DS_CFG, il_cfg=ILConfig(), cache_cfg=CacheConfig())


if __name__ == "__main__":
    main()
