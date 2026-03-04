# src/experiments/run_il_cache2018_topk.py

from __future__ import annotations

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config.experiment_config import WIKI2018, ILConfig, CacheConfig
from src.experiments.run_il_cache import run_experiment


def main() -> None:
    cache_cfg = CacheConfig()
    for top_percent in (0.10, 0.20, 0.30):
        il_cfg = ILConfig(pop_top_percent=top_percent)
        model_name = f"ilnse_top{int(round(top_percent * 100))}"
        run_experiment(
            WIKI2018,
            il_cfg=il_cfg,
            cache_cfg=cache_cfg,
            model_name=model_name,
        )


if __name__ == "__main__":
    main()
