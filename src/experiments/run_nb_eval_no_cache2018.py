# src/experiments/run_nb_eval_no_cache2018.py

from __future__ import annotations

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config.experiment_config import WIKI2018, ILConfig
from src.experiments.run_nb_eval_no_cache import run_nb_eval_no_cache, FEATURE_SETS


def main() -> None:
    ds = WIKI2018
    il_cfg = ILConfig()
    for feature_set in FEATURE_SETS:
        run_nb_eval_no_cache(
            trace_path=ds.path,
            total_requests=ds.num_total_requests,
            warmup_requests=ds.num_warmup_requests,
            slot_size=ds.slot_size,
            il_cfg=il_cfg,
            feature_set=feature_set,
            results_root="results_nse",
            dataset_name=f"{ds.name}_{feature_set}",
            dataset_base_name=ds.name,
        )


if __name__ == "__main__":
    main()
