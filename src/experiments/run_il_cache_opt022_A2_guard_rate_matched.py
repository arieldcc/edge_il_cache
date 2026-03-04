from __future__ import annotations

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.experiments import run_il_cache_opt022_guard_rate_matched as il


def main() -> None:
    model_name = "ilnse_A2_guard_rate_matched"
    for ds in (il.WIKIPEDIA_SEPTEMBER_2007, il.WIKI2018):
        il.run_experiment(
            ds,
            feature_set="A2",
            model_name=model_name,
            rate_match_model=il.RATE_MATCH_DEFAULT_MODEL,
        )


if __name__ == "__main__":
    main()
