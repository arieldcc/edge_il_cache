# src/experiments/run_il_cache_opt2007.py

from __future__ import annotations

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.experiments.run_il_cache_opt import WIKIPEDIA_SEPTEMBER_2007
from src.experiments.run_il_cache_opt import run_experiment


def main() -> None:
    for feature_set in ("A1", "A2", "A3"):
        model_name = f"ilnse_{feature_set}"
        run_experiment(WIKIPEDIA_SEPTEMBER_2007, feature_set=feature_set, model_name=model_name)


if __name__ == "__main__":
    main()
