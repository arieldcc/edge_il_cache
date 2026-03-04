# src/experiments/run_il_cache_opt2018.py

from __future__ import annotations

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.experiments.run_il_cache_opt import WIKI2018
from src.experiments.run_il_cache_opt import run_experiment


def main() -> None:
    # for feature_set in ("A0", "A1", "A2", "A3"):
    model_name = f"ilnse_A2"
    run_experiment(WIKI2018, feature_set="A2", model_name=model_name)


if __name__ == "__main__":
    main()
