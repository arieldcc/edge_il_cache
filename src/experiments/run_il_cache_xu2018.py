# src/experiments/run_il_cache_xu2018.py

from __future__ import annotations

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config.experiment_config import WIKI2018
from src.experiments.run_il_cache_xu import run_experiment


def main() -> None:
    run_experiment(WIKI2018)


if __name__ == "__main__":
    main()
