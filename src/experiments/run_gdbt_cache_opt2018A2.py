# src/experiments/run_gdbt_cache_opt2018A2.py

from __future__ import annotations

from src.experiments.run_gdbt_cache_opt import WIKI2018
from src.experiments.run_gdbt_cache_opt import run_experiment


def main() -> None:
    run_experiment(WIKI2018, feature_set="A2", model_name="gdbt_A2")


if __name__ == "__main__":
    main()
