# src/experiments/run_gdbt_cache_opt2007A2_okt.py

from __future__ import annotations

from src.experiments.run_gdbt_cache_opt import WIKIPEDIA_OKTOBER_2007
from src.experiments.run_gdbt_cache_opt import run_experiment


def main() -> None:
    run_experiment(WIKIPEDIA_OKTOBER_2007, feature_set="A2", model_name="gdbt_A2")


if __name__ == "__main__":
    main()
