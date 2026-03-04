from __future__ import annotations

import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.experiments import run_il_cache_opt022_drift_only as drift_only


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run drift-only experiments with different drift detectors.",
    )
    parser.add_argument(
        "--drift-methods",
        nargs="+",
        choices=("jsd", "adwin", "page_hinkley"),
        default=("jsd", "adwin", "page_hinkley"),
        help="Drift detectors to compare.",
    )
    parser.add_argument(
        "--feature-sets",
        nargs="+",
        choices=("A0", "A1", "A2", "A3"),
        default=("A2",),
        help="Feature sets to run.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=("wikipedia_september_2007", "wiki2018", "both"),
        default=("wikipedia_september_2007", "wiki2018"),
        help="Datasets to run.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    datasets = []
    if "both" in args.datasets:
        datasets = [
            drift_only.WIKIPEDIA_SEPTEMBER_2007,
            drift_only.WIKI2018,
        ]
    else:
        if "wikipedia_september_2007" in args.datasets:
            datasets.append(drift_only.WIKIPEDIA_SEPTEMBER_2007)
        if "wiki2018" in args.datasets:
            datasets.append(drift_only.WIKI2018)

    for ds in datasets:
        for feature_set in args.feature_sets:
            for method in args.drift_methods:
                model_name = f"ilnse_{feature_set}_drift_{method}"
                drift_only.run_experiment(
                    ds_cfg=ds,
                    feature_set=feature_set,
                    model_name=model_name,
                    drift_method=method,
                )


if __name__ == "__main__":
    main()
