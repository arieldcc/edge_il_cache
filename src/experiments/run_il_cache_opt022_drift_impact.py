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
        description="Run drift-only with a stronger drift controller to force HR impact.",
    )
    parser.add_argument(
        "--profile",
        choices=("scaled", "piecewise_aggressive", "piecewise_conservative", "fixed"),
        default="piecewise_aggressive",
        help="Drift control profile.",
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
        choices=("A1", "A2", "A3"),
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


def _apply_profile(name: str) -> None:
    if name == "fixed":
        drift_only.DRIFT_CONTROL_MODE = "fixed"
        drift_only.DRIFT_GAIN = 0.0
        return
    if name == "scaled":
        drift_only.DRIFT_CONTROL_MODE = "scaled"
        drift_only.DRIFT_GAIN = 2.5
        drift_only.DRIFT_NORM_POWER = 1.0
        drift_only.DRIFT_ALPHA_MIN = 0.01
        drift_only.DRIFT_ALPHA_MAX = 0.35
        drift_only.DRIFT_USE_CAPACITY_SCALE = True
        return
    if name == "piecewise_conservative":
        drift_only.DRIFT_CONTROL_MODE = "piecewise"
        drift_only.DRIFT_NORM_POWER = 1.0
        drift_only.DRIFT_THRESHOLD = 0.6
        drift_only.DRIFT_ALPHA_LOW = 0.04
        drift_only.DRIFT_ALPHA_HIGH = 0.18
        drift_only.DRIFT_ALPHA_MIN = 0.01
        drift_only.DRIFT_ALPHA_MAX = 0.25
        drift_only.DRIFT_USE_CAPACITY_SCALE = True
        return
    # piecewise_aggressive (default)
    drift_only.DRIFT_CONTROL_MODE = "piecewise"
    drift_only.DRIFT_NORM_POWER = 0.7
    drift_only.DRIFT_THRESHOLD = 0.5
    drift_only.DRIFT_ALPHA_LOW = 0.02
    drift_only.DRIFT_ALPHA_HIGH = 0.30
    drift_only.DRIFT_ALPHA_MIN = 0.005
    drift_only.DRIFT_ALPHA_MAX = 0.40
    drift_only.DRIFT_USE_CAPACITY_SCALE = True


def main() -> None:
    args = _parse_args()
    _apply_profile(args.profile)

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
                model_name = f"ilnse_{feature_set}_drift_{method}_{args.profile}"
                drift_only.run_experiment(
                    ds_cfg=ds,
                    feature_set=feature_set,
                    model_name=model_name,
                    drift_method=method,
                )


if __name__ == "__main__":
    main()
