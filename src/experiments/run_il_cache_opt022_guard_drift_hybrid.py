from __future__ import annotations

import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.experiments import run_il_cache_opt022_guard_only as guard_full


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run guard_full with drift-aware budgeting (hybrid).",
    )
    parser.add_argument(
        "--profile",
        choices=("scaled", "piecewise_aggressive", "piecewise_conservative", "fixed"),
        default="piecewise_aggressive",
        help="Drift control profile for guard_full.",
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
    parser.add_argument(
        "--base-learners",
        nargs="+",
        choices=("nb", "svm", "dt"),
        default=("nb",),
        help="Base learners to run.",
    )
    return parser.parse_args()


def _apply_profile(name: str) -> None:
    if name == "fixed":
        guard_full.DRIFT_CONTROL_MODE = "fixed"
        guard_full.DRIFT_GAIN = 0.0
        return
    if name == "scaled":
        guard_full.DRIFT_CONTROL_MODE = "scaled"
        guard_full.DRIFT_GAIN = 2.5
        guard_full.DRIFT_NORM_POWER = 1.0
        guard_full.DRIFT_ALPHA_MIN = 0.01
        guard_full.DRIFT_ALPHA_MAX = 0.35
        guard_full.DRIFT_USE_CAPACITY_SCALE = True
        return
    if name == "piecewise_conservative":
        guard_full.DRIFT_CONTROL_MODE = "piecewise"
        guard_full.DRIFT_NORM_POWER = 1.0
        guard_full.DRIFT_THRESHOLD = 0.6
        guard_full.DRIFT_ALPHA_LOW = 0.04
        guard_full.DRIFT_ALPHA_HIGH = 0.18
        guard_full.DRIFT_ALPHA_MIN = 0.01
        guard_full.DRIFT_ALPHA_MAX = 0.25
        guard_full.DRIFT_USE_CAPACITY_SCALE = True
        return
    # piecewise_aggressive (default)
    guard_full.DRIFT_CONTROL_MODE = "piecewise"
    guard_full.DRIFT_NORM_POWER = 0.7
    guard_full.DRIFT_THRESHOLD = 0.5
    guard_full.DRIFT_ALPHA_LOW = 0.02
    guard_full.DRIFT_ALPHA_HIGH = 0.30
    guard_full.DRIFT_ALPHA_MIN = 0.005
    guard_full.DRIFT_ALPHA_MAX = 0.40
    guard_full.DRIFT_USE_CAPACITY_SCALE = True


def main() -> None:
    args = _parse_args()
    _apply_profile(args.profile)

    datasets = []
    if "both" in args.datasets:
        datasets = [
            guard_full.WIKIPEDIA_SEPTEMBER_2007,
            guard_full.WIKI2018,
        ]
    else:
        if "wikipedia_september_2007" in args.datasets:
            datasets.append(guard_full.WIKIPEDIA_SEPTEMBER_2007)
        if "wiki2018" in args.datasets:
            datasets.append(guard_full.WIKI2018)

    for ds in datasets:
        for feature_set in args.feature_sets:
            for learner in args.base_learners:
                model_name = f"ilnse_{feature_set}_guard_full_{learner}_drift_{args.profile}"
                guard_full.run_experiment(
                    ds_cfg=ds,
                    feature_set=feature_set,
                    model_name=model_name,
                    base_learner=learner,
                )


if __name__ == "__main__":
    main()
