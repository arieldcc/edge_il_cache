from __future__ import annotations

import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.experiments import run_il_cache_opt022_guard_drift_adaptive as adaptive


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run guard_full + drift-aware adaptive gating (A2 default).",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=("wikipedia_september_2007", "wiki2018", "both"),
        default=("wikipedia_september_2007", "wiki2018"),
    )
    parser.add_argument(
        "--feature-sets",
        nargs="+",
        choices=("A1", "A2", "A3"),
        default=("A2",),
    )
    parser.add_argument(
        "--base-learners",
        nargs="+",
        choices=("nb", "svm", "dt"),
        default=("nb",),
    )
    parser.add_argument(
        "--model-suffix",
        default="drift_adapt_gate",
        help="Suffix appended to model_name for output files.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    datasets = []
    if "both" in args.datasets:
        datasets = [
            adaptive.WIKIPEDIA_SEPTEMBER_2007,
            adaptive.WIKI2018,
        ]
    else:
        if "wikipedia_september_2007" in args.datasets:
            datasets.append(adaptive.WIKIPEDIA_SEPTEMBER_2007)
        if "wiki2018" in args.datasets:
            datasets.append(adaptive.WIKI2018)

    for ds in datasets:
        for feature_set in args.feature_sets:
            for learner in args.base_learners:
                model_name = f"ilnse_{feature_set}_guard_full_{learner}_{args.model_suffix}"
                adaptive.run_experiment(
                    ds_cfg=ds,
                    feature_set=feature_set,
                    model_name=model_name,
                    base_learner=learner,
                )


if __name__ == "__main__":
    main()
