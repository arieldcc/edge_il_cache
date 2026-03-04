from __future__ import annotations

import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.experiments import run_il_cache_opt022_guard_no_guard as guard_no_guard
from src.experiments import run_il_cache_opt022_guard_only as guard_full


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run guard_full/guard_no_guard with selectable features and base learners.",
    )
    parser.add_argument(
        "--model",
        choices=("guard_full", "guard_no_guard", "both"),
        default="both",
        help="Which guard model to run.",
    )
    parser.add_argument(
        "--feature-sets",
        nargs="+",
        choices=("A1", "A2", "A3"),
        default=("A1", "A2", "A3"),
        help="Feature sets to run.",
    )
    parser.add_argument(
        "--base-learners",
        nargs="+",
        choices=("nb", "svm", "dt"),
        default=("nb", "svm", "dt"),
        help="Base learners to run.",
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
            guard_no_guard.WIKIPEDIA_SEPTEMBER_2007,
            guard_no_guard.WIKI2018,
        ]
    else:
        if "wikipedia_september_2007" in args.datasets:
            datasets.append(guard_no_guard.WIKIPEDIA_SEPTEMBER_2007)
        if "wiki2018" in args.datasets:
            datasets.append(guard_no_guard.WIKI2018)

    feature_sets = tuple(args.feature_sets)
    learners = []
    for key in args.base_learners:
        if key == "nb":
            learners.append(("nb", "NB"))
        elif key == "svm":
            learners.append(("svm", "SVM"))
        elif key == "dt":
            learners.append(("dt", "DT"))

    def run_guard_no_guard(ds_cfg):
        for feature_set in feature_sets:
            for learner_key, learner_label in learners:
                model_name = f"ilnse_{feature_set}_guard_no_guard_{learner_label}"
                guard_no_guard.run_experiment(
                    ds_cfg=ds_cfg,
                    feature_set=feature_set,
                    model_name=model_name,
                    base_learner=learner_key,
                )

    def run_guard_full(ds_cfg):
        for feature_set in feature_sets:
            for learner_key, learner_label in learners:
                model_name = f"ilnse_{feature_set}_guard_full_{learner_label}"
                guard_full.run_experiment(
                    ds_cfg=ds_cfg,
                    feature_set=feature_set,
                    model_name=model_name,
                    base_learner=learner_key,
                )

    for ds in datasets:
        if args.model in ("guard_no_guard", "both"):
            run_guard_no_guard(ds)
        if args.model in ("guard_full", "both"):
            run_guard_full(ds)


if __name__ == "__main__":
    main()
