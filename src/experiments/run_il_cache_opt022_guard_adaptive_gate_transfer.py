from __future__ import annotations

import argparse
import os
import sys
from typing import Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.experiments import run_il_cache_opt022_guard_drift_adaptive as adaptive


def _parse_tag(tag: str) -> Tuple[float, float, float, float]:
    parts = tag.split("_")
    kv = {}
    for p in parts:
        if p.startswith("gmn"):
            kv["gate_min"] = int(p[3:]) / 1000.0
        elif p.startswith("gmx"):
            kv["gate_max"] = int(p[3:]) / 1000.0
        elif p.startswith("sl"):
            kv["spread_low"] = int(p[2:]) / 1000.0
        elif p.startswith("sh"):
            kv["spread_high"] = int(p[2:]) / 1000.0
    if len(kv) != 4:
        raise ValueError(f"Invalid tag format: {tag}")
    return kv["gate_min"], kv["gate_max"], kv["spread_low"], kv["spread_high"]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run adaptive-gate transfer test with fixed parameters.",
    )
    parser.add_argument(
        "--dataset",
        choices=("wikipedia_september_2007", "wiki2018"),
        required=True,
    )
    parser.add_argument(
        "--feature-set",
        choices=("A1", "A2", "A3"),
        default="A2",
    )
    parser.add_argument(
        "--base-learner",
        choices=("nb", "svm", "dt"),
        default="nb",
    )
    parser.add_argument(
        "--gate-min",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--gate-max",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--spread-low",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--spread-high",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--from-tag",
        type=str,
        default="",
        help="Use params from tag like gmn24_gmx221_sl2_sh61.",
    )
    parser.add_argument(
        "--model-suffix",
        type=str,
        default="gate_transfer",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.from_tag:
        gate_min, gate_max, spread_low, spread_high = _parse_tag(args.from_tag)
    else:
        if None in (args.gate_min, args.gate_max, args.spread_low, args.spread_high):
            raise ValueError("Provide --from-tag or all --gate-min/--gate-max/--spread-low/--spread-high")
        gate_min, gate_max = float(args.gate_min), float(args.gate_max)
        spread_low, spread_high = float(args.spread_low), float(args.spread_high)

    adaptive.SCORE_GATE_USE_ADAPTIVE = True
    adaptive.SCORE_GATE_MIN_PERCENT = gate_min
    adaptive.SCORE_GATE_MAX_PERCENT = gate_max
    adaptive.SCORE_GATE_SPREAD_LOW = spread_low
    adaptive.SCORE_GATE_SPREAD_HIGH = spread_high

    if args.dataset == "wikipedia_september_2007":
        ds_cfg = adaptive.WIKIPEDIA_SEPTEMBER_2007
    else:
        ds_cfg = adaptive.WIKI2018

    tag = args.from_tag if args.from_tag else f"gmn{int(round(gate_min*1000))}_gmx{int(round(gate_max*1000))}_sl{int(round(spread_low*1000))}_sh{int(round(spread_high*1000))}"
    model_name = f"ilnse_{args.feature_set}_guard_full_{args.base_learner}_{args.model_suffix}_{tag}"
    adaptive.run_experiment(
        ds_cfg=ds_cfg,
        feature_set=args.feature_set,
        model_name=model_name,
        base_learner=args.base_learner,
    )


if __name__ == "__main__":
    main()
