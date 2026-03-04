from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare adaptive drift+guard vs baseline guard_full (A2).",
    )
    parser.add_argument(
        "--dataset",
        choices=("wikipedia_september_2007", "wiki2018"),
        required=True,
    )
    parser.add_argument(
        "--model-suffix",
        default="drift_adapt_gate",
        help="Suffix used in adaptive model name.",
    )
    parser.add_argument(
        "--base-learner",
        choices=("nb", "svm", "dt"),
        default="nb",
    )
    parser.add_argument(
        "--feature",
        choices=("A1", "A2", "A3"),
        default="A2",
    )
    parser.add_argument(
        "--target",
        type=float,
        default=0.01,
        help="Target HR gain threshold (e.g., 0.01 for +1%).",
    )
    return parser.parse_args()


def _load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    args = _parse_args()
    base_dir = Path("results") / args.dataset

    base_name = f"ilnse_{args.feature}_guard_full_{args.base_learner.upper()}"
    adapt_name = f"ilnse_{args.feature}_guard_full_{args.base_learner}_{args.model_suffix}"

    base_path = base_dir / f"001_summary_{base_name}_all_sizes.json"
    adapt_path = base_dir / f"001_summary_{adapt_name}_all_sizes.json"

    if not base_path.exists():
        raise FileNotFoundError(base_path)
    if not adapt_path.exists():
        raise FileNotFoundError(adapt_path)

    base = _load(base_path)
    adapt = _load(adapt_path)

    base_curve = {r["cache_size_objects"]: r["hit_ratio"] for r in base["hr_curve"]}
    adapt_curve = {r["cache_size_objects"]: r["hit_ratio"] for r in adapt["hr_curve"]}

    sizes = sorted(base_curve.keys())
    print(f"Dataset: {args.dataset}")
    print(f"Baseline: {base_name}")
    print(f"Adaptive: {adapt_name}")
    print(f"Target gain >= {args.target:.3f}")
    print()

    hits = 0
    for s in sizes:
        b = base_curve[s]
        a = adapt_curve.get(s)
        if a is None:
            print(f"cap={s}: missing adaptive result")
            continue
        delta = a - b
        ok = delta >= args.target
        if ok:
            hits += 1
        print(f"cap={s}: base={b:.6f} adapt={a:.6f} delta={delta:+.6f} {'OK' if ok else 'FAIL'}")

    print()
    avg_base = base.get("avg_hr")
    avg_adapt = adapt.get("avg_hr")
    if avg_base is not None and avg_adapt is not None:
        avg_delta = avg_adapt - avg_base
        print(f"avg: base={avg_base:.6f} adapt={avg_adapt:.6f} delta={avg_delta:+.6f}")
    print(f"capacities meeting target: {hits}/{len(sizes)}")


if __name__ == "__main__":
    main()
