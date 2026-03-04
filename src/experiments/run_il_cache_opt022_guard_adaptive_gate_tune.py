from __future__ import annotations

import argparse
import json
import os
import sys
from itertools import product
from typing import Dict, Iterable, List, Tuple, Optional

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.experiments import run_il_cache_opt022_guard_drift_adaptive as adaptive


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Adaptive gate tuning for guard_full (grid or Bayesian).",
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
        "--target",
        type=float,
        default=0.01,
        help="Target HR gain per cache size (e.g., 0.01 for +1%).",
    )
    parser.add_argument(
        "--no-drift",
        action="store_true",
        help="Disable drift gain (tune gate only).",
    )
    parser.add_argument(
        "--grid",
        type=str,
        default="",
        help="Optional JSON file for grid search values.",
    )
    parser.add_argument(
        "--bayes",
        action="store_true",
        help="Use Bayesian optimization (Gaussian process) instead of full grid.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=24,
        help="Number of Bayesian trials (includes init samples).",
    )
    parser.add_argument(
        "--init-samples",
        type=int,
        default=8,
        help="Number of initial random samples for Bayesian tuning.",
    )
    parser.add_argument(
        "--candidates",
        type=int,
        default=200,
        help="Number of random candidates used for EI maximization.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed for Bayesian tuning.",
    )
    return parser.parse_args()


def _load_grid(path: str) -> Dict[str, List[float]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _default_grid() -> Dict[str, List[float]]:
    return {
        # Aggressive grid defaults
        "gate_min": [0.01, 0.02, 0.03, 0.05],
        "gate_max": [0.15, 0.20, 0.25, 0.30],
        "spread_low": [0.002, 0.003, 0.005, 0.008],
        "spread_high": [0.02, 0.03, 0.05, 0.08],
    }


def _datasets(arg_list: Iterable[str]) -> List[dict]:
    if "both" in arg_list:
        return [adaptive.WIKIPEDIA_SEPTEMBER_2007, adaptive.WIKI2018]
    out = []
    if "wikipedia_september_2007" in arg_list:
        out.append(adaptive.WIKIPEDIA_SEPTEMBER_2007)
    if "wiki2018" in arg_list:
        out.append(adaptive.WIKI2018)
    return out


def _latest_summary(dataset_dir: str, model_name: str) -> str:
    suffix = f"_summary_{model_name}_all_sizes.json"
    files = []
    for fname in os.listdir(dataset_dir):
        if fname.endswith(suffix):
            files.append(os.path.join(dataset_dir, fname))
    if not files:
        raise FileNotFoundError(f"No summary for model {model_name} in {dataset_dir}")

    def _run_id(path: str) -> int:
        try:
            return int(os.path.basename(path).split("_", 1)[0])
        except Exception:
            return -1

    return sorted(files, key=_run_id)[-1]


def _baseline_summary(dataset_dir: str, feature: str, learner: str) -> dict:
    # Baseline: guard_full NB/SVM/DT (no adaptive gate suffix)
    model = f"ilnse_{feature}_guard_full_{learner.upper()}"
    path = os.path.join(dataset_dir, f"001_summary_{model}_all_sizes.json")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _evaluate(model_summary: dict, baseline_summary: dict, target: float) -> Tuple[float, int, int]:
    base_curve = {r["cache_size_objects"]: r["hit_ratio"] for r in baseline_summary["hr_curve"]}
    curve = {r["cache_size_objects"]: r["hit_ratio"] for r in model_summary["hr_curve"]}
    sizes = sorted(base_curve.keys())
    hits = 0
    for s in sizes:
        if (curve.get(s, -1) - base_curve[s]) >= target:
            hits += 1
    avg_delta = model_summary["avg_hr"] - baseline_summary["avg_hr"]
    return avg_delta, hits, len(sizes)


def _apply_params(params: Dict[str, float], no_drift: bool) -> None:
    adaptive.SCORE_GATE_USE_ADAPTIVE = True
    adaptive.SCORE_GATE_MIN_PERCENT = params["gate_min"]
    adaptive.SCORE_GATE_MAX_PERCENT = params["gate_max"]
    adaptive.SCORE_GATE_SPREAD_LOW = params["spread_low"]
    adaptive.SCORE_GATE_SPREAD_HIGH = params["spread_high"]

    if no_drift:
        adaptive.DRIFT_GAIN = 0.0


def _valid_params(params: Dict[str, float]) -> bool:
    return params["gate_min"] < params["gate_max"] and params["spread_low"] < params["spread_high"]


def _score_objective(avg_delta: float, hits: int) -> float:
    # prioritize meeting target across sizes
    return float(hits) + (avg_delta * 100.0)


def _bayes_suggest(
    rng: np.random.Generator,
    bounds: Dict[str, Tuple[float, float]],
    tried: List[Dict[str, float]],
    scores: List[float],
    candidates: int,
) -> Dict[str, float]:
    try:
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, WhiteKernel
    except Exception:
        return _random_sample(rng, bounds)

    if len(tried) < 2:
        return _random_sample(rng, bounds)

    X = np.array([[p["gate_min"], p["gate_max"], p["spread_low"], p["spread_high"]] for p in tried])
    y = np.array(scores, dtype=float)

    kernel = RBF(length_scale=[0.05, 0.1, 0.02, 0.05]) + WhiteKernel(noise_level=1e-6)
    gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=0)
    gpr.fit(X, y)

    cand = _random_samples(rng, bounds, candidates)
    Xc = np.array([[p["gate_min"], p["gate_max"], p["spread_low"], p["spread_high"]] for p in cand])
    mu, std = gpr.predict(Xc, return_std=True)
    best = float(np.max(y))
    # expected improvement
    z = (mu - best) / (std + 1e-9)
    from math import erf, sqrt
    cdf = 0.5 * (1.0 + np.vectorize(erf)(z / sqrt(2.0)))
    pdf = (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * z * z)
    ei = (mu - best) * cdf + std * pdf
    idx = int(np.argmax(ei))
    return cand[idx]


def _random_sample(rng: np.random.Generator, bounds: Dict[str, Tuple[float, float]]) -> Dict[str, float]:
    for _ in range(1000):
        params = {
            "gate_min": float(rng.uniform(*bounds["gate_min"])),
            "gate_max": float(rng.uniform(*bounds["gate_max"])),
            "spread_low": float(rng.uniform(*bounds["spread_low"])),
            "spread_high": float(rng.uniform(*bounds["spread_high"])),
        }
        if _valid_params(params):
            return params
    return {
        "gate_min": bounds["gate_min"][0],
        "gate_max": bounds["gate_max"][1],
        "spread_low": bounds["spread_low"][0],
        "spread_high": bounds["spread_high"][1],
    }


def _random_samples(
    rng: np.random.Generator, bounds: Dict[str, Tuple[float, float]], n: int
) -> List[Dict[str, float]]:
    out = []
    for _ in range(n):
        out.append(_random_sample(rng, bounds))
    return out


def _encode_params(params: Dict[str, float]) -> str:
    def q(x: float) -> int:
        return int(round(x * 1000))

    return f"gmn{q(params['gate_min'])}_gmx{q(params['gate_max'])}_sl{q(params['spread_low'])}_sh{q(params['spread_high'])}"


def main() -> None:
    args = _parse_args()
    grid = _load_grid(args.grid) if args.grid else _default_grid()

    keys = ["gate_min", "gate_max", "spread_low", "spread_high"]
    combos = list(product(*(grid[k] for k in keys)))

    datasets = _datasets(args.datasets)

    for ds in datasets:
        dataset_name = ds["name"]
        dataset_dir = os.path.join("results", dataset_name)
        print(f"=== DATASET {dataset_name} ===")
        for feature in args.feature_sets:
            for learner in args.base_learners:
                baseline = _baseline_summary(dataset_dir, feature, learner)
                best = None
                if not args.bayes:
                    for combo in combos:
                        params = dict(zip(keys, combo))
                        if not _valid_params(params):
                            continue
                        _apply_params(params, args.no_drift)
                        tag = _encode_params(params)
                        model_name = f"ilnse_{feature}_guard_full_{learner}_gate_tune_{tag}"
                        adaptive.run_experiment(
                            ds_cfg=ds,
                            feature_set=feature,
                            model_name=model_name,
                            base_learner=learner,
                        )
                        summary_path = _latest_summary(dataset_dir, model_name)
                        with open(summary_path, "r", encoding="utf-8") as f:
                            summary = json.load(f)
                        avg_delta, hits, total = _evaluate(summary, baseline, args.target)
                        print(
                            f"{model_name}: avg_delta={avg_delta:+.6f} "
                            f"meets_target={hits}/{total}"
                        )
                        if best is None or hits > best["hits"] or (hits == best["hits"] and avg_delta > best["avg_delta"]):
                            best = {
                                "params": params,
                                "model_name": model_name,
                                "avg_delta": avg_delta,
                                "hits": hits,
                                "total": total,
                            }
                else:
                    rng = np.random.default_rng(args.seed)
                    bounds = {
                        "gate_min": (min(grid["gate_min"]), max(grid["gate_min"])),
                        "gate_max": (min(grid["gate_max"]), max(grid["gate_max"])),
                        "spread_low": (min(grid["spread_low"]), max(grid["spread_low"])),
                        "spread_high": (min(grid["spread_high"]), max(grid["spread_high"])),
                    }
                    tried: List[Dict[str, float]] = []
                    scores: List[float] = []
                    trials = max(args.trials, args.init_samples)
                    for t in range(trials):
                        if t < args.init_samples:
                            params = _random_sample(rng, bounds)
                        else:
                            params = _bayes_suggest(rng, bounds, tried, scores, args.candidates)
                        if not _valid_params(params):
                            continue
                        _apply_params(params, args.no_drift)
                        tag = _encode_params(params)
                        model_name = f"ilnse_{feature}_guard_full_{learner}_gate_bayes_{tag}"
                        adaptive.run_experiment(
                            ds_cfg=ds,
                            feature_set=feature,
                            model_name=model_name,
                            base_learner=learner,
                        )
                        summary_path = _latest_summary(dataset_dir, model_name)
                        with open(summary_path, "r", encoding="utf-8") as f:
                            summary = json.load(f)
                        avg_delta, hits, total = _evaluate(summary, baseline, args.target)
                        score = _score_objective(avg_delta, hits)
                        tried.append(params)
                        scores.append(score)
                        print(
                            f"{model_name}: avg_delta={avg_delta:+.6f} "
                            f"meets_target={hits}/{total} score={score:.3f}"
                        )
                        if best is None or hits > best["hits"] or (hits == best["hits"] and avg_delta > best["avg_delta"]):
                            best = {
                                "params": params,
                                "model_name": model_name,
                                "avg_delta": avg_delta,
                                "hits": hits,
                                "total": total,
                            }
                if best:
                    print(
                        f"BEST {feature}/{learner}: {best['model_name']} "
                        f"avg_delta={best['avg_delta']:+.6f} "
                        f"meets_target={best['hits']}/{best['total']}"
                    )
        print()


if __name__ == "__main__":
    main()
