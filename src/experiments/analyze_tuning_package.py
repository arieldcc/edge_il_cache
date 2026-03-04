from __future__ import annotations

import json
import math
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np


def _load_summary(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_slot_metric(path: str, key: str = "slot_hit_ratio") -> List[float]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("phase") != "cache":
                continue
            if key in row:
                out.append(float(row[key]))
    return out


def _block_bootstrap_rmr(
    base_hr: List[float], model_hr: List[float], block_len: int = 5, n_boot: int = 2000, seed: int = 7
) -> Tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = min(len(base_hr), len(model_hr))
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    base = 1.0 - np.array(base_hr[:n])
    model = 1.0 - np.array(model_hr[:n])
    base_mr = float(np.mean(base))
    model_mr = float(np.mean(model))
    rmr = (base_mr - model_mr) / base_mr if base_mr > 0 else float("nan")

    max_start = max(0, n - block_len)
    starts = np.arange(max_start + 1)
    n_blocks = int(math.ceil(n / block_len))
    boot = []
    for _ in range(n_boot):
        idx = rng.choice(starts, size=n_blocks, replace=True)
        blocks = []
        for s in idx:
            blocks.append(np.arange(s, min(s + block_len, n)))
        samp = np.concatenate(blocks)[:n]
        b_mr = float(np.mean(base[samp]))
        m_mr = float(np.mean(model[samp]))
        boot.append((b_mr - m_mr) / b_mr if b_mr > 0 else float("nan"))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return rmr, float(lo), float(hi)


def _best_bayes_summary(dataset: str) -> Optional[dict]:
    files = []
    for fname in os.listdir(os.path.join("results", dataset)):
        if fname.endswith("_all_sizes.json") and "gate_bayes" in fname:
            files.append(os.path.join("results", dataset, fname))
    if not files:
        return None
    best = None
    for path in files:
        s = _load_summary(path)
        if best is None or s["avg_hr"] > best["avg_hr"]:
            best = s
            best["__path"] = path
    return best


def _best_bayes_tag(summary: dict) -> str:
    base = os.path.basename(summary["__path"])
    m = re.search(r"gate_bayes_(.*)_all_sizes\.json", base)
    if not m:
        return "unknown"
    return m.group(1)


def _rmr_table(dataset: str, model_prefix: str) -> List[dict]:
    base_summary = _load_summary(f"results/{dataset}/001_summary_ilnse_A2_guard_full_NB_all_sizes.json")
    sizes = [r["cache_size_objects"] for r in base_summary["hr_curve"]]
    rows = []
    for size in sizes:
        base_log = f"results/{dataset}/001_ilnse_A2_guard_full_NB_{size}.jsonl"
        model_log = f"results/{dataset}/001_{model_prefix}_{size}.jsonl"
        base_hr = _load_slot_metric(base_log)
        model_hr = _load_slot_metric(model_log)
        rmr, lo, hi = _block_bootstrap_rmr(base_hr, model_hr)
        rows.append({"size": size, "rmr": rmr, "ci_lo": lo, "ci_hi": hi})
    return rows


def _print_rmr_table(title: str, rows: List[dict]) -> None:
    print(f"\n{title}")
    print("size | RMR | 95% CI")
    print("--- | --- | ---")
    for r in rows:
        print(f"{r['size']} | {r['rmr']:+.6f} | [{r['ci_lo']:+.6f}, {r['ci_hi']:+.6f}]")


def _negative_case_analysis(dataset: str, size: int, model_prefix: str) -> dict:
    base_log = f"results/{dataset}/001_ilnse_A2_guard_full_NB_{size}.jsonl"
    model_log = f"results/{dataset}/001_{model_prefix}_{size}.jsonl"

    def _stats(path: str) -> Dict[str, float]:
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if r.get("phase") != "cache":
                    continue
                rows.append(r)
        def mean(key: str) -> float:
            vals = [r.get(key) for r in rows if key in r and r.get(key) is not None]
            return float(np.mean(vals)) if vals else float("nan")
        def rate_bool(key: str) -> float:
            vals = [r.get(key) for r in rows if key in r]
            return float(np.mean(vals)) if vals else float("nan")
        return {
            "miss_rate": mean("miss_rate"),
            "admit_budget": mean("admit_budget"),
            "admit_applied": mean("admit_applied"),
            "admission_precision_eff": mean("admission_precision_eff"),
            "score_spread": mean("score_spread"),
            "score_gate_applied_rate": rate_bool("score_gate_applied"),
            "score_gate_percent": mean("score_gate_percent"),
        }

    base = _stats(base_log)
    model = _stats(model_log)
    out = {"base": base, "model": model}
    return out


def main() -> None:
    # Best Bayes vs baseline
    for dataset in ("wikipedia_september_2007", "wiki2018"):
        best = _best_bayes_summary(dataset)
        if not best:
            continue
        tag = _best_bayes_tag(best)
        model_prefix = f"ilnse_A2_guard_full_nb_gate_bayes_{tag}"
        rows = _rmr_table(dataset, model_prefix)
        _print_rmr_table(f"{dataset} RMR (Bayes best vs guard_full_NB)", rows)

    # Transfer test (expected model names)
    transfer_models = {
        "wikipedia_september_2007": "ilnse_A2_guard_full_nb_gate_transfer_from_wiki2018_gmn17_gmx172_sl5_sh77",
        "wiki2018": "ilnse_A2_guard_full_nb_gate_transfer_from_wiki2007_gmn24_gmx221_sl2_sh61",
    }
    for dataset, model_prefix in transfer_models.items():
        log_path = f"results/{dataset}/001_{model_prefix}_all_sizes.jsonl"
        if not os.path.exists(log_path.replace("_all_sizes.jsonl", "_12221.jsonl")) and not os.path.exists(
            log_path.replace("_all_sizes.jsonl", "_16123.jsonl")
        ):
            continue
        rows = _rmr_table(dataset, model_prefix)
        _print_rmr_table(f"{dataset} RMR (Transfer vs guard_full_NB)", rows)

    # Negative-case analysis for wiki2007 largest size
    dataset = "wikipedia_september_2007"
    best = _best_bayes_summary(dataset)
    if best:
        tag = _best_bayes_tag(best)
        model_prefix = f"ilnse_A2_guard_full_nb_gate_bayes_{tag}"
        neg = _negative_case_analysis(dataset, 76386, model_prefix)
        print("\nNegative-case analysis (wiki2007 size=76386):")
        print("metric | baseline | bayes_best")
        print("--- | --- | ---")
        for k in ["miss_rate", "admit_budget", "admit_applied", "admission_precision_eff", "score_spread", "score_gate_applied_rate", "score_gate_percent"]:
            print(f"{k} | {neg['base'][k]:.6f} | {neg['model'][k]:.6f}")


if __name__ == "__main__":
    main()
