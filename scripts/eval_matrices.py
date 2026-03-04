#!/usr/bin/env python3
"""Generate evaluation matrices/tables from simulation logs.

Example:
  python scripts/eval_matrices.py --dataset wiki2018 --models ilnse_A2 ilnse_xu lru
"""
from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def _md_table(headers: List[str], rows: List[List[Any]]) -> str:
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(_fmt(v) for v in row) + " |")
    return "\n".join(lines)


def _fmt(val: Any) -> str:
    if val is None:
        return "NA"
    if isinstance(val, float):
        if math.isnan(val):
            return "NA"
        return f"{val:.6g}"
    return str(val)


def _iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _rankdata(x: List[float]) -> List[float]:
    # average ranks for ties
    order = sorted(range(len(x)), key=lambda i: x[i])
    ranks = [0.0] * len(x)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and x[order[j]] == x[order[i]]:
            j += 1
        avg = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[order[k]] = avg
        i = j
    return ranks


def _spearman(x: List[float], y: List[float]) -> float | None:
    if len(x) != len(y) or len(x) < 2:
        return None
    rx = _rankdata(x)
    ry = _rankdata(y)
    mx = sum(rx) / len(rx)
    my = sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    denx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    deny = math.sqrt(sum((b - my) ** 2 for b in ry))
    if denx == 0 or deny == 0:
        return None
    return num / (denx * deny)


def _load_summaries(results_root: str, dataset: str) -> Dict[str, Dict[int, Dict[str, Any]]]:
    root = Path(results_root) / dataset
    summaries: Dict[str, Dict[int, Dict[str, Any]]] = defaultdict(dict)
    for path in root.glob("*_summary_*.json"):
        if path.name.endswith("_all_sizes.json"):
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("dataset") != dataset:
            continue
        if "cache_size_objects" not in data:
            continue
        model = data.get("model", "unknown")
        cache_size = int(data["cache_size_objects"])
        summaries[model][cache_size] = data
    return summaries


def _compute_slot_stats(slot_path: str) -> Dict[str, Any]:
    drift = []
    budget = []
    admit = []
    miss_rate = []
    pressure = []
    quality = []
    spread = []
    prec_eff = []
    gate_applied = 0
    slots = 0
    evictions = []
    inserts = []
    for row in _iter_jsonl(slot_path):
        if row.get("phase") != "cache":
            continue
        slots += 1
        if row.get("drift_norm") is not None:
            drift.append(row.get("drift_norm", 0.0))
        budget.append(row.get("admit_budget", 0))
        admit.append(row.get("admit_selected", 0))
        miss_rate.append(row.get("miss_rate", 0.0))
        pressure.append(row.get("pressure_mult", 0.0))
        quality.append(row.get("quality_mult", 0.0))
        spread.append(row.get("score_spread", 0.0))
        pe = row.get("admission_precision_eff")
        if pe is not None:
            prec_eff.append(pe)
        if row.get("score_gate_applied"):
            gate_applied += 1
        if "slot_cache_evictions" in row:
            evictions.append(row.get("slot_cache_evictions", 0))
        if "slot_cache_inserts" in row:
            inserts.append(row.get("slot_cache_inserts", 0))
    return {
        "slots": slots,
        "corr_drift_budget": _spearman(drift, budget) if drift else None,
        "corr_drift_admit": _spearman(drift, admit) if drift else None,
        "corr_miss_pressure": _spearman(miss_rate, pressure) if miss_rate else None,
        "quality_avg": sum(quality) / len(quality) if quality else None,
        "spread_avg": sum(spread) / len(spread) if spread else None,
        "precision_eff_avg": sum(prec_eff) / len(prec_eff) if prec_eff else None,
        "gate_rate": (gate_applied / slots) if slots > 0 else None,
        "evictions_avg": (sum(evictions) / len(evictions)) if evictions else None,
        "inserts_avg": (sum(inserts) / len(inserts)) if inserts else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--baseline", default=None, help="model name for gain vs baseline")
    parser.add_argument("--out-dir", default="results/paper_tables")
    args = parser.parse_args()

    summaries = _load_summaries(args.results_root, args.dataset)
    if not summaries:
        raise SystemExit(f"No summaries found for dataset {args.dataset}")

    models = args.models or sorted(summaries.keys())
    cache_sizes = sorted({c for m in models for c in summaries.get(m, {}).keys()})

    os.makedirs(args.out_dir, exist_ok=True)

    # Table 1: HR matrix
    hr_headers = ["Method"] + [str(c) for c in cache_sizes]
    hr_rows = []
    for m in models:
        row = [m]
        for c in cache_sizes:
            row.append(summaries.get(m, {}).get(c, {}).get("hit_ratio"))
        hr_rows.append(row)
    Path(args.out_dir, f"table_hr_{args.dataset}.md").write_text(_md_table(hr_headers, hr_rows))

    # Table 2: Gain vs baseline (if available)
    if args.baseline and args.baseline in summaries:
        gain_headers = ["Method"] + [str(c) for c in cache_sizes]
        gain_rows = []
        for m in models:
            row = [m]
            for c in cache_sizes:
                base = summaries[args.baseline].get(c, {}).get("hit_ratio")
                cur = summaries.get(m, {}).get(c, {}).get("hit_ratio")
                row.append((cur - base) if (cur is not None and base is not None) else None)
            gain_rows.append(row)
        Path(args.out_dir, f"table_gain_{args.dataset}.md").write_text(
            _md_table(gain_headers, gain_rows)
        )

    # Table 3: Admission intensity summary (per cache size)
    ctrl_headers = ["Method", "Cache", "avg_M", "avg_A", "admit_rate", "miss_rate_avg"]
    ctrl_rows = []
    for m in models:
        for c in cache_sizes:
            s = summaries.get(m, {}).get(c, {})
            cache_slots = s.get("cache_slots") or 0
            admit_budget_total = s.get("admit_budget_total")
            admit_selected_total = s.get("admit_selected_total")
            miss_candidates_total = s.get("miss_candidates_total")
            avg_M = (
                admit_budget_total / cache_slots
                if (admit_budget_total is not None and cache_slots)
                else None
            )
            avg_A = (
                admit_selected_total / cache_slots
                if (admit_selected_total is not None and cache_slots)
                else None
            )
            admit_rate = (
                admit_selected_total / miss_candidates_total
                if (admit_selected_total is not None and miss_candidates_total)
                else None
            )
            ctrl_rows.append([m, c, avg_M, avg_A, admit_rate, s.get("miss_rate_avg")])
    Path(args.out_dir, f"table_control_{args.dataset}.md").write_text(
        _md_table(ctrl_headers, ctrl_rows)
    )

    # Table 4: Stability & overhead (summary + slot-level when available)
    stab_headers = [
        "Method",
        "Cache",
        "precision_eff_avg",
        "pollution_rate",
        "cache_evictions_avg",
        "cache_inserts_avg",
        "quality_mult_avg",
        "score_spread_avg",
        "gate_rate",
        "avg_update_time_s",
        "avg_score_time_s",
    ]
    stab_rows = []
    for m in models:
        for c in cache_sizes:
            s = summaries.get(m, {}).get(c, {})
            slot_stats = {}
            slot_path = s.get("slot_log_path")
            if slot_path and os.path.exists(slot_path):
                slot_stats = _compute_slot_stats(slot_path)
            stab_rows.append(
                [
                    m,
                    c,
                    slot_stats.get("precision_eff_avg") or s.get("admission_precision_eff_avg"),
                    s.get("pollution_rate_total"),
                    slot_stats.get("evictions_avg") or s.get("cache_evictions_avg"),
                    slot_stats.get("inserts_avg") or s.get("cache_inserts_avg"),
                    slot_stats.get("quality_avg") or s.get("quality_mult_avg"),
                    slot_stats.get("spread_avg") or s.get("score_spread_avg"),
                    slot_stats.get("gate_rate"),
                    s.get("avg_update_time_s"),
                    s.get("avg_score_time_s"),
                ]
            )
    Path(args.out_dir, f"table_stability_{args.dataset}.md").write_text(
        _md_table(stab_headers, stab_rows)
    )

    # Table 5: Drift/control correlations (slot-level)
    corr_headers = ["Method", "Cache", "corr_drift_budget", "corr_drift_admit", "corr_miss_pressure"]
    corr_rows = []
    for m in models:
        for c in cache_sizes:
            s = summaries.get(m, {}).get(c, {})
            slot_path = s.get("slot_log_path")
            if slot_path and os.path.exists(slot_path):
                slot_stats = _compute_slot_stats(slot_path)
                corr_rows.append(
                    [
                        m,
                        c,
                        slot_stats.get("corr_drift_budget"),
                        slot_stats.get("corr_drift_admit"),
                        slot_stats.get("corr_miss_pressure"),
                    ]
                )
    if corr_rows:
        Path(args.out_dir, f"table_corr_{args.dataset}.md").write_text(
            _md_table(corr_headers, corr_rows)
        )


if __name__ == "__main__":
    main()
