#!/usr/bin/env python3
"""
Analyze whether Learn++.NSE signals respond to drift.

Usage:
  python -m src.experiments.analyze_nse_drift --log results/...jsonl
  python -m src.experiments.analyze_nse_drift --log results/...jsonl --trace /path/to/trace.gz
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.config.experiment_config_all_ds import (
    WIKI2018,
    WIKIPEDIA_SEPTEMBER_2007,
    WIKIPEDIA_OKTOBER_2007,
)
from src.data.trace_reader import TraceReader


@dataclass
class DriftSignals:
    slot_index: int
    drift_jaccard: Optional[float]
    drift_tv: Optional[float]
    err_t: Optional[float]
    weight_delta_l1: Optional[float]
    freeze_ratio: Optional[float]


def _rankdata(values: Sequence[float]) -> List[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[order[j]] == values[order[i]]:
            j += 1
        avg_rank = (i + j - 1) / 2.0 + 1.0
        for k in range(i, j):
            ranks[order[k]] = avg_rank
        i = j
    return ranks


def _pearson(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    if len(x) < 2 or len(y) < 2:
        return None
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    x_mean = float(np.mean(x_arr))
    y_mean = float(np.mean(y_arr))
    x_dev = x_arr - x_mean
    y_dev = y_arr - y_mean
    denom = float(np.sqrt(np.sum(x_dev * x_dev) * np.sum(y_dev * y_dev)))
    if denom <= 0.0:
        return None
    return float(np.sum(x_dev * y_dev) / denom)


def spearman(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    if len(x) < 2 or len(y) < 2:
        return None
    rx = _rankdata(x)
    ry = _rankdata(y)
    return _pearson(rx, ry)


def resolve_trace_path(dataset_name: str, override: Optional[str]) -> str:
    if override:
        return override
    name = dataset_name.lower()
    if name == WIKIPEDIA_SEPTEMBER_2007.name:
        return WIKIPEDIA_SEPTEMBER_2007.path
    if name == WIKIPEDIA_OKTOBER_2007.name:
        return WIKIPEDIA_OKTOBER_2007.path
    if name == WIKI2018.name:
        return WIKI2018.path
    raise ValueError(f"Unknown dataset '{dataset_name}'. Pass --trace explicitly.")


def compute_top_ids(
    freqs: Dict[str, int],
    label_mode: str,
    top_ratio: float,
    capacity_objects: int,
    label_capacity_gamma: float,
) -> List[str]:
    items = sorted(freqs.items(), key=lambda kv: kv[1], reverse=True)
    n_objects = len(items)
    if n_objects == 0:
        return []
    if label_mode == "capacity":
        k = int(round(float(capacity_objects) * float(label_capacity_gamma)))
        k = max(1, min(k, n_objects))
    elif label_mode == "top_ratio":
        k = int(n_objects * top_ratio)
        k = max(1, min(k, n_objects))
    else:
        raise ValueError(f"Unknown label_mode: {label_mode}")
    return [obj_id for obj_id, _ in items[:k]]


def drift_metrics(
    freqs: Dict[str, int],
    prev_freqs: Dict[str, int],
    top_ids: List[str],
    prev_top_ids: List[str],
) -> Tuple[Optional[float], Optional[float]]:
    if not top_ids or not prev_top_ids:
        drift_jaccard = None
    else:
        a = set(top_ids)
        b = set(prev_top_ids)
        denom = float(len(a | b))
        drift_jaccard = None if denom == 0 else float(1.0 - (len(a & b) / denom))

    # TV distance between slot frequency distributions (optional but cheap)
    total = float(sum(freqs.values()))
    prev_total = float(sum(prev_freqs.values()))
    if total <= 0.0 or prev_total <= 0.0:
        drift_tv = None
    else:
        tv = 0.0
        keys = set(freqs.keys()) | set(prev_freqs.keys())
        for k in keys:
            p = freqs.get(k, 0) / total
            q = prev_freqs.get(k, 0) / prev_total
            tv += abs(p - q)
        drift_tv = 0.5 * tv

    return drift_jaccard, drift_tv


def weight_delta_l1(prev_w: Sequence[float], cur_w: Sequence[float]) -> Optional[float]:
    if not prev_w or not cur_w:
        return None
    n = max(len(prev_w), len(cur_w))
    a = np.zeros(n, dtype=float)
    b = np.zeros(n, dtype=float)
    a[: len(prev_w)] = np.asarray(prev_w, dtype=float)
    b[: len(cur_w)] = np.asarray(cur_w, dtype=float)
    if a.sum() <= 0.0 or b.sum() <= 0.0:
        return None
    a = a / a.sum()
    b = b / b.sum()
    return float(np.sum(np.abs(a - b)))


def freeze_ratio(weights: Sequence[float], eps: float = 1e-6) -> Optional[float]:
    if not weights:
        return None
    w = np.asarray(weights, dtype=float)
    return float(np.mean(w <= eps))


def analyze(log_path: str, trace_path: str, include_warmup: bool) -> List[DriftSignals]:
    with open(log_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    slots = payload.get("slots", [])
    dataset_name = payload.get("dataset")
    total_requests = int(payload.get("total_requests"))
    warmup_requests = int(payload.get("warmup_requests"))
    slot_size = int(payload.get("slot_size"))
    label_mode = str(payload.get("label_mode", "capacity"))
    label_capacity_gamma = float(payload.get("label_capacity_gamma", 1.2))
    top_ratio = float(payload.get("pop_top_percent", 0.2))
    capacity_objects = int(payload.get("cache_size_objects"))

    warmup_slots = warmup_requests // slot_size if slot_size > 0 else 0

    # Build per-slot freq dicts from trace
    reader = TraceReader(path=trace_path, max_rows=total_requests)
    freqs_by_slot: List[Dict[str, int]] = []
    current: Dict[str, int] = {}
    idx = 0

    for req in reader.iter_requests():
        obj_id = req["object_id"]
        current[obj_id] = current.get(obj_id, 0) + 1
        idx += 1
        if idx % slot_size == 0:
            freqs_by_slot.append(current)
            current = {}
        if idx >= total_requests:
            break
    if current:
        freqs_by_slot.append(current)

    top_ids_by_slot: List[List[str]] = []
    for freqs in freqs_by_slot:
        top_ids = compute_top_ids(
            freqs,
            label_mode=label_mode,
            top_ratio=top_ratio,
            capacity_objects=capacity_objects,
            label_capacity_gamma=label_capacity_gamma,
        )
        top_ids_by_slot.append(top_ids)

    slot_map = {int(s.get("slot_index")): s for s in slots if s.get("slot_index")}

    signals: List[DriftSignals] = []
    prev_freqs: Dict[str, int] = {}
    prev_top: List[str] = []
    prev_weights: Optional[List[float]] = None

    for slot_idx in range(1, len(freqs_by_slot) + 1):
        freqs = freqs_by_slot[slot_idx - 1]
        top_ids = top_ids_by_slot[slot_idx - 1]
        drift_j, drift_tv = drift_metrics(freqs, prev_freqs, top_ids, prev_top) if prev_freqs else (None, None)

        slot = slot_map.get(slot_idx, {})
        if slot_idx <= warmup_slots and not include_warmup:
            prev_freqs = freqs
            prev_top = top_ids
            if "weights" in slot:
                prev_weights = slot.get("weights")
            continue

        err_t = slot.get("E_t")
        if err_t is None and slot.get("acc_before") is not None:
            err_t = 1.0 - float(slot.get("acc_before"))

        weights = slot.get("weights")
        delta_w = None
        freeze = None
        if isinstance(weights, list) and weights:
            delta_w = weight_delta_l1(prev_weights or [], weights) if prev_weights is not None else None
            freeze = freeze_ratio(weights)
            prev_weights = weights

        signals.append(
            DriftSignals(
                slot_index=slot_idx,
                drift_jaccard=drift_j,
                drift_tv=drift_tv,
                err_t=err_t,
                weight_delta_l1=delta_w,
                freeze_ratio=freeze,
            )
        )

        prev_freqs = freqs
        prev_top = top_ids

    return signals


def summarize(signals: List[DriftSignals]) -> Dict[str, Optional[float]]:
    def collect(name: str) -> List[float]:
        out = []
        for s in signals:
            val = getattr(s, name)
            if val is not None:
                out.append(float(val))
        return out

    drift = collect("drift_jaccard")
    err = collect("err_t")
    tv = collect("drift_tv")
    delta_w = collect("weight_delta_l1")
    freeze = collect("freeze_ratio")

    summary = {
        "n_slots": len(signals),
        "spearman_err_vs_drift": spearman(err, drift) if drift and err else None,
        "spearman_err_vs_tv": spearman(err, tv) if tv and err else None,
        "spearman_weight_delta_vs_drift": spearman(delta_w, drift) if drift and delta_w else None,
        "spearman_freeze_vs_drift": spearman(freeze, drift) if drift and freeze else None,
        "drift_jaccard_mean": float(np.mean(drift)) if drift else None,
        "drift_tv_mean": float(np.mean(tv)) if tv else None,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate NSE drift signals against trace drift.")
    parser.add_argument("--log", required=True, help="Per-slot JSON log (e.g., results/...jsonl)")
    parser.add_argument("--trace", default=None, help="Override trace path if dataset name is unknown")
    parser.add_argument("--include-warmup", action="store_true", help="Include warmup slots in analysis")
    parser.add_argument("--out", default=None, help="Optional JSON output path")
    args = parser.parse_args()

    with open(args.log, "r", encoding="utf-8") as f:
        payload = json.load(f)
    dataset_name = payload.get("dataset")
    trace_path = resolve_trace_path(dataset_name, args.trace)

    signals = analyze(args.log, trace_path, include_warmup=args.include_warmup)
    summary = summarize(signals)

    print("=== Drift Signal Summary ===")
    for k, v in summary.items():
        print(f"{k}: {v}")

    if args.out:
        out_payload = {
            "log": args.log,
            "trace": trace_path,
            "include_warmup": bool(args.include_warmup),
            "summary": summary,
            "signals": [s.__dict__ for s in signals],
        }
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(out_payload, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
