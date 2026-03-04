#!/usr/bin/env python3
"""
Direct drift evaluation for Learn++.NSE without cache/admit/evict.

This script:
  - reads the trace directly (no cache simulation),
  - builds per-slot D_t,
  - updates Learn++.NSE per slot,
  - computes drift from slot frequency changes,
  - correlates NSE signals (E_t / weights) with drift.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from tqdm import tqdm

from src.config.experiment_config_all_ds import (
    CacheConfig,
    ILConfig,
    WIKI2018,
    WIKIPEDIA_OKTOBER_2007,
    WIKIPEDIA_SEPTEMBER_2007,
)
from src.data.feature_table import FeatureTable
from src.data.trace_reader import TraceReader
from src.ml.learn_nse_all_ds import LearnNSE
from src.experiments.run_il_cache_all_ds_template import (
    build_features_from_gaps,
    build_slot_dataset_from_stats,
)


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


def resolve_dataset(name: str):
    name = name.lower()
    if name == WIKIPEDIA_SEPTEMBER_2007.name:
        return WIKIPEDIA_SEPTEMBER_2007
    if name == WIKIPEDIA_OKTOBER_2007.name:
        return WIKIPEDIA_OKTOBER_2007
    if name == WIKI2018.name:
        return WIKI2018
    raise ValueError(f"Unknown dataset '{name}'.")


def count_unique(trace_path: str, total_requests: int) -> int:
    reader = TraceReader(path=trace_path, max_rows=total_requests)
    unique = set()
    for req in tqdm(reader.iter_requests(), total=total_requests, desc="Counting unique"):
        unique.add(req["object_id"])
        if len(unique) > total_requests:
            break
    return len(unique)


def compute_top_ids(
    slot_stats: Dict[str, Dict],
    label_mode: str,
    top_ratio: float,
    capacity_objects: int,
    label_capacity_gamma: float,
) -> List[str]:
    items = list(slot_stats.items())
    if not items:
        return []
    items.sort(key=lambda kv: kv[1]["freq"], reverse=True)
    n_objects = len(items)
    if label_mode == "capacity":
        k = int(round(float(capacity_objects) * float(label_capacity_gamma)))
        k = max(1, min(k, n_objects))
    elif label_mode == "top_ratio":
        k = int(n_objects * top_ratio)
        k = max(1, min(k, n_objects))
    else:
        raise ValueError(f"Unknown label_mode: {label_mode}")
    return [obj_id for obj_id, _ in items[:k]]


def analyze_direct(
    trace_path: str,
    total_requests: int,
    warmup_requests: int,
    slot_size: int,
    il_cfg: ILConfig,
    capacity_objects: int,
    include_warmup: bool,
) -> List[DriftSignals]:
    feature_table = FeatureTable(L=il_cfg.num_gaps, missing_gap_value=il_cfg.missing_gap_value)
    il_model = LearnNSE(
        n_features=il_cfg.num_gaps + (1 if il_cfg.use_freq_feature else 0),
        a=il_cfg.sigmoid_a,
        b=il_cfg.sigmoid_b,
        max_learners=il_cfg.max_classifiers,
    )

    slot_stats: Dict[str, Dict] = {}
    prev_slot_freq: Dict[str, int] = {}
    prev_freqs: Dict[str, int] = {}
    prev_top_ids: List[str] = []
    prev_weights: Optional[List[float]] = None

    warmup_slots = warmup_requests // slot_size if slot_size > 0 else 0
    signals: List[DriftSignals] = []

    reader = TraceReader(path=trace_path, max_rows=total_requests)
    pbar = tqdm(total=total_requests, desc="Reading trace")
    slot_index = 1
    slot_req_count = 0
    global_idx = 0
    prev_ts: Optional[float] = None

    for req in reader.iter_requests():
        if global_idx >= total_requests:
            break

        obj_id = req["object_id"]
        ts_raw = req.get("timestamp", None)
        if ts_raw is None:
            ts = (prev_ts + 1.0) if prev_ts is not None else float(global_idx + 1)
        else:
            ts_raw = float(ts_raw)
            if prev_ts is None:
                ts = ts_raw
            else:
                ts = ts_raw if ts_raw >= prev_ts else prev_ts
        prev_ts = ts

        gaps = feature_table.update_and_get_gaps(obj_id, ts)
        info = slot_stats.get(obj_id)
        if info is None:
            slot_stats[obj_id] = {"freq": 1, "last_gaps": gaps}
        else:
            info["freq"] += 1
            info["last_gaps"] = gaps

        slot_req_count += 1
        global_idx += 1
        pbar.update(1)

        if slot_req_count < slot_size:
            continue

        # finalize slot
        D_t, _label_info = build_slot_dataset_from_stats(
            slot_stats=slot_stats,
            top_ratio=il_cfg.pop_top_percent,
            num_gaps=il_cfg.num_gaps,
            missing_gap_value=il_cfg.missing_gap_value,
            label_mode=il_cfg.label_mode,
            capacity_objects=capacity_objects,
            label_capacity_gamma=il_cfg.label_capacity_gamma,
            use_freq_feature=il_cfg.use_freq_feature,
            freq_feature_mode=il_cfg.freq_feature_mode,
            freq_feature_source=il_cfg.freq_feature_source,
            prev_slot_freq=prev_slot_freq,
            slot_size=slot_size,
        )
        slot_info = il_model.update_slot(D_t)

        top_ids = compute_top_ids(
            slot_stats=slot_stats,
            label_mode=il_cfg.label_mode,
            top_ratio=il_cfg.pop_top_percent,
            capacity_objects=capacity_objects,
            label_capacity_gamma=il_cfg.label_capacity_gamma,
        )
        slot_freqs = {obj_id: int(stats["freq"]) for obj_id, stats in slot_stats.items()}
        drift_j, drift_tv = drift_metrics(slot_freqs, prev_freqs, top_ids, prev_top_ids) if prev_freqs else (None, None)

        if slot_index > warmup_slots or include_warmup:
            err_t = slot_info.get("E_t")
            if err_t is None and slot_info.get("acc_before") is not None:
                err_t = 1.0 - float(slot_info.get("acc_before"))

            weights = slot_info.get("weights")
            delta_w = None
            freeze = None
            if isinstance(weights, list) and weights:
                delta_w = weight_delta_l1(prev_weights or [], weights) if prev_weights is not None else None
                freeze = freeze_ratio(weights)
                prev_weights = weights

            signals.append(
                DriftSignals(
                    slot_index=slot_index,
                    drift_jaccard=drift_j,
                    drift_tv=drift_tv,
                    err_t=err_t,
                    weight_delta_l1=delta_w,
                    freeze_ratio=freeze,
                )
            )

        prev_freqs = slot_freqs
        prev_top_ids = top_ids
        prev_slot_freq = slot_freqs

        slot_stats = {}
        feature_table.reset_freqs()
        slot_index += 1
        slot_req_count = 0

    pbar.close()
    return signals


def summarize(signals: List[DriftSignals]) -> Dict[str, Optional[float]]:
    def collect(name: str) -> List[float]:
        out = []
        for s in signals:
            val = getattr(s, name)
            if val is not None:
                out.append(float(val))
        return out

    def collect_pair(name_x: str, name_y: str) -> Tuple[List[float], List[float]]:
        xs: List[float] = []
        ys: List[float] = []
        for s in signals:
            vx = getattr(s, name_x)
            vy = getattr(s, name_y)
            if vx is None or vy is None:
                continue
            xs.append(float(vx))
            ys.append(float(vy))
        return xs, ys

    drift = collect("drift_jaccard")
    err = collect("err_t")
    tv = collect("drift_tv")
    delta_w = collect("weight_delta_l1")
    freeze = collect("freeze_ratio")
    err_drift_x, err_drift_y = collect_pair("err_t", "drift_jaccard")
    err_tv_x, err_tv_y = collect_pair("err_t", "drift_tv")
    delta_drift_x, delta_drift_y = collect_pair("weight_delta_l1", "drift_jaccard")
    freeze_drift_x, freeze_drift_y = collect_pair("freeze_ratio", "drift_jaccard")

    return {
        "n_slots": len(signals),
        "spearman_err_vs_drift": spearman(err_drift_x, err_drift_y) if err_drift_x and err_drift_y else None,
        "spearman_err_vs_tv": spearman(err_tv_x, err_tv_y) if err_tv_x and err_tv_y else None,
        "spearman_weight_delta_vs_drift": spearman(delta_drift_x, delta_drift_y) if delta_drift_x and delta_drift_y else None,
        "spearman_freeze_vs_drift": spearman(freeze_drift_x, freeze_drift_y) if freeze_drift_x and freeze_drift_y else None,
        "drift_jaccard_mean": float(np.mean(drift)) if drift else None,
        "drift_tv_mean": float(np.mean(tv)) if tv else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Direct NSE drift evaluation (no cache).")
    parser.add_argument("--dataset", default=WIKIPEDIA_SEPTEMBER_2007.name, help="Dataset name in config.")
    parser.add_argument("--trace", default=None, help="Override trace path.")
    parser.add_argument("--include-warmup", action="store_true", help="Include warmup slots in analysis.")
    parser.add_argument("--capacity-objects", type=int, default=None, help="Capacity objects for label_mode=capacity.")
    parser.add_argument("--capacity-percent", type=float, default=None, help="Capacity percent for label_mode=capacity.")
    parser.add_argument("--out", default=None, help="Optional JSON output path.")
    args = parser.parse_args()

    ds_cfg = resolve_dataset(args.dataset)
    trace_path = args.trace or ds_cfg.path

    il_cfg = ILConfig()
    cache_cfg = CacheConfig()

    capacity_objects = args.capacity_objects
    if il_cfg.label_mode == "capacity" and capacity_objects is None:
        percent = args.capacity_percent
        if percent is None:
            percent = float(cache_cfg.cache_size_percentages[0])
        total_unique = count_unique(trace_path, ds_cfg.num_total_requests)
        capacity_objects = int(round(total_unique * (percent / 100.0)))

    if capacity_objects is None:
        capacity_objects = 1

    signals = analyze_direct(
        trace_path=trace_path,
        total_requests=ds_cfg.num_total_requests,
        warmup_requests=ds_cfg.num_warmup_requests,
        slot_size=ds_cfg.slot_size,
        il_cfg=il_cfg,
        capacity_objects=capacity_objects,
        include_warmup=bool(args.include_warmup),
    )
    summary = summarize(signals)

    print("=== Drift Signal Summary (direct) ===")
    for k, v in summary.items():
        print(f"{k}: {v}")

    if args.out:
        payload = {
            "dataset": ds_cfg.name,
            "trace": trace_path,
            "total_requests": ds_cfg.num_total_requests,
            "warmup_requests": ds_cfg.num_warmup_requests,
            "slot_size": ds_cfg.slot_size,
            "label_mode": il_cfg.label_mode,
            "capacity_objects": capacity_objects,
            "summary": summary,
            "signals": [s.__dict__ for s in signals],
        }
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
