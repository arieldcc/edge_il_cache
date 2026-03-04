# src/experiments/run_il_cache_opt_drift.py

from __future__ import annotations

import json
import math
import os
import sys
from typing import Dict, List, Tuple, Any, Optional
import numpy as np

from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.trace_reader import TraceReader

# Dataset configs (inline, no external config dependency)
WIKIPEDIA_SEPTEMBER_2007 = {
    "name": "wikipedia_september_2007",
    "path": "data/raw/wikipedia_september_2007/wiki.1190153705.gz",
    "num_total_requests": 9_000_000,
    "num_warmup_requests": 1_000_000,
    "slot_size": 100_000,
}
WIKIPEDIA_OKTOBER_2007 = {
    "name": "wikipedia_oktober_2007",
    "path": "data/raw/wikipedia_oktober_2007/wiki.1191201596.gz",
    "num_total_requests": 8_000_000,
    "num_warmup_requests": 1_000_000,
    "slot_size": 100_000,
}
WIKI2018 = {
    "name": "wiki2018",
    "path": "data/raw/wiki2018/wiki2018.gz",
    "num_total_requests": 10_000_000,
    "num_warmup_requests": 1_000_000,
    "slot_size": 100_000,
}

# Drift detection settings
IL_POP_TOP_PERCENT = 0.20
IL_LABEL_TOPK_ROUNDING = "floor"
IL_LABEL_TIE_BREAK = "none"
FREQ_HIST_BINS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)

DRIFT_NORM_EMA_ALPHA = 0.1
DRIFT_NORM_Z_MAX = 2.5


def _compute_topk_k(n_objects: int, top_ratio: float, rounding: str) -> int:
    raw_k = n_objects * top_ratio
    if rounding == "ceil":
        k = int(math.ceil(raw_k))
    elif rounding == "floor":
        k = int(raw_k)
    else:
        raise ValueError(f"Unknown label_topk_rounding: {rounding}")
    if k <= 0:
        k = 1
    if k > n_objects:
        k = n_objects
    return k


def _select_top_ids_from_stats(
    slot_stats: Dict[str, Dict],
    top_ratio: float,
    label_topk_rounding: str,
    label_tie_break: str,
) -> Tuple[set[str], Dict[str, Any]]:
    label_info: Dict[str, Any] = {
        "n_objects": 0,
        "k_target": 0,
        "k_actual": 0,
        "pos_ratio": 0.0,
        "freq_at_k": 0,
        "tie_count": 0,
        "tie_rate": 0.0,
    }
    if not slot_stats:
        return set(), label_info

    items: List[Tuple[str, Dict]] = list(slot_stats.items())
    items.sort(key=lambda kv: kv[1]["freq"], reverse=True)

    n_objects = len(items)
    k = _compute_topk_k(n_objects, top_ratio, label_topk_rounding)

    kth_freq = items[k - 1][1]["freq"]
    tie_count = sum(1 for _, stats in items if stats["freq"] == kth_freq)
    if label_tie_break == "include_ties":
        top_ids = {obj_id for obj_id, stats in items if stats["freq"] >= kth_freq}
    elif label_tie_break == "none":
        top_ids = {items[i][0] for i in range(k)}
    else:
        raise ValueError(f"Unknown label_tie_break: {label_tie_break}")

    label_info.update(
        {
            "n_objects": int(n_objects),
            "k_target": int(k),
            "k_actual": int(len(top_ids)),
            "pos_ratio": float(len(top_ids) / n_objects) if n_objects > 0 else 0.0,
            "freq_at_k": int(kth_freq),
            "tie_count": int(tie_count),
            "tie_rate": float(tie_count / n_objects) if n_objects > 0 else 0.0,
        }
    )
    return top_ids, label_info


def _compute_jsd(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    if p.size == 0 or q.size == 0:
        return 0.0
    p = p / max(float(p.sum()), 1e-12)
    q = q / max(float(q.sum()), 1e-12)
    m = 0.5 * (p + q)
    eps = 1e-12
    kl_pm = np.sum(p * (np.log(p + eps) - np.log(m + eps)))
    kl_qm = np.sum(q * (np.log(q + eps) - np.log(m + eps)))
    jsd = 0.5 * (kl_pm + kl_qm)
    return float(jsd / math.log(2.0))


def _compute_jsd_identity(
    prev_dist: Optional[Dict[str, float]],
    curr_dist: Optional[Dict[str, float]],
) -> float:
    if not prev_dist or not curr_dist:
        return 0.0
    keys = list(set(prev_dist) | set(curr_dist))
    if not keys:
        return 0.0
    p = np.fromiter((prev_dist.get(k, 0.0) for k in keys), dtype=float)
    q = np.fromiter((curr_dist.get(k, 0.0) for k in keys), dtype=float)
    return _compute_jsd(p, q)


def _slot_freq_hist(
    bins: Tuple[int, ...],
    freq_values: Optional[List[int]] = None,
) -> np.ndarray:
    if freq_values is None:
        return np.zeros(len(bins), dtype=float)
    values = np.asarray(freq_values, dtype=int)
    values = values[values > 0]
    if values.size == 0:
        return np.zeros(len(bins), dtype=float)
    edges = np.array(list(bins) + [float("inf")], dtype=float)
    hist, _ = np.histogram(values, bins=edges)
    return hist.astype(float)


def _compute_overlap_metrics(
    prev_ids: Optional[set[str]],
    curr_ids: Optional[set[str]],
) -> Tuple[float, float]:
    if not prev_ids or not curr_ids:
        return 0.0, 0.0
    inter = len(prev_ids & curr_ids)
    union = len(prev_ids | curr_ids)
    overlap = float(inter / max(1, len(curr_ids)))
    jaccard = float(inter / max(1, union))
    return overlap, jaccard


def save_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def get_next_run_id_drift(results_root: str, dataset: str, model: str) -> Tuple[str, str]:
    dataset_dir = os.path.join(results_root, dataset)
    os.makedirs(dataset_dir, exist_ok=True)

    existing_ids: List[int] = []
    for fname in os.listdir(dataset_dir):
        if not fname.endswith(f"_{model}.jsonl") and not fname.endswith(f"_summary_{model}.json"):
            continue
        prefix = fname.split("_", 1)[0]
        if len(prefix) == 3 and prefix.isdigit():
            existing_ids.append(int(prefix))

    next_id = 1 if not existing_ids else max(existing_ids) + 1
    run_id = f"{next_id:03d}"
    return run_id, dataset_dir


def run_drift_detection(
    trace_path: str,
    total_requests: int,
    warmup_requests: int,
    slot_size: int,
    slot_log_path: Optional[str] = None,
) -> Dict[str, Any]:
    reader = TraceReader(path=trace_path, max_rows=total_requests)
    req_iter = reader.iter_requests()

    label_topk_rounding = IL_LABEL_TOPK_ROUNDING
    label_tie_break = IL_LABEL_TIE_BREAK

    total_slots_expected = (total_requests + slot_size - 1) // slot_size
    pbar = tqdm(total=total_slots_expected, desc="Processing slots", unit="slot")

    global_idx = 0
    slot_req_count = 0
    slot_index = 0
    slot_stats: Dict[str, Dict[str, int]] = {}

    prev_topk_dist: Optional[Dict[str, float]] = None
    prev_topk_ids: Optional[set[str]] = None
    prev_topk_hist: Optional[np.ndarray] = None

    drift_mean_ema: Optional[float] = None
    drift_var_ema: Optional[float] = None

    # summary accumulators (cache phase only)
    drift_identity_sum = 0.0
    drift_identity_sq_sum = 0.0
    drift_shape_sum = 0.0
    drift_shape_sq_sum = 0.0
    drift_norm_sum = 0.0
    overlap_sum = 0.0
    jaccard_sum = 0.0
    total_cache_slots = 0
    total_slots_processed = 0

    slot_log_file = open(slot_log_path, "w", encoding="utf-8") if slot_log_path else None

    def _write_slot_log(payload: Dict[str, Any]) -> None:
        if slot_log_file is None:
            return
        slot_log_file.write(json.dumps(payload) + "\n")
        slot_log_file.flush()

    def _finalize_slot() -> None:
        nonlocal slot_index
        nonlocal slot_stats
        nonlocal prev_topk_dist
        nonlocal prev_topk_ids
        nonlocal prev_topk_hist
        nonlocal drift_mean_ema
        nonlocal drift_var_ema
        nonlocal drift_identity_sum
        nonlocal drift_identity_sq_sum
        nonlocal drift_shape_sum
        nonlocal drift_shape_sq_sum
        nonlocal drift_norm_sum
        nonlocal overlap_sum
        nonlocal jaccard_sum
        nonlocal total_cache_slots
        nonlocal total_slots_processed

        if not slot_stats:
            return

        slot_index += 1
        slot_num = slot_index
        total_slots_processed += 1
        warmup_slots = warmup_requests // slot_size
        is_cache_slot = slot_num > warmup_slots

        items: List[Tuple[str, Dict[str, int]]] = list(slot_stats.items())
        items.sort(key=lambda kv: kv[1]["freq"], reverse=True)
        top_ids, label_info = _select_top_ids_from_stats(
            slot_stats,
            IL_POP_TOP_PERCENT,
            label_topk_rounding,
            label_tie_break,
        )
        top_k = int(label_info.get("k_actual", 0))
        if top_k > len(items):
            top_k = len(items)

        top_items = items[:top_k]
        top_freq_total = sum(int(stats["freq"]) for _, stats in top_items)
        curr_topk_dist = None
        if top_freq_total > 0:
            curr_topk_dist = {
                obj_id: float(stats["freq"] / top_freq_total)
                for obj_id, stats in top_items
            }
        jsd_identity = _compute_jsd_identity(prev_topk_dist, curr_topk_dist)
        curr_topk_ids = set(curr_topk_dist.keys()) if curr_topk_dist else set()
        overlap, jaccard = _compute_overlap_metrics(prev_topk_ids, curr_topk_ids)

        top_freqs = [int(stats["freq"]) for _, stats in top_items]
        curr_hist = _slot_freq_hist(FREQ_HIST_BINS, freq_values=top_freqs)
        jsd_shape = _compute_jsd(prev_topk_hist, curr_hist) if prev_topk_hist is not None else 0.0

        drift_norm = 0.0
        if drift_mean_ema is None:
            drift_mean_ema = jsd_identity
            drift_var_ema = 0.0
        else:
            delta = jsd_identity - drift_mean_ema
            drift_mean_ema = (
                (1.0 - DRIFT_NORM_EMA_ALPHA) * drift_mean_ema
                + DRIFT_NORM_EMA_ALPHA * jsd_identity
            )
            drift_var_ema = (
                (1.0 - DRIFT_NORM_EMA_ALPHA) * drift_var_ema
                + DRIFT_NORM_EMA_ALPHA * (delta ** 2)
            )
            drift_std = math.sqrt((drift_var_ema or 0.0) + 1e-12)
            drift_z = (jsd_identity - drift_mean_ema) / drift_std if drift_std > 0 else 0.0
            if drift_z < 0.0:
                drift_z = 0.0
            if drift_z > DRIFT_NORM_Z_MAX:
                drift_z = DRIFT_NORM_Z_MAX
            drift_norm = drift_z / DRIFT_NORM_Z_MAX if DRIFT_NORM_Z_MAX > 0 else 0.0

        if is_cache_slot:
            total_cache_slots += 1
            drift_identity_sum += jsd_identity
            drift_identity_sq_sum += jsd_identity ** 2
            drift_shape_sum += jsd_shape
            drift_shape_sq_sum += jsd_shape ** 2
            drift_norm_sum += drift_norm
            overlap_sum += overlap
            jaccard_sum += jaccard

        slot_log = {
            "slot_index": slot_num,
            "phase": "warmup" if not is_cache_slot else "cache",
            "slot_requests": slot_req_count,
            "unique_objects": int(label_info.get("n_objects", 0)),
            "unique_ratio": float(label_info.get("n_objects", 0) / max(1, slot_req_count)),
            "topk_target": int(label_info.get("k_target", 0)),
            "topk_actual": int(label_info.get("k_actual", 0)),
            "topk_overlap": overlap,
            "topk_jaccard": jaccard,
            "jsd_identity": jsd_identity,
            "jsd_shape": jsd_shape,
            "drift_norm": drift_norm,
        }
        _write_slot_log(slot_log)

        prev_topk_dist = curr_topk_dist
        prev_topk_ids = curr_topk_ids
        prev_topk_hist = curr_hist
        slot_stats = {}

    try:
        while True:
            try:
                req = next(req_iter)
            except StopIteration:
                if slot_stats:
                    _finalize_slot()
                    pbar.update(1)
                break

            if global_idx >= total_requests:
                break

            obj_id = req["object_id"]
            info = slot_stats.get(obj_id)
            if info is None:
                slot_stats[obj_id] = {"freq": 1}
            else:
                info["freq"] += 1

            global_idx += 1
            slot_req_count += 1

            if slot_req_count >= slot_size:
                _finalize_slot()
                pbar.update(1)
                slot_req_count = 0
    finally:
        if slot_log_file is not None:
            slot_log_file.close()
        pbar.close()

    drift_identity_avg = float(drift_identity_sum / total_cache_slots) if total_cache_slots > 0 else None
    drift_identity_std = None
    if total_cache_slots > 0:
        mean = drift_identity_avg
        drift_identity_std = math.sqrt(
            max(drift_identity_sq_sum / total_cache_slots - mean * mean, 0.0)
        )
    drift_shape_avg = float(drift_shape_sum / total_cache_slots) if total_cache_slots > 0 else None
    drift_shape_std = None
    if total_cache_slots > 0:
        mean = drift_shape_avg
        drift_shape_std = math.sqrt(
            max(drift_shape_sq_sum / total_cache_slots - mean * mean, 0.0)
        )

    summary_metrics = {
        "slots_processed": total_slots_processed,
        "cache_slots_processed": total_cache_slots,
        "drift_identity_avg": drift_identity_avg,
        "drift_identity_std": drift_identity_std,
        "drift_shape_avg": drift_shape_avg,
        "drift_shape_std": drift_shape_std,
        "drift_norm_avg": float(drift_norm_sum / total_cache_slots) if total_cache_slots > 0 else None,
        "topk_overlap_avg": float(overlap_sum / total_cache_slots) if total_cache_slots > 0 else None,
        "topk_jaccard_avg": float(jaccard_sum / total_cache_slots) if total_cache_slots > 0 else None,
    }

    return summary_metrics


def run_experiment(dataset_cfg: Dict[str, Any]) -> None:
    dataset_name = dataset_cfg["name"]
    trace_path = dataset_cfg["path"]
    total_requests = dataset_cfg["num_total_requests"]
    warmup_requests = dataset_cfg["num_warmup_requests"]
    slot_size = dataset_cfg["slot_size"]
    results_root = "results"
    model_name = "drift"

    print(f"=== Drift Detection (no cache) {dataset_name} ===")
    print(f"Trace path        : {trace_path}")
    print(f"Total requests    : {total_requests}")
    print(f"Warm-up requests  : {warmup_requests}")
    print(f"Slot size         : {slot_size}")
    print(f"Top-K percent     : {IL_POP_TOP_PERCENT}")
    print(f"Top-K rounding    : {IL_LABEL_TOPK_ROUNDING}")
    print(f"Top-K tie-break   : {IL_LABEL_TIE_BREAK}")
    print()

    run_id, dataset_dir = get_next_run_id_drift(results_root, dataset_name, model_name)
    per_slot_path = os.path.join(dataset_dir, f"{run_id}_{model_name}.jsonl")
    summary_path = os.path.join(dataset_dir, f"{run_id}_summary_{model_name}.json")

    summary_metrics = run_drift_detection(
        trace_path=trace_path,
        total_requests=total_requests,
        warmup_requests=warmup_requests,
        slot_size=slot_size,
        slot_log_path=per_slot_path,
    )

    slots_processed = int(summary_metrics.get("slots_processed", 0))
    warmup_slots = warmup_requests // slot_size
    cache_slots = max(0, slots_processed - warmup_slots)

    summary_payload = {
        "dataset": dataset_name,
        "model": model_name,
        "total_requests": total_requests,
        "warmup_requests": warmup_requests,
        "slot_size": slot_size,
        "slot_log_path": per_slot_path,
        "num_slots": slots_processed,
        "warmup_slots": warmup_slots,
        "cache_slots": cache_slots,
        "pop_top_percent": IL_POP_TOP_PERCENT,
        "label_topk_rounding": IL_LABEL_TOPK_ROUNDING,
        "label_tie_break": IL_LABEL_TIE_BREAK,
        "freq_hist_bins": FREQ_HIST_BINS,
        "drift_norm_ema_alpha": DRIFT_NORM_EMA_ALPHA,
        "drift_norm_z_max": DRIFT_NORM_Z_MAX,
    }
    summary_payload.update(summary_metrics)
    save_json(summary_path, summary_payload)

    print(f"      [LOG] per-slot  -> {per_slot_path}")
    print(f"      [LOG] summary   -> {summary_path}")
    print()


def main() -> None:
    run_experiment(WIKIPEDIA_SEPTEMBER_2007)


if __name__ == "__main__":
    main()
