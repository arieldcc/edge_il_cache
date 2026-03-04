# src/experiments/run_il_cache.py

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

from src.config.experiment_config import (
    WIKIPEDIA_SEPTEMBER_2007, # WIKIPEDIA_OKTOBER_2007/WIKIPEDIA_SEPTEMBER_2007/WIKI2018
    DatasetConfig,
    ILConfig,
    CacheConfig,
)
from src.data.trace_reader import TraceReader
from src.data.feature_table import FeatureTable
from src.ml.learn_nse import LearnNSE
from src.cache.cache_simulator import CacheStats
from src.cache.lru import LRUCache


# ---------------------------------------------------------------------------
# Utilitas untuk membangun D_t dari statistik slot
# ---------------------------------------------------------------------------

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


def build_slot_dataset_from_stats(
    slot_stats: Dict[str, Dict],
    top_ratio: float,
    num_features: int,
    missing_gap_value: float,
    label_topk_rounding: str = "floor",
    label_tie_break: str = "none",
) -> Tuple[List[Dict], Dict[str, Any]]:
    """
    Membangun dataset D_t untuk Learn++.NSE dari statistik per-objek dalam satu slot.

    slot_stats: dict[object_id] -> {"freq": int, "last_gaps": List[float]}
    top_ratio: misalnya 0.20 (top-20% populer)
    num_features: L = jumlah fitur gap (6 di eksperimen utama)

    return: list of dict:
      {
          "x": List[float],  # Gap1..GapL
          "y": int,          # 0/1 (popularitas)
          "freq": int,
          "object_id": str,
      }
    """
    label_info: Dict[str, Any] = {
        "n_objects": 0,
        "k_target": 0,
        "k_actual": 0,
        "pos_ratio": 0.0,
        "top_ratio": float(top_ratio),
        "rounding": label_topk_rounding,
        "tie_break": label_tie_break,
        "freq_at_k": None,
        "tie_count": 0,
        "tie_rate": 0.0,
    }
    if not slot_stats:
        return [], label_info

    items: List[Tuple[str, Dict]] = list(slot_stats.items())
    # urutkan berdasarkan freq menurun (popularitas per-slot)
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

    dataset: List[Dict] = []
    for obj_id, stats in items:
        gaps = stats["last_gaps"]

        # safety: pastikan panjang fitur benar
        if len(gaps) != num_features:
            if len(gaps) < num_features:
                # gaps = list(gaps) + [0.0] * (num_features - len(gaps))
                gaps = list(gaps) + [missing_gap_value] * (num_features - len(gaps))
            else:
                gaps = list(gaps[:num_features])

        y = 1 if obj_id in top_ids else 0
        dataset.append(
            {
                "x": gaps,
                "y": y,
                "freq": stats["freq"],
                "object_id": obj_id,
            }
        )

    return dataset, label_info

# ---------------------------------------------------------------------------
# Ringkasan statistik D_t (fitur & frekuensi) + sampel objek per slot
# ---------------------------------------------------------------------------

def summarize_slot_dataset(
    D_t: List[Dict],
    num_features: int,
    sample_k: int = 20,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Menghasilkan:
      - feature_stats: mean/std/min/max per fitur (overall, y=1, y=0)
      - freq_stats   : mean/std/min/max frekuensi (overall, y=1, y=0)
      - sample_objects: sampai sample_k objek (freq desc) dengan x, y, freq, object_id
    """
    if not D_t:
        return {}, []

    X = np.asarray([row["x"] for row in D_t], dtype=float)      # shape (N, L)
    y = np.asarray([row["y"] for row in D_t], dtype=int)        # shape (N,)
    freq = np.asarray([row["freq"] for row in D_t], dtype=float)  # shape (N,)

    if X.shape[1] != num_features:
        raise ValueError(f"n_features mismatch in summarize_slot_dataset: expected {num_features}, got {X.shape[1]}")

    def _feat_stats(mask: np.ndarray) -> Dict[str, List[float]]:
        if mask.sum() == 0:
            return {"mean": [], "std": [], "min": [], "max": []}
        Xm = X[mask]
        return {
            "mean": Xm.mean(axis=0).tolist(),
            "std":  Xm.std(axis=0).tolist(),
            "min":  Xm.min(axis=0).tolist(),
            "max":  Xm.max(axis=0).tolist(),
        }

    def _freq_stats(mask: np.ndarray) -> Dict[str, float]:
        if mask.sum() == 0:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
        fm = freq[mask]
        return {
            "mean": float(fm.mean()),
            "std":  float(fm.std()),
            "min":  float(fm.min()),
            "max":  float(fm.max()),
        }

    mask_all = np.ones_like(y, dtype=bool)
    mask_y1 = (y == 1)
    mask_y0 = (y == 0)

    summary = {
        "feature_stats": {
            "overall": _feat_stats(mask_all),
            "y1":      _feat_stats(mask_y1),
            "y0":      _feat_stats(mask_y0),
        },
        "freq_stats": {
            "overall": _freq_stats(mask_all),
            "y1":      _freq_stats(mask_y1),
            "y0":      _freq_stats(mask_y0),
        },
    }

    # Ambil beberapa contoh objek (D_t sudah terurut freq desc)
    sample_k = min(sample_k, len(D_t))
    sample_objs: List[Dict[str, Any]] = []
    for row in D_t[:sample_k]:
        sample_objs.append(
            {
                "object_id": row["object_id"],
                "y": int(row["y"]),
                "freq": int(row["freq"]),
                "x": [float(v) for v in row["x"]],  # Gap1..GapL
            }
        )

    return summary, sample_objs


def compute_cold_start_stats(
    D_t: List[Dict],
    num_features: int,
    missing_gap_value: float,
) -> Dict[str, Any]:
    """
    Ringkas proporsi objek yang masih cold-start (hist_len < L).
    """
    if not D_t:
        return {
            "num_cold": 0,
            "ratio_cold": 0.0,
            "hist_len_mean": 0.0,
            "hist_len_min": 0,
            "hist_len_max": 0,
        }

    hist_lens: List[int] = []
    for row in D_t:
        gaps = row["x"][:num_features]
        hist_len = sum(1 for g in gaps if g != missing_gap_value)
        hist_lens.append(hist_len)

    num_cold = sum(1 for h in hist_lens if h < num_features)
    return {
        "num_cold": int(num_cold),
        "ratio_cold": float(num_cold / len(hist_lens)),
        "hist_len_mean": float(np.mean(hist_lens)),
        "hist_len_min": int(min(hist_lens)),
        "hist_len_max": int(max(hist_lens)),
    }


def compute_class1_metrics(
    D_t: List[Dict],
    il_model: LearnNSE,
) -> Dict[str, Any]:
    """
    Hitung confusion matrix dan metrik untuk kelas-1 (positive class) pada D_t,
    menggunakan model saat ini (sebelum update_slot).

    Output:
      {
        "tp": int, "fp": int, "fn": int, "tn": int,
        "precision": float, "recall": float, "f1": float,
        "support_pos": int, "support_neg": int,
        "pred_pos": int, "pred_neg": int
      }
    """
    tp = fp = fn = tn = 0

    for row in D_t:
        y_true = int(row["y"])
        y_pred = int(il_model.predict(row["x"]))  # pakai ensemble saat ini

        if y_true == 1 and y_pred == 1:
            tp += 1
        elif y_true == 0 and y_pred == 1:
            fp += 1
        elif y_true == 1 and y_pred == 0:
            fn += 1
        else:
            tn += 1

    pred_pos = tp + fp
    pred_neg = tn + fn
    support_pos = tp + fn
    support_neg = tn + fp

    precision = (tp / pred_pos) if pred_pos > 0 else 0.0
    recall = (tp / support_pos) if support_pos > 0 else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "support_pos": support_pos,
        "support_neg": support_neg,
        "pred_pos": pred_pos,
        "pred_neg": pred_neg,
    }

# ---------------------------------------------------------------------------
# Satu run IL+LRU untuk satu kapasitas cache
# ---------------------------------------------------------------------------

def run_single_capacity(
    trace_path: str,
    total_requests: int,
    warmup_requests: int,
    slot_size: int,
    capacity_objects: int,
    il_cfg: ILConfig,
) -> Tuple[CacheStats, List[Dict[str, Any]]]:
    """
    Menjalankan satu eksperimen IL+LRU untuk satu kapasitas cache.

    - Stream trace sekali:
        * slot demi slot (slot_size dari config, misal 100k)
        * warm-up pada prefix warmup_requests pertama (tanpa cache)
        * caching pada sisa request dengan IL+LRU
    - Learn++.NSE di-update setiap akhir slot dengan label top-20% per-slot.
    """
    # TraceReader otomatis deteksi:
    # - jika trace_path file .gz -> mode raw_gz
    # - jika trace_path direktori .parquet -> mode parquet
    reader = TraceReader(path=trace_path, max_rows=total_requests)
    req_iter = reader.iter_requests()

    feature_table = FeatureTable(
        L=il_cfg.num_gaps,
        missing_gap_value=il_cfg.missing_gap_value,
    )
    il_model = LearnNSE(
        n_features=il_cfg.num_gaps,
        a=il_cfg.sigmoid_a,
        b=il_cfg.sigmoid_b,
        max_learners=il_cfg.max_classifiers,
    )
    cache = LRUCache(capacity_objects=capacity_objects)
    stats = CacheStats(capacity_objects=capacity_objects)

    label_topk_rounding = getattr(il_cfg, "label_topk_rounding", "floor")
    label_tie_break = getattr(il_cfg, "label_tie_break", "none")

    slot_logs: List[Dict[str, Any]] = []

    total_slots_expected = (total_requests + slot_size - 1) // slot_size
    pbar = tqdm(total=total_slots_expected, desc="Processing slots", unit="slot")

    global_idx = 0          # counter global request (0-based)
    slot_req_count = 0      # jumlah request dalam slot saat ini
    slot_index = 0          # 1-based untuk logging
    slot_stats: Dict[str, Dict] = {}  # per-slot: freq & last_gaps
    prev_ts: Optional[float] = None
    slot_ts_with_value = 0
    slot_ts_corrected = 0
    total_ts_with_value = 0
    total_ts_corrected = 0
    slot_cache_requests = 0
    slot_cache_hits = 0

    while True:
        try:
            req = next(req_iter)
        except StopIteration:
            # flush slot terakhir jika ada
            if slot_stats:
                D_t, label_info = build_slot_dataset_from_stats(
                    slot_stats,
                    il_cfg.pop_top_percent,
                    il_cfg.num_gaps,
                    il_cfg.missing_gap_value,
                    label_topk_rounding,
                    label_tie_break,
                )

                # Ringkasan statistik & sampel objek untuk analisa mendalam
                dt_summary, sample_objs = summarize_slot_dataset(
                    D_t, il_cfg.num_gaps, sample_k=20
                )
                cold_start_stats = compute_cold_start_stats(
                    D_t, il_cfg.num_gaps, il_cfg.missing_gap_value
                )
                ts_ratio = (
                    float(slot_ts_corrected / slot_ts_with_value)
                    if slot_ts_with_value > 0
                    else 0.0
                )

                slot_index += 1
                # metrik sebelum update (H_{t-1} dievaluasi pada D_t)
                class1_metrics = compute_class1_metrics(D_t, il_model)

                slot_info = il_model.update_slot(D_t)

                # metrics AFTER update (model H_t)
                cls1_after = compute_class1_metrics(D_t, il_model)

                start_req = (slot_index - 1) * slot_size
                end_req = min(start_req + slot_req_count, total_requests) - 1
                if end_req < warmup_requests:
                    phase = "warmup"
                else:
                    phase = "cache"

                slot_log: Dict[str, Any] = {
                    "slot_index": slot_index,
                    "phase": phase,
                    "slot_size_requests": slot_req_count,
                    "slot_start_request_idx": start_req,
                    "slot_end_request_idx": end_req,
                    "slot_cache_requests": slot_cache_requests,
                    "slot_cache_hits": slot_cache_hits,
                    "slot_hit_ratio": (
                        float(slot_cache_hits / slot_cache_requests)
                        if slot_cache_requests > 0
                        else None
                    ),
                    "stats_total_requests": stats.total_requests,
                    "stats_cache_hits": stats.cache_hits,
                    "hit_ratio": stats.hit_ratio,
                    "cls1_metrics_before_update": class1_metrics,
                    "cls1_metrics_after_update": cls1_after,
                    # tambahan untuk analisa:
                    "label_sanity": label_info,
                    "cold_start": cold_start_stats,
                    "timestamp_corrections": {
                        "corrected": slot_ts_corrected,
                        "total_with_ts": slot_ts_with_value,
                        "ratio": ts_ratio,
                    },
                    "feature_stats": dt_summary.get("feature_stats", {}),
                    "freq_stats": dt_summary.get("freq_stats", {}),
                    "sample_objects": sample_objs,
                }
                slot_log.update(slot_info)
                slot_logs.append(slot_log)

                pbar.update(1)
                pbar.set_postfix(
                    hr=f"{stats.hit_ratio:.4f}" if stats.total_requests > 0 else "0.0000"
                )
            break

        if global_idx >= total_requests:
            break

        obj_id = req["object_id"]

        # gunakan timestamp dataset yang dimonotonkan (non-decreasing)
        ts_raw = req.get("timestamp", None)
        if ts_raw is None:
            ts = (prev_ts + 1.0) if prev_ts is not None else float(global_idx + 1)
        else:
            ts_raw = float(ts_raw)
            slot_ts_with_value += 1
            total_ts_with_value += 1
            if prev_ts is None:
                ts = ts_raw
            else:
                if ts_raw < prev_ts:
                    slot_ts_corrected += 1
                    total_ts_corrected += 1
                ts = ts_raw if ts_raw >= prev_ts else prev_ts
        prev_ts = ts

        # 1) update fitur (gap) & freq (slot_stats)
        gaps = feature_table.update_and_get_gaps(obj_id, ts)

        info = slot_stats.get(obj_id)
        if info is None:
            slot_stats[obj_id] = {
                "freq": 1,
                "last_gaps": gaps,
            }
        else:
            info["freq"] += 1
            info["last_gaps"] = gaps

        # 2) fase caching (setelah warm-up selesai)
        if global_idx >= warmup_requests:
            stats.total_requests += 1
            slot_cache_requests += 1
            if cache.access(obj_id):
                stats.cache_hits += 1
                slot_cache_hits += 1
            else:
                # cache miss → tanya IL
                y_hat = il_model.predict(gaps)
                if y_hat == 1:
                    cache.insert(obj_id)

        # 3) update counter & cek boundary slot
        global_idx += 1
        slot_req_count += 1

        if slot_req_count >= slot_size:
            D_t, label_info = build_slot_dataset_from_stats(
                slot_stats,
                il_cfg.pop_top_percent,
                il_cfg.num_gaps,
                il_cfg.missing_gap_value,
                label_topk_rounding,
                label_tie_break,
            )

            dt_summary, sample_objs = summarize_slot_dataset(
                D_t, il_cfg.num_gaps, sample_k=20
            )
            cold_start_stats = compute_cold_start_stats(
                D_t, il_cfg.num_gaps, il_cfg.missing_gap_value
            )
            ts_ratio = (
                float(slot_ts_corrected / slot_ts_with_value)
                if slot_ts_with_value > 0
                else 0.0
            )

            slot_index += 1
            # metrik sebelum update (H_{t-1} dievaluasi pada D_t)
            class1_metrics = compute_class1_metrics(D_t, il_model)
            
            slot_info = il_model.update_slot(D_t)

            # metrics AFTER update (model H_t)
            cls1_after = compute_class1_metrics(D_t, il_model)

            start_req = (slot_index - 1) * slot_size
            end_req = start_req + slot_req_count - 1
            if end_req < warmup_requests:
                phase = "warmup"
            else:
                phase = "cache"

            slot_log: Dict[str, Any] = {
                "slot_index": slot_index,
                "phase": phase,
                "slot_size_requests": slot_req_count,
                "slot_start_request_idx": start_req,
                "slot_end_request_idx": end_req,
                "slot_cache_requests": slot_cache_requests,
                "slot_cache_hits": slot_cache_hits,
                "slot_hit_ratio": (
                    float(slot_cache_hits / slot_cache_requests)
                    if slot_cache_requests > 0
                    else None
                ),
                "stats_total_requests": stats.total_requests,
                "stats_cache_hits": stats.cache_hits,
                "hit_ratio": stats.hit_ratio,
                "cls1_metrics_before_update": class1_metrics,
                "cls1_metrics_after_update": cls1_after,
                # tambahan:
                "label_sanity": label_info,
                "cold_start": cold_start_stats,
                "timestamp_corrections": {
                    "corrected": slot_ts_corrected,
                    "total_with_ts": slot_ts_with_value,
                    "ratio": ts_ratio,
                },
                "feature_stats": dt_summary.get("feature_stats", {}),
                "freq_stats": dt_summary.get("freq_stats", {}),
                "sample_objects": sample_objs,
            }
            slot_log.update(slot_info)
            slot_logs.append(slot_log)

            pbar.update(1)
            current_hr = stats.hit_ratio if stats.total_requests > 0 else 0.0
            pbar.set_postfix(hr=f"{current_hr:.4f}")

            slot_stats = {}
            slot_req_count = 0
            slot_ts_with_value = 0
            slot_ts_corrected = 0
            slot_cache_requests = 0
            slot_cache_hits = 0

    pbar.close()
    return stats, slot_logs


# ---------------------------------------------------------------------------
# Pre-scan: hitung jumlah objek unik
# ---------------------------------------------------------------------------

def count_distinct_objects(trace_path: str, total_requests: int) -> int:
    """
    Satu pass ringan untuk menghitung jumlah object_id unik
    pada prefix trace sepanjang total_requests.
    """
    reader = TraceReader(path=trace_path, max_rows=total_requests)
    req_iter = reader.iter_requests()

    unique_objects = set()
    count = 0

    for req in req_iter:
        obj_id = req["object_id"]
        unique_objects.add(obj_id)
        count += 1
        if count >= total_requests:
            break

    return len(unique_objects)


# ---------------------------------------------------------------------------
# Utilitas penomoran run dan penulisan JSON
# ---------------------------------------------------------------------------

def get_next_run_id(results_root: str, dataset: str, model: str, cache_size: int) -> Tuple[str, str]:
    """
    Menentukan ID run berikutnya (001, 002, ...) agar tidak overwrite.
    """
    dataset_dir = os.path.join(results_root, dataset)
    os.makedirs(dataset_dir, exist_ok=True)

    existing_ids: List[int] = []
    for fname in os.listdir(dataset_dir):
        if not fname.endswith(f"_{model}_{cache_size}.json") and \
           not fname.endswith(f"_summary_{model}_{cache_size}.json"):
            continue
        prefix = fname.split("_", 1)[0]
        if len(prefix) == 3 and prefix.isdigit():
            existing_ids.append(int(prefix))

    next_id = 1 if not existing_ids else max(existing_ids) + 1
    run_id = f"{next_id:03d}"
    return run_id, dataset_dir


def save_json(path: str, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _format_ratio(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def print_sample_slot_sanity(slot_logs: List[Dict[str, Any]], max_samples: int = 5) -> None:
    if not slot_logs:
        return

    n = len(slot_logs)
    if n <= max_samples:
        indices = list(range(n))
    else:
        candidates = [0, n // 4, n // 2, (3 * n) // 4, n - 1]
        indices = []
        for idx in candidates:
            if idx not in indices and 0 <= idx < n:
                indices.append(idx)

    print("Sample slot sanity (slot, n_obj, k, pos, tie_rate, cold_start, ts_corr, hr_slot)")
    for idx in indices:
        log = slot_logs[idx]
        label = log.get("label_sanity", {})
        cold = log.get("cold_start", {})
        ts = log.get("timestamp_corrections", {})
        print(
            "  "
            f"slot={log.get('slot_index')} "
            f"n_obj={label.get('n_objects')} "
            f"k={label.get('k_target')}/{label.get('k_actual')} "
            f"pos={label.get('k_actual')} "
            f"tie_rate={_format_ratio(label.get('tie_rate'))} "
            f"cold={_format_ratio(cold.get('ratio_cold'))} "
            f"ts_corr={_format_ratio(ts.get('ratio'))} "
            f"hr_slot={_format_ratio(log.get('slot_hit_ratio'))}"
        )

# hitung jumlah oobjek uniq
def get_dynamic_capacities(trace_path: str, total_requests: int, percentages: List[float]) -> List[int]:
    """
    Hitung kapasitas cache dalam jumlah objek berdasarkan persentase.
    """
    n_unique = count_distinct_objects(trace_path, total_requests)
    capacities = [int(n_unique * p / 100) for p in percentages]
    print(f"Jumlah objek unik: {n_unique}")
    print(f"Kapasitas cache dinamis (berdasarkan %): {capacities}")
    return capacities

# ---------------------------------------------------------------------------
# Main: TANPA argparse, semua pakai experiment_config
# ---------------------------------------------------------------------------

def run_experiment(
    ds_cfg: DatasetConfig,
    il_cfg: Optional[ILConfig] = None,
    cache_cfg: Optional[CacheConfig] = None,
    model_name: Optional[str] = None,
) -> None:
    if il_cfg is None:
        il_cfg = ILConfig()
    if cache_cfg is None:
        cache_cfg = CacheConfig()
    if model_name is None:
        model_name = "ilnse"
    label_topk_rounding = getattr(il_cfg, "label_topk_rounding", "floor")
    label_tie_break = getattr(il_cfg, "label_tie_break", "none")

    trace_path = ds_cfg.path                # path ke .gz (seperti di debug)
    slot_size = ds_cfg.slot_size            # misal 100000
    warmup_requests = ds_cfg.num_warmup_requests  # misal 1_000_000
    # asumsi: total request juga sudah ada di config
    # kalau nama field beda (misal ds_cfg.total_requests), ganti baris ini
    total_requests = ds_cfg.num_total_requests

    if warmup_requests % slot_size != 0:
        raise ValueError(
            f"warmup_requests ({warmup_requests}) harus kelipatan slot_size ({slot_size}) "
            "agar sesuai dengan setup eksperimen Xu."
        )

    dataset_name = ds_cfg.name     # misal "WIKI2018"
    results_root = "results"

    print(f"=== IL-based Edge Cache Experiment ({dataset_name} trace) ===")
    print(f"Trace path        : {trace_path}")
    print(f"Total requests    : {total_requests}")
    print(f"Warm-up requests  : {warmup_requests}")
    print(f"Slot size         : {slot_size}")
    print(f"IL num_gaps       : {il_cfg.num_gaps}")
    print(f"IL top_percent    : {il_cfg.pop_top_percent}")
    print(f"Label rounding    : {label_topk_rounding}")
    print(f"Label tie-break   : {label_tie_break}")
    print(f"IL sigmoid (a, b) : ({il_cfg.sigmoid_a}, {il_cfg.sigmoid_b})")
    print(f"Max learners      : {il_cfg.max_classifiers}")
    print(f"Cache capacities  : {list(cache_cfg.cache_size_percentages)}")
    print()

    # Hitung kapasitas dinamis
    capacities_objects = get_dynamic_capacities(trace_path, total_requests, cache_cfg.cache_size_percentages)

    results: List[CacheStats] = []

    for capacity in capacities_objects:
        print(f"[RUN] Capacity Objects={capacity}")

        stats, slot_logs = run_single_capacity(
            trace_path=trace_path,
            total_requests=total_requests,
            warmup_requests=warmup_requests,
            slot_size=slot_size,
            capacity_objects=capacity,
            il_cfg=il_cfg,
        )
        results.append(stats)

        total_ts_with_value = sum(
            log.get("timestamp_corrections", {}).get("total_with_ts", 0)
            for log in slot_logs
        )
        total_ts_corrected = sum(
            log.get("timestamp_corrections", {}).get("corrected", 0)
            for log in slot_logs
        )

        run_id, dataset_dir = get_next_run_id(results_root, dataset_name, model_name, capacity)

        per_slot_path = os.path.join(
            dataset_dir,
            f"{run_id}_{model_name}_{capacity}.jsonl",
        )
        summary_path = os.path.join(
            dataset_dir,
            f"{run_id}_summary_{model_name}_{capacity}.json",
        )

        per_slot_payload = {
            "dataset": dataset_name,
            "model": model_name,
            "cache_size_objects": capacity,
            "total_requests": total_requests,
            "warmup_requests": warmup_requests,
            "slot_size": slot_size,
            "slots": slot_logs,
        }
        save_json(per_slot_path, per_slot_payload)

        summary_payload = {
            "dataset": dataset_name,
            "model": model_name,
            "cache_size_objects": capacity,
            "total_requests": total_requests,
            "warmup_requests": warmup_requests,
            "slot_size": slot_size,
            "hit_ratio": stats.hit_ratio,
            "cache_hits": stats.cache_hits,
            "cache_requests": stats.total_requests,
            "num_slots": len(slot_logs),
            "warmup_slots": warmup_requests // slot_size,
            "cache_slots": max(0, len(slot_logs) - (warmup_requests // slot_size)),
            "pop_top_percent": il_cfg.pop_top_percent,
            "label_topk_rounding": label_topk_rounding,
            "label_tie_break": label_tie_break,
            "timestamp_corrections": {
                "corrected": total_ts_corrected,
                "total_with_ts": total_ts_with_value,
                "ratio": (
                    float(total_ts_corrected / total_ts_with_value)
                    if total_ts_with_value > 0
                    else 0.0
                ),
            },
        }
        save_json(summary_path, summary_payload)

        print_sample_slot_sanity(slot_logs, max_samples=5)
        print(f"      [LOG] per-slot  -> {per_slot_path}")
        print(f"      [LOG] summary   -> {summary_path}")
        print()

    print("=== Summary ===")
    for capacity, stats in zip(capacities_objects, results):
        print(
            f"cache_size={stats.capacity_objects}, "
            f"hit_ratio={stats.hit_ratio:.4f} "
            f"({stats.cache_hits}/{stats.total_requests})"
        )


def main() -> None:
    run_experiment(WIKIPEDIA_SEPTEMBER_2007)


if __name__ == "__main__":
    main()
# 822
