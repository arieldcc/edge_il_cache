# src/experiments/run_il_cache_xu.py

from __future__ import annotations

import json
import os
import sys
from typing import Dict, List, Tuple, Any, Optional

from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config.experiment_config import (
    WIKIPEDIA_SEPTEMBER_2007,  # WIKIPEDIA_OKTOBER_2007/WIKIPEDIA_SEPTEMBER_2007/WIKI2018
    DatasetConfig,
    ILConfig,
    CacheConfig,
)
from src.data.trace_reader import TraceReader
from src.data.feature_table import FeatureTable
from src.ml.learn_nse import LearnNSE
from src.cache.cache_simulator import CacheStats
from src.cache.lru import LRUCache


def build_slot_dataset_from_stats(
    slot_stats: Dict[str, Dict],
    top_ratio: float,
    num_features: int,
    missing_gap_value: float,
) -> List[Dict]:
    if not slot_stats:
        return []

    items: List[Tuple[str, Dict]] = list(slot_stats.items())
    items.sort(key=lambda kv: kv[1]["freq"], reverse=True)

    n_objects = len(items)
    k = int(n_objects * top_ratio)
    if k <= 0:
        k = 1
    if k > n_objects:
        k = n_objects

    top_ids = {items[i][0] for i in range(k)}

    dataset: List[Dict] = []
    for obj_id, stats in items:
        gaps = stats["last_gaps"]
        if len(gaps) != num_features:
            if len(gaps) < num_features:
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

    return dataset


def run_single_capacity(
    trace_path: str,
    total_requests: int,
    warmup_requests: int,
    slot_size: int,
    capacity_objects: int,
    il_cfg: ILConfig,
) -> Tuple[CacheStats, List[Dict[str, Any]]]:
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

    slot_logs: List[Dict[str, Any]] = []

    total_slots_expected = (total_requests + slot_size - 1) // slot_size
    pbar = tqdm(total=total_slots_expected, desc="Processing slots", unit="slot")

    global_idx = 0
    slot_req_count = 0
    slot_index = 0
    slot_stats: Dict[str, Dict] = {}
    prev_ts: Optional[float] = None

    while True:
        try:
            req = next(req_iter)
        except StopIteration:
            if slot_stats:
                end_req = min(slot_index * slot_size + slot_req_count, total_requests) - 1
                D_t = build_slot_dataset_from_stats(
                    slot_stats,
                    il_cfg.pop_top_percent,
                    il_cfg.num_gaps,
                    il_cfg.missing_gap_value,
                )
                slot_index += 1
                il_model.update_slot(D_t)
                phase = "warmup" if end_req < warmup_requests else "cache"
                slot_logs.append(
                    {
                        "slot_index": slot_index,
                        "phase": phase,
                        "slot_size_requests": slot_req_count,
                        "slot_start_request_idx": (slot_index - 1) * slot_size,
                        "slot_end_request_idx": end_req,
                        "stats_total_requests": stats.total_requests,
                        "stats_cache_hits": stats.cache_hits,
                        "hit_ratio": stats.hit_ratio,
                    }
                )
                pbar.update(1)
                pbar.set_postfix(
                    hr=f"{stats.hit_ratio:.4f}" if stats.total_requests > 0 else "0.0000"
                )
            break

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

        if global_idx >= warmup_requests:
            stats.total_requests += 1
            if cache.access(obj_id):
                stats.cache_hits += 1
            else:
                y_hat = il_model.predict(gaps)
                if y_hat == 1:
                    cache.insert(obj_id)

        global_idx += 1
        slot_req_count += 1

        if slot_req_count >= slot_size:
            D_t = build_slot_dataset_from_stats(
                slot_stats,
                il_cfg.pop_top_percent,
                il_cfg.num_gaps,
                il_cfg.missing_gap_value,
            )
            slot_index += 1
            il_model.update_slot(D_t)

            start_req = (slot_index - 1) * slot_size
            end_req = start_req + slot_req_count - 1
            phase = "warmup" if end_req < warmup_requests else "cache"
            slot_logs.append(
                {
                    "slot_index": slot_index,
                    "phase": phase,
                    "slot_size_requests": slot_req_count,
                    "slot_start_request_idx": start_req,
                    "slot_end_request_idx": end_req,
                    "stats_total_requests": stats.total_requests,
                    "stats_cache_hits": stats.cache_hits,
                    "hit_ratio": stats.hit_ratio,
                }
            )

            pbar.update(1)
            current_hr = stats.hit_ratio if stats.total_requests > 0 else 0.0
            pbar.set_postfix(hr=f"{current_hr:.4f}")

            slot_stats = {}
            slot_req_count = 0

    pbar.close()
    return stats, slot_logs


def count_distinct_objects(trace_path: str, total_requests: int) -> int:
    reader = TraceReader(path=trace_path, max_rows=total_requests)
    req_iter = reader.iter_requests()

    unique_objects = set()
    count = 0

    for req in req_iter:
        unique_objects.add(req["object_id"])
        count += 1
        if count >= total_requests:
            break

    return len(unique_objects)


def get_dynamic_capacities(trace_path: str, total_requests: int, percentages: List[float]) -> List[int]:
    n_unique = count_distinct_objects(trace_path, total_requests)
    capacities = [int(n_unique * p / 100) for p in percentages]
    print(f"Jumlah objek unik: {n_unique}")
    print(f"Kapasitas cache dinamis (berdasarkan %): {capacities}")
    return capacities


def get_next_run_id(results_root: str, dataset: str, model: str, cache_size: int) -> Tuple[str, str]:
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


def run_experiment(
    ds_cfg: DatasetConfig,
    il_cfg: Optional[ILConfig] = None,
    cache_cfg: Optional[CacheConfig] = None,
    model_name: str = "ilnse_xu",
) -> None:
    if il_cfg is None:
        il_cfg = ILConfig()
    if cache_cfg is None:
        cache_cfg = CacheConfig()

    trace_path = ds_cfg.path
    slot_size = ds_cfg.slot_size
    warmup_requests = ds_cfg.num_warmup_requests
    total_requests = ds_cfg.num_total_requests

    if warmup_requests % slot_size != 0:
        raise ValueError(
            f"warmup_requests ({warmup_requests}) harus kelipatan slot_size ({slot_size})."
        )

    dataset_name = ds_cfg.name
    results_root = "results"

    print(f"=== IL-based Edge Cache Experiment (Xu baseline) {dataset_name} ===")
    print(f"Trace path        : {trace_path}")
    print(f"Total requests    : {total_requests}")
    print(f"Warm-up requests  : {warmup_requests}")
    print(f"Slot size         : {slot_size}")
    print(f"IL num_gaps       : {il_cfg.num_gaps}")
    print(f"IL top_percent    : {il_cfg.pop_top_percent}")
    print(f"IL sigmoid (a, b) : ({il_cfg.sigmoid_a}, {il_cfg.sigmoid_b})")
    print(f"Max learners      : {il_cfg.max_classifiers}")
    print(f"Cache capacities  : {list(cache_cfg.cache_size_percentages)}")
    print()

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
        }
        save_json(summary_path, summary_payload)

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
