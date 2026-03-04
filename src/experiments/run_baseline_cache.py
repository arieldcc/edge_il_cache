# src/experiments/run_baseline_cache.py

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, List, Tuple
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config.experiment_config import (
    WIKIPEDIA_SEPTEMBER_2007,   # WIKIPEDIA_OKTOBER_2007/WIKIPEDIA_SEPTEMBER_2007/WIKI2018
    CacheConfig,
)
from src.data.trace_reader import TraceReader
from src.cache.cache_simulator import CacheStats
from src.cache.lru import LRUCache
from src.cache.lfuda import LFUDACache
from src.cache.lru2 import LRU2Cache
from src.cache.tinylfu import TinyLFUCache


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


def make_cache(policy: str, capacity_objects: int):
    policy = policy.upper()
    if policy == "LRU":
        return LRUCache(capacity_objects)
    elif policy in ("LRU2", "LRU-K", "LRU-K2"):
        return LRU2Cache(capacity_objects)
    elif policy == "LFUDA":
        return LFUDACache(capacity_objects)
    elif policy == "TINYLFU":
        return TinyLFUCache(capacity_objects)
    else:
        raise ValueError(f"Unknown policy: {policy}")


def run_single_policy(
    policy: str,
    capacity_objects: int,
) -> CacheStats:
    """
    Baseline cache (LRU / LRU2 / LFUDA) pada raw Wikipedia trace.

    - Dataset & parameter eksperimen diambil dari experiment_config:
        * ds_cfg.path            -> file .gz asli
        * ds_cfg.num_total_requests
        * ds_cfg.num_warmup_requests
    """
    ds_cfg = WIKIPEDIA_SEPTEMBER_2007   # ganti manual ke WIKI2018 kalau mau Oktober

    total_requests = ds_cfg.num_total_requests
    warmup_requests = ds_cfg.num_warmup_requests

    # TraceReader generik: path bisa .gz atau direktori parquet
    reader = TraceReader(path=ds_cfg.path, max_rows=total_requests)
    req_iter = reader.iter_requests()

    cache = make_cache(policy, capacity_objects)
    stats = CacheStats(capacity_objects=capacity_objects)

    global_idx = 0

    # Hanya request setelah warm-up yang dihitung ke hit ratio
    eval_requests_planned = max(total_requests - warmup_requests, 0)
    pbar = tqdm(
        total=eval_requests_planned,
        desc=f"Eval {policy} (cap={capacity_objects})",
        unit="req",
    )

    for req in req_iter:
        if global_idx >= total_requests:
            break

        obj_id = req["object_id"]

        if global_idx >= warmup_requests:
            stats.total_requests += 1
            if cache.access(obj_id):
                stats.cache_hits += 1
            else:
                # admit-all baseline:
                # - LRU & LFUDA: insert langsung
                # - LRU2: admission rule di-handle dalam insert()
                cache.insert(obj_id)
            # progress bar hanya untuk fase eval
            if stats.total_requests <= eval_requests_planned:
                pbar.update(1)

        global_idx += 1

    pbar.close()
    return stats
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

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run baseline cache policies (LRU, LRU-2, LFUDA) on raw Wikipedia 2007 trace."
    )
    parser.add_argument(
        "--policy",
        type=str,
        required=True,
        choices=["LRU", "LRU2", "LRU-K", "LRU-K2", "LFUDA", "TINYLFU"],
        help="Cache replacement policy.",
    )

    args = parser.parse_args()
    policy = args.policy.upper()

    # Ambil konfigurasi dataset & cache dari experiment_config
    ds_cfg = WIKIPEDIA_SEPTEMBER_2007   # ganti manual ke WIKI2018 jika ingin trace Oktober
    cache_cfg = CacheConfig()
    trace_path = ds_cfg.path

    total_requests = ds_cfg.num_total_requests
    warmup_requests = ds_cfg.num_warmup_requests

    dataset_name = ds_cfg.name
    model_name = f"baseline_{policy.lower()}"
    results_root = "results"

    print(f"=== Baseline Cache Experiment ({dataset_name} raw trace) ===")
    print(f"Dataset          : {dataset_name}")
    print(f"Trace path       : {ds_cfg.path}")
    print(f"Policy           : {policy}")
    print(f"Total requests   : {total_requests}")
    print(f"Warm-up requests : {warmup_requests}")
    print(f"Cache sizes      : {list(cache_cfg.cache_size_percentages)}")
    print()

    # Hitung kapasitas dinamis
    capacities_objects = get_dynamic_capacities(trace_path, total_requests, cache_cfg.cache_size_percentages)

    results: List[CacheStats] = []

    for capacity in capacities_objects:
        print(f"[RUN] Policy={policy}, Capacity={capacity} objects")

        stats = run_single_policy(
            policy=policy,
            capacity_objects=capacity,
        )
        results.append(stats)

        print(
            f"      Hit ratio={stats.hit_ratio:.4f} "
            f"({stats.cache_hits}/{stats.total_requests})"
        )

        # Simpan summary ke JSON
        run_id, dataset_dir = get_next_run_id(results_root, dataset_name, model_name, capacity)
        summary_path = os.path.join(
            dataset_dir,
            f"{run_id}_summary_{model_name}_{capacity}.json",
        )
        payload = {
            "dataset": dataset_name,
            "policy": policy,
            "cache_size_objects": capacity,
            "total_requests_config": total_requests,
            "warmup_requests_config": warmup_requests,
            "cache_hits": stats.cache_hits,
            "cache_requests": stats.total_requests,   # hanya fase eval
            "hit_ratio": stats.hit_ratio,
        }
        save_json(summary_path, payload)
        print(f"      [LOG] summary -> {summary_path}")
        print()

    print("=== Summary ===")
    for capacity, stats in zip(capacities_objects, results):
        print(
            f"policy={policy}, cache_size={capacity}, "
            f"hit_ratio={stats.hit_ratio:.4f} "
            f"({stats.cache_hits}/{stats.total_requests})"
        )


if __name__ == "__main__":
    main()
