# src/experiments/run_gdbt_cache.py

"""
Eksperimen GDBT-based Edge Caching (replikasi baseline Xu et al.)

GDBT (Gradient Boosted Decision Trees) di sini diimplementasikan sebagai:
- Baseline ML offline / batch.
- Fitur sama dengan IL: Gap1..GapL (default L = 6).
- Label: 1 untuk objek yang termasuk top pop_top_percent (default 20%) populer
  di setiap time slot (100K request), 0 untuk lainnya.
- Model di-rebuild dari awal setiap update_interval_requests (default 1M) request.
- Jumlah tree (iterations) = n_estimators (default 30).

Underlying cache:
- LRU sebagai replacement policy.
- GDBT hanya memutuskan admission pada cache miss.

Struktur eksperimen:
- Warm-up phase:
    - num_warmup_requests pertama (default 1M) digunakan untuk mengumpulkan
      fitur dan label, melatih model GDBT pertama.
    - Pada fase ini, TIDAK ada caching (cache tidak diakses).
- Evaluation phase:
    - Sisa total_requests - num_warmup_requests dipakai untuk evaluasi.
    - Cache aktif dengan GDBT-based admission + LRU eviction.
    - GDBT di-update setiap update_interval_requests (1M) request evaluasi.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Tuple

import numpy as np
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Setup path
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config.experiment_config import (  # type: ignore
    WIKIPEDIA_OKTOBER_2007,
    CacheConfig,
    ILConfig,
    GDBTConfig,
)
from src.data.trace_reader import TraceReader  # type: ignore
from src.data.feature_table import FeatureTable  # type: ignore
from src.ml.gdbt_model_xu import GDBTCachePredictor  # type: ignore
from src.cache.cache_simulator import CacheStats  # type: ignore
from src.cache.lru import LRUCache  # type: ignore


# ---------------------------------------------------------------------------
# Membangun dataset D_t per slot (100K request)
# ---------------------------------------------------------------------------

def build_slot_dataset_from_stats(
    slot_stats: Dict[str, Dict[str, Any]],
    top_ratio: float,
    num_features: int,
    missing_gap_value: float,
) -> List[Dict[str, Any]]:
    """
    Membangun dataset D_t untuk training dari statistik per-objek dalam satu slot.

    slot_stats:
        object_id -> {
            "freq": int  (jumlah request objek ini dalam slot),
            "last_gaps": List[float] (Gap1..GapL terakhir)
        }

    Prosedur:
    - Urutkan objek dalam slot berdasarkan 'freq' (descending).
    - Ambil top (top_ratio * n_objects) sebagai label 1.
    - Sisanya label 0.
    - Fitur x = last_gaps yang sudah dipotong / dipad jadi panjang num_features.
    """
    if not slot_stats:
        return []

    # Urutkan berdasarkan frekuensi dalam slot (descending)
    items: List[Tuple[str, Dict[str, Any]]] = list(slot_stats.items())
    items.sort(key=lambda kv: kv[1]["freq"], reverse=True)

    n_objects = len(items)
    k = max(1, min(int(n_objects * top_ratio), n_objects))

    top_ids = {items[i][0] for i in range(k)}

    dataset: List[Dict[str, Any]] = []
    for obj_id, stats in items:
        gaps = stats["last_gaps"]

        # Normalisasi panjang fitur: harus num_features
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


# ---------------------------------------------------------------------------
# Pre-scan: hitung jumlah objek unik dan kapasitas cache
# ---------------------------------------------------------------------------

def count_distinct_objects(trace_path: str, total_requests: int) -> int:
    """
    Hitung jumlah object_id unik pada prefix trace sepanjang total_requests.
    """
    reader = TraceReader(path=trace_path, max_rows=total_requests)
    unique_objects = set()
    count = 0

    for req in reader.iter_requests():
        unique_objects.add(req["object_id"])
        count += 1
        if count >= total_requests:
            break

    return len(unique_objects)


def get_dynamic_capacities(
    trace_path: str,
    total_requests: int,
    percentages: List[float],
) -> List[int]:
    """
    Hitung kapasitas cache berdasarkan persentase objek unik.

    percentages diekspresikan dalam persen, mis. 0.8, 1.0, ..., 5.0.
    """
    n_unique = count_distinct_objects(trace_path, total_requests)
    capacities = [max(1, int(n_unique * p / 100.0)) for p in percentages]

    print(f"Jumlah objek unik (approx): {n_unique}")
    print(f"Kapasitas cache (objek)   : {capacities}")

    return capacities


# ---------------------------------------------------------------------------
# Satu run GDBT+LRU untuk satu kapasitas cache
# ---------------------------------------------------------------------------

def run_single_capacity(
    trace_path: str,
    total_requests: int,
    warmup_requests: int,
    slot_size: int,
    capacity_objects: int,
    il_cfg: ILConfig,
    gdbt_cfg: GDBTConfig,
) -> Tuple[CacheStats, List[Dict[str, Any]]]:
    """
    Menjalankan eksperimen GDBT+LRU untuk satu kapasitas cache.

    Workflow (disederhanakan sesuai artikel Xu et al.):

    - Warm-up phase:
        * request ke-0 .. ke-(warmup_requests-1)
        * Tidak ada caching
        * Slot demi slot (100K) dijadikan dataset D_t dan ditambahkan ke buffer GDBT
        * Di akhir warm-up, model GDBT dilatih untuk pertama kali

    - Evaluation phase:
        * request ke-warmup_requests .. ke-(total_requests-1)
        * Cache aktif dengan:
            - LRU sebagai eviction
            - GDBT sebagai admission decision
        * Fitur Gap1..GapL diambil dari FeatureTable
        * GDBT di-update setiap update_interval_requests (1M) request evaluasi
          (dibulatkan ke slot: update setiap K_slot = update_interval / slot_size)
    """
    # Reader trace
    reader = TraceReader(path=trace_path, max_rows=total_requests)
    req_iter = reader.iter_requests()

    # Feature table untuk Gap1..GapL
    feature_table = FeatureTable(
        L=il_cfg.num_gaps,
        missing_gap_value=il_cfg.missing_gap_value,
    )

    # GDBT predictor (menggunakan config IL + GDBT)
    gdbt_model = GDBTCachePredictor(il_config=il_cfg, gdbt_config=gdbt_cfg)

    # Cache dengan LRU replacement
    cache = LRUCache(capacity_objects=capacity_objects)
    stats = CacheStats(capacity_objects=capacity_objects)

    # Logging rebuild
    rebuild_logs: List[Dict[str, Any]] = []

    # Counters global
    global_idx = 0  # index request
    slot_req_count = 0  # jumlah request dalam slot berjalan
    slot_index = 0  # indeks slot (0-based)
    slot_stats: Dict[str, Dict[str, Any]] = {}

    # Untuk interval update GDBT di evaluation phase
    eval_requests_planned = max(total_requests - warmup_requests, 0)
    pbar = tqdm(
        total=eval_requests_planned,
        desc=f"GDBT Eval (cap={capacity_objects})",
        unit="req",
    )

    # Slot yang termasuk dalam warm-up
    warmup_slots = warmup_requests // slot_size if slot_size > 0 else 0
    # Berapa slot per satu interval update GDBT
    slots_per_rebuild = (
        gdbt_cfg.update_interval_requests // slot_size if slot_size > 0 else 1
    )
    slots_since_rebuild = 0  # hanya dihitung di evaluation phase

    while True:
        try:
            req = next(req_iter)
        except StopIteration:
            # Akhir trace: flush slot terakhir ke buffer GDBT
            if slot_stats:
                D_t = build_slot_dataset_from_stats(
                    slot_stats,
                    il_cfg.pop_top_percent,
                    il_cfg.num_gaps,
                    il_cfg.missing_gap_value,
                )
                gdbt_model.add_training_batch(D_t)
            break

        if global_idx >= total_requests:
            break

        obj_id = req["object_id"]
        ts = float(global_idx + 1)  # timestamp sintetis monoton

        # mmengabil timestamps dari dataset:
        ts_raw = req.get("timestamp", None)

        # ---------------------------------------------------------------------
        # 1) Update fitur (Gap1..GapL) via FeatureTable
        # ---------------------------------------------------------------------
        gaps = feature_table.update_and_get_gaps(obj_id, ts_raw)

        # Update statistik per-objek di slot
        info = slot_stats.get(obj_id)
        if info is None:
            slot_stats[obj_id] = {
                "freq": 1,
                "last_gaps": gaps,
            }
        else:
            info["freq"] += 1
            info["last_gaps"] = gaps

        # ---------------------------------------------------------------------
        # 2) Fase caching (hanya setelah warm-up)
        # ---------------------------------------------------------------------
        if global_idx >= warmup_requests:
            # Evaluation phase
            stats.total_requests += 1

            if cache.access(obj_id):
                stats.cache_hits += 1
            else:
                # Cache miss → tanya GDBT
                y_hat = gdbt_model.predict(gaps)
                if y_hat == 1:
                    cache.insert(obj_id)

            pbar.update(1)

        # ---------------------------------------------------------------------
        # 3) Boundary slot: saat slot_req_count mencapai slot_size
        # ---------------------------------------------------------------------
        slot_req_count += 1
        if slot_req_count >= slot_size:
            # Bangun dataset D_t dari slot ini lalu tambahkan ke buffer GDBT
            D_t = build_slot_dataset_from_stats(
                slot_stats,
                il_cfg.pop_top_percent,
                il_cfg.num_gaps,
                il_cfg.missing_gap_value,
            )
            gdbt_model.add_training_batch(D_t)

            slot_index += 1
            slot_stats = {}
            slot_req_count = 0

            # --- Setelah selesai satu slot ---
            if slot_index == warmup_slots:
                # Akhir warm-up phase → train model pertama
                rebuild_info = gdbt_model.rebuild_model()
                rebuild_info["phase"] = "warmup_complete"
                rebuild_info["global_request_idx"] = global_idx
                rebuild_logs.append(rebuild_info)

                gdbt_model.clear_buffer()
                slots_since_rebuild = 0

            elif slot_index > warmup_slots and global_idx >= warmup_requests:
                # Evaluation phase:
                # Tambah jumlah slot sejak rebuild terakhir
                slots_since_rebuild += 1

                if slots_since_rebuild >= max(1, slots_per_rebuild):
                    # Saatnya rebuild GDBT menggunakan data dari slot-slot
                    # setelah rebuild sebelumnya.
                    rebuild_info = gdbt_model.rebuild_model()
                    rebuild_info["phase"] = "eval_rebuild"
                    rebuild_info["global_request_idx"] = global_idx
                    rebuild_info["cache_hit_ratio_so_far"] = stats.hit_ratio
                    rebuild_logs.append(rebuild_info)

                    gdbt_model.clear_buffer()
                    slots_since_rebuild = 0

                    pbar.set_postfix(
                        hr=f"{stats.hit_ratio:.4f}",
                        rebuilds=gdbt_model.get_stats()["num_rebuilds"],
                    )

        # Update global index
        global_idx += 1

    pbar.close()

    # Final log
    final_stats = gdbt_model.get_stats()
    fi = gdbt_model.get_feature_importances()
    final_log = {
        "final_model_stats": final_stats,
        "feature_importances": fi.tolist() if fi is not None else None,
    }
    rebuild_logs.append(final_log)

    return stats, rebuild_logs


# ---------------------------------------------------------------------------
# Utilitas untuk menyimpan hasil
# ---------------------------------------------------------------------------

def get_next_run_id(
    results_root: str,
    dataset: str,
    model: str,
    cache_size: int,
) -> Tuple[str, str]:
    """
    Menentukan ID run berikutnya untuk kombinasi dataset + model + cache_size.
    """
    dataset_dir = os.path.join(results_root, dataset)
    os.makedirs(dataset_dir, exist_ok=True)

    existing_ids: List[int] = []
    for fname in os.listdir(dataset_dir):
        if not (
            fname.endswith(f"_{model}_{cache_size}.json")
            or fname.endswith(f"_summary_{model}_{cache_size}.json")
        ):
            continue
        prefix = fname.split("_", 1)[0]
        if len(prefix) == 3 and prefix.isdigit():
            existing_ids.append(int(prefix))

    next_id = 1 if not existing_ids else max(existing_ids) + 1
    return f"{next_id:03d}", dataset_dir


def save_json(path: str, data: Any) -> None:
    """Simpan data ke file JSON dengan indent rapi."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    # --- Konfigurasi dataset dan model ---
    ds_cfg = WIKIPEDIA_OKTOBER_2007
    cache_cfg = CacheConfig()
    il_cfg = ILConfig()
    gdbt_cfg = GDBTConfig()

    trace_path = ds_cfg.path
    slot_size = ds_cfg.slot_size
    warmup_requests = ds_cfg.num_warmup_requests
    total_requests = ds_cfg.num_total_requests

    dataset_name = ds_cfg.name
    model_name = "gdbt"
    results_root = "results"

    print(f"=== GDBT-based Edge Cache Experiment (Xu et al. baseline) {dataset_name} ===")
    print(f"Trace path          : {trace_path}")
    print(f"Total requests      : {total_requests}")
    print(f"Warm-up requests    : {warmup_requests}")
    print(f"Slot size           : {slot_size}")
    print(f"IL num_gaps         : {il_cfg.num_gaps}")
    print(f"IL top percent      : {il_cfg.pop_top_percent}")
    print(f"GDBT n_estimators   : {gdbt_cfg.n_estimators}")
    print(f"GDBT update_interval: {gdbt_cfg.update_interval_requests}")
    print(f"Cache size (%)      : {list(cache_cfg.cache_size_percentages)}")
    print()

    # Kapasitas cache dinamis berdasarkan persentase distinct objects
    capacities_objects = get_dynamic_capacities(
        trace_path,
        total_requests,
        list(cache_cfg.cache_size_percentages),
    )

    results: List[CacheStats] = []

    for capacity in capacities_objects:
        print(f"\n[RUN] GDBT+LRU, Capacity={capacity} objects")

        stats, rebuild_logs = run_single_capacity(
            trace_path=trace_path,
            total_requests=total_requests,
            warmup_requests=warmup_requests,
            slot_size=slot_size,
            capacity_objects=capacity,
            il_cfg=il_cfg,
            gdbt_cfg=gdbt_cfg,
        )
        results.append(stats)

        print(
            f"      Hit ratio = {stats.hit_ratio:.4f} "
            f"({stats.cache_hits}/{stats.total_requests})"
        )

        # Simpan hasil
        run_id, dataset_dir = get_next_run_id(
            results_root, dataset_name, model_name, capacity
        )

        # Per-rebuild log
        rebuild_path = os.path.join(
            dataset_dir,
            f"{run_id}_{model_name}_{capacity}_rebuilds.json",
        )
        save_json(
            rebuild_path,
            {
                "dataset": dataset_name,
                "model": model_name,
                "cache_size_objects": capacity,
                "rebuilds": rebuild_logs,
            },
        )

        # Summary
        summary_path = os.path.join(
            dataset_dir,
            f"{run_id}_summary_{model_name}_{capacity}.json",
        )
        summary_payload = {
            "dataset": dataset_name,
            "model": model_name,
            "cache_size_objects": capacity,
            "total_requests": total_requests,
            "warmup_requests": warmup_requests,
            "slot_size": slot_size,
            "gdbt_n_estimators": gdbt_cfg.n_estimators,
            "gdbt_update_interval": gdbt_cfg.update_interval_requests,
            "pop_top_percent": il_cfg.pop_top_percent,
            "hit_ratio": stats.hit_ratio,
            "cache_hits": stats.cache_hits,
            "cache_requests": stats.total_requests,
            "num_rebuilds": gdbt_cfg.update_interval_requests
            and rebuild_logs[-1]["final_model_stats"]["num_rebuilds"],
        }
        save_json(summary_path, summary_payload)

        print(f"      [LOG] rebuilds -> {rebuild_path}")
        print(f"      [LOG] summary  -> {summary_path}")

    print("\n=== Summary ===")
    for capacity, stats in zip(capacities_objects, results):
        print(
            f"GDBT+LRU, cache_size={capacity}, "
            f"hit_ratio={stats.hit_ratio:.4f} "
            f"({stats.cache_hits}/{stats.total_requests})"
        )


if __name__ == "__main__":
    main()
# 518