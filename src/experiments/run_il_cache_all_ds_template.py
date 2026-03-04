# src/experiments/run_il_cache_all_ds_template.py

from __future__ import annotations

import json
import itertools
import math
import os
import sys
import time
from typing import Dict, List, Tuple, Any, Optional
import numpy as np

from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.trace_reader import TraceReader
from src.data.feature_table import FeatureTable
from src.ml.learn_nse_all_ds import LearnNSE
from src.cache.cache_simulator import CacheStats
from src.cache.lru import LRUCache


# ---------------------------------------------------------------------------
# Utilitas untuk membangun D_t dari statistik slot
# ---------------------------------------------------------------------------

def compute_freq_feature(freq: int, slot_size: int, mode: str) -> float:
    freq = max(int(freq), 0)
    if mode == "log_norm":
        denom = math.log1p(max(int(slot_size), 1))
        return (math.log1p(freq) / denom) if denom > 0.0 else 0.0
    if mode == "log":
        return math.log1p(freq)
    raise ValueError(f"Unknown freq_feature_mode: {mode}")

def build_features_from_gaps(
    gaps: List[float],
    freq: int,
    num_gaps: int,
    use_freq_feature: bool,
    freq_feature_mode: str,
    slot_size: int,
    missing_gap_value: float,
) -> List[float]:
    g = list(gaps)
    if len(g) < num_gaps:
        g = g + [missing_gap_value] * (num_gaps - len(g))
    else:
        g = g[:num_gaps]
    if use_freq_feature:
        g.append(compute_freq_feature(freq, slot_size, freq_feature_mode))
    return g

def build_slot_dataset_from_stats(
    slot_stats: Dict[str, Dict],
    top_ratio: float,
    num_gaps: int,
    missing_gap_value: float,
    label_mode: str,
    capacity_objects: int,
    label_capacity_gamma: float,
    use_freq_feature: bool,
    freq_feature_mode: str,
    freq_feature_source: str,
    prev_slot_freq: Dict[str, int],
    slot_size: int,
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
        "label_mode": label_mode,
        "pos_ratio": float("nan"),
        "coverage_actual": float("nan"),
        "capacity_k": None,
        "n_objects": 0,
        "n_pos": 0,
    }
    if not slot_stats:
        return [], label_info

    items: List[Tuple[str, Dict]] = list(slot_stats.items())
    # urutkan berdasarkan freq menurun (popularitas per-slot)
    items.sort(key=lambda kv: kv[1]["freq"], reverse=True)

    n_objects = len(items)
    label_info["n_objects"] = int(n_objects)
    total_reqs = sum(int(stats["freq"]) for _, stats in items)
    top_ids: set = set()

    if label_mode == "capacity":
        k = int(round(float(capacity_objects) * float(label_capacity_gamma)))
        k = max(1, min(k, n_objects))
        label_info["capacity_k"] = int(k)
        top_ids = {items[i][0] for i in range(k)}
    elif label_mode == "top_ratio":
        k = int(n_objects * top_ratio)
        k = max(1, min(k, n_objects))
        top_ids = {items[i][0] for i in range(k)}
    else:
        raise ValueError(f"Unknown label_mode: {label_mode}")

    label_info["n_pos"] = int(len(top_ids))
    if n_objects > 0:
        label_info["pos_ratio"] = float(len(top_ids) / float(n_objects))
    if total_reqs > 0:
        label_info["coverage_actual"] = float(
            sum(int(slot_stats[obj]["freq"]) for obj in top_ids) / float(total_reqs)
        )

    dataset: List[Dict] = []
    for obj_id, stats in items:
        gaps = stats["last_gaps"]
        freq = int(stats["freq"])
        if freq_feature_source == "prev_slot":
            freq_val = int(prev_slot_freq.get(obj_id, 0))
        elif freq_feature_source == "slot_total":
            freq_val = freq
        else:
            raise ValueError(f"Unknown freq_feature_source: {freq_feature_source}")

        features = build_features_from_gaps(
            gaps=gaps,
            freq=freq_val,
            num_gaps=num_gaps,
            use_freq_feature=use_freq_feature,
            freq_feature_mode=freq_feature_mode,
            slot_size=slot_size,
            missing_gap_value=missing_gap_value,
        )

        y = 1 if obj_id in top_ids else 0
        dataset.append(
            {
                "x": features,
                "y": y,
                "freq": freq,
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

def compute_admit_tau(
    D_t: List[Dict],
    il_model: LearnNSE,
    target_pos_rate: float,
) -> Tuple[Optional[float], Dict[str, Any]]:
    """
    Hitung threshold skor (tau) agar proporsi positif ~ target_pos_rate.
    Tau dihitung dari skor model saat ini pada D_t (post-update).
    """
    if not D_t or target_pos_rate <= 0.0:
        return None, {}

    X = np.asarray([row["x"] for row in D_t], dtype=float)
    scores = il_model.score_batch(X)
    n = len(scores)
    if n == 0:
        return None, {}

    k = int(round(target_pos_rate * n))
    k = max(1, min(k, n))
    order = np.argsort(-scores)
    tau = float(scores[order[k - 1]])

    stats = {
        "score_p50": float(np.percentile(scores, 50)),
        "score_p90": float(np.percentile(scores, 90)),
        "score_p99": float(np.percentile(scores, 99)),
        "score_max": float(np.max(scores)),
    }
    return tau, stats

def reservoir_update(
    reservoir: List[float],
    seen: int,
    x: float,
    max_size: int,
    rng: np.random.Generator,
) -> int:
    seen += 1
    if max_size <= 0:
        return seen
    if len(reservoir) < max_size:
        reservoir.append(float(x))
    else:
        j = int(rng.integers(0, seen))
        if j < max_size:
            reservoir[j] = float(x)
    return seen

def compute_tau_from_scores(
    scores: List[float],
    target_pos_rate: float,
) -> Optional[float]:
    if not scores or target_pos_rate <= 0.0:
        return None
    arr = np.asarray(scores, dtype=float)
    n = arr.size
    if n == 0:
        return None
    k = int(round(target_pos_rate * n))
    k = max(1, min(k, n))
    order = np.argsort(-arr)
    return float(arr[order[k - 1]])

def sample_lru_tail(cache: LRUCache, k: int) -> List[str]:
    if k <= 0:
        return []
    data = getattr(cache, "_data", None)
    if not data:
        return []
    return list(itertools.islice(data.keys(), k))

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
    - Learn++.NSE di-update setiap akhir slot sesuai label_mode di config.
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
    num_gaps = int(il_cfg.num_gaps)
    n_features = num_gaps + (1 if il_cfg.use_freq_feature else 0)
    il_model = LearnNSE(
        n_features=n_features,
        a=il_cfg.sigmoid_a,
        b=il_cfg.sigmoid_b,
        max_learners=il_cfg.max_classifiers,
    )
    cache = LRUCache(capacity_objects=capacity_objects)
    stats = CacheStats(capacity_objects=capacity_objects)

    slot_logs: List[Dict[str, Any]] = []

    total_slots_expected = (total_requests + slot_size - 1) // slot_size
    pbar = tqdm(total=total_slots_expected, desc="Processing slots", unit="slot")

    global_idx = 0          # counter global request (0-based)
    slot_req_count = 0      # jumlah request dalam slot saat ini
    slot_index = 0          # 1-based untuk logging
    slot_stats: Dict[str, Dict] = {}  # per-slot: freq & last_gaps
    prev_ts: Optional[float] = None
    prev_slot_freq: Dict[str, int] = {}

    admit_tau_current: Optional[float] = None
    admit_tau_ema_alpha = float(il_cfg.admit_tau_ema_alpha) if il_cfg.admit_use_tau else 0.0
    freq_feature_source = str(il_cfg.freq_feature_source)
    gate_min_hist_len = 0
    warmup_prefill = True
    tau_use_candidate_reservoir = bool(getattr(il_cfg, "tau_use_candidate_reservoir", True))
    tau_score_sample_size = int(getattr(il_cfg, "tau_score_sample_size", 4096))
    tau_reservoir_min_samples = int(getattr(il_cfg, "tau_reservoir_min_samples", 512))
    tau_reservoir_fallback = str(getattr(il_cfg, "tau_reservoir_fallback", "ema"))
    tau_capacity_gamma = float(getattr(il_cfg, "tau_capacity_gamma", 1.0))
    tau_pos_rate_min = float(getattr(il_cfg, "tau_pos_rate_min", 0.002))
    tau_pos_rate_max = float(getattr(il_cfg, "tau_pos_rate_max", 0.30))
    rng = np.random.default_rng(int(getattr(il_cfg, "seed", 42)))
    slot_log_mode = str(getattr(il_cfg, "slot_log_mode", "light"))
    slot_log_sample_k = int(getattr(il_cfg, "slot_log_sample_k", 0))
    victim_sample_k = int(getattr(il_cfg, "victim_sample_k", 0))
    victim_sample_mode = "tail"
    cache_score_refresh_k = int(getattr(il_cfg, "cache_score_refresh_k", 0))
    cache_score_refresh_mode = "tail"
    cache_score_stale_slots = int(getattr(il_cfg, "cache_score_stale_slots", 1))

    tau_capacity_gamma_current = float(tau_capacity_gamma)
    cap_ratio_current = 0.0
    cache_score_map: Dict[str, float] = {}
    cache_score_slot: Dict[str, int] = {}
    cand_score_reservoir: List[float] = []
    cand_score_seen = 0

    def _compact_slot_info(slot_info: Dict[str, Any]) -> Dict[str, Any]:
        if slot_log_mode == "full":
            return slot_info
        out: Dict[str, Any] = {}
        for key in (
            "E_t",
            "acc_before",
            "acc_after",
            "num_learners_before",
            "num_learners_after",
            "new_learner_id",
            "prune",
        ):
            if key in slot_info:
                out[key] = slot_info[key]
        weights = slot_info.get("weights")
        if isinstance(weights, list) and weights:
            w = np.asarray(weights, dtype=float)
            out["weights_summary"] = {
                "mean": float(np.mean(w)),
                "std": float(np.std(w)),
                "min": float(np.min(w)),
                "max": float(np.max(w)),
            }
        return out

    def _score_with_role(features: List[float], role: str) -> float:
        nonlocal slot_admit_score_time_sec
        nonlocal slot_score_candidate_calls, slot_score_candidate_time_sec
        nonlocal slot_score_victim_calls, slot_score_victim_time_sec
        nonlocal slot_score_refresh_calls, slot_score_refresh_time_sec

        score_start = time.perf_counter()
        score = il_model.score_one(features)
        dt = time.perf_counter() - score_start

        if role == "candidate":
            slot_score_candidate_calls += 1
            slot_score_candidate_time_sec += dt
            slot_admit_score_time_sec += dt
        elif role == "victim":
            slot_score_victim_calls += 1
            slot_score_victim_time_sec += dt
            slot_admit_score_time_sec += dt
        elif role == "refresh":
            slot_score_refresh_calls += 1
            slot_score_refresh_time_sec += dt
        else:
            slot_admit_score_time_sec += dt

        return float(score)

    def _score_obj(obj_id: Optional[str], role: str) -> Optional[float]:
        if not obj_id:
            return None
        gaps = feature_table.get_last_gaps(obj_id)
        if gaps is None:
            return None
        if freq_feature_source == "prev_slot":
            freq_val = int(prev_slot_freq.get(obj_id, 0))
        else:
            freq_val = int(feature_table.get_freq(obj_id))
        feats = build_features_from_gaps(
            gaps=gaps,
            freq=freq_val,
            num_gaps=num_gaps,
            use_freq_feature=il_cfg.use_freq_feature,
            freq_feature_mode=il_cfg.freq_feature_mode,
            slot_size=slot_size,
            missing_gap_value=il_cfg.missing_gap_value,
        )
        return _score_with_role(feats, role)

    def _is_score_stale(obj_id: str, current_slot: int) -> bool:
        if cache_score_stale_slots <= 0:
            return False
        last = cache_score_slot.get(obj_id)
        if last is None:
            return True
        return (current_slot - last) >= cache_score_stale_slots

    def _get_cached_score(obj_id: str, current_slot: int) -> Optional[float]:
        nonlocal slot_victim_scored_total, slot_victim_scored_map, slot_victim_scored_fresh
        score = cache_score_map.get(obj_id)
        stale = False
        if score is not None:
            stale = _is_score_stale(obj_id, current_slot)
        if score is None or stale:
            score = _score_obj(obj_id, "victim")
            if score is not None:
                cache_score_map[obj_id] = score
                cache_score_slot[obj_id] = current_slot
                slot_victim_scored_fresh += 1
        else:
            slot_victim_scored_map += 1
        if score is not None:
            slot_victim_scored_total += 1
        return score

    def _select_victim_score(current_slot: int) -> Tuple[Optional[str], Optional[float]]:
        nonlocal slot_victim_select_calls, slot_victim_scored_total
        nonlocal slot_victim_scored_map, slot_victim_scored_fresh

        if cache.capacity <= 0 or len(cache) == 0:
            return None, None

        slot_victim_select_calls += 1

        if victim_sample_mode == "lru" or victim_sample_k <= 1:
            victim_id = cache.peek_lru()
            if victim_id is None:
                return None, None
            score = _get_cached_score(victim_id, current_slot)
            return victim_id, score

        candidates = sample_lru_tail(cache, victim_sample_k)
        if not candidates:
            victim_id = cache.peek_lru()
            if victim_id is None:
                return None, None
            score = _get_cached_score(victim_id, current_slot)
            return victim_id, score

        best_id = None
        best_score = None
        for cand_id in candidates:
            score = _get_cached_score(cand_id, current_slot)
            if score is None:
                continue
            if best_score is None or score < best_score:
                best_score = score
                best_id = cand_id

        if best_id is None:
            victim_id = cache.peek_lru()
            if victim_id is None:
                return None, None
            score = _get_cached_score(victim_id, current_slot)
            return victim_id, score

        return best_id, best_score

    def _insert_with_victim(obj_id: str, victim_id: Optional[str]) -> Tuple[bool, Optional[str], bool]:
        """
        Insert dengan victim spesifik (score-aware). Return (inserted, evicted_id, used_specific).
        """
        if cache.capacity <= 0:
            return False, None, False
        data = getattr(cache, "_data", None)
        if data is None:
            inserted, evicted_id = cache.insert_with_eviction(obj_id)
            return inserted, evicted_id, False
        if obj_id in data:
            data.move_to_end(obj_id, last=True)
            return False, None, False
        evicted_id = None
        used_specific = False
        if len(data) >= cache.capacity:
            if victim_id and victim_id in data:
                del data[victim_id]
                evicted_id = victim_id
                used_specific = True
            else:
                evicted_id, _ = data.popitem(last=False)
        data[obj_id] = None
        return True, evicted_id, used_specific

    slot_cache_requests = 0
    slot_cache_hits = 0
    slot_admitted_objects: set[str] = set()
    slot_admit_eligible = 0
    slot_admit = 0
    slot_reject_gate = 0
    slot_reject_tau = 0
    slot_reject_replace = 0
    slot_insert = 0
    slot_evict = 0
    slot_evict_score = 0
    slot_admit_score_time_sec = 0.0
    slot_score_candidate_calls = 0
    slot_score_candidate_time_sec = 0.0
    slot_score_victim_calls = 0
    slot_score_victim_time_sec = 0.0
    slot_score_refresh_calls = 0
    slot_score_refresh_time_sec = 0.0
    slot_victim_select_calls = 0
    slot_victim_scored_total = 0
    slot_victim_scored_map = 0
    slot_victim_scored_fresh = 0
    slot_cache_score_refresh_count = 0

    def _prefill_cache(D_t: List[Dict[str, Any]]) -> int:
        if not D_t or cache.capacity <= 0:
            return 0
        X = np.asarray([row["x"] for row in D_t], dtype=float)
        scores = il_model.score_batch(X)
        order = np.argsort(-scores)
        filled = 0
        for idx in order:
            if len(cache) >= cache.capacity:
                break
            cache.insert(D_t[idx]["object_id"])
            obj_id = D_t[idx]["object_id"]
            cache_score_map[obj_id] = float(scores[idx])
            cache_score_slot[obj_id] = slot_index
            filled += 1
        return filled

    def _sample_cache_ids(mode: str, k: int) -> List[str]:
        if k <= 0 or len(cache) == 0:
            return []
        if mode == "lru":
            victim_id = cache.peek_lru()
            return [victim_id] if victim_id else []
        if mode == "sample":
            data = getattr(cache, "_data", None)
            if not data:
                return []
            keys = list(data.keys())
            if len(keys) <= k:
                return keys
            idx = rng.choice(len(keys), size=k, replace=False)
            return [keys[i] for i in idx]
        return sample_lru_tail(cache, k)

    def _refresh_cache_scores(current_slot: int) -> Tuple[int, float]:
        if cache_score_refresh_k <= 0 or len(cache) == 0:
            return 0, 0.0
        ids = _sample_cache_ids(cache_score_refresh_mode, cache_score_refresh_k)
        if not ids:
            return 0, 0.0
        refresh_count = 0
        refresh_start = time.perf_counter()
        for obj_id in ids:
            score = _score_obj(obj_id, "refresh")
            if score is None:
                continue
            cache_score_map[obj_id] = score
            cache_score_slot[obj_id] = current_slot
            refresh_count += 1
        refresh_time = time.perf_counter() - refresh_start
        return refresh_count, refresh_time


    def _finalize_slot(slot_req_count_local: int, end_req_idx: int) -> None:
        nonlocal slot_index, admit_tau_current, prev_slot_freq
        nonlocal slot_cache_requests, slot_admit_eligible, slot_admit, slot_reject_gate
        nonlocal slot_reject_tau, slot_reject_replace
        nonlocal slot_insert, slot_evict, slot_evict_score
        nonlocal slot_admit_score_time_sec, slot_cache_hits
        nonlocal slot_cache_score_refresh_count, slot_score_refresh_time_sec
        nonlocal slot_score_candidate_calls, slot_score_candidate_time_sec
        nonlocal slot_score_victim_calls, slot_score_victim_time_sec
        nonlocal slot_score_refresh_calls
        nonlocal slot_victim_select_calls, slot_victim_scored_total
        nonlocal slot_victim_scored_map, slot_victim_scored_fresh
        nonlocal slot_admitted_objects, cap_ratio_current
        nonlocal cand_score_reservoir, cand_score_seen

        if not slot_stats:
            return

        slot_compute_start = time.perf_counter()

        D_t, label_info = build_slot_dataset_from_stats(
            slot_stats,
            il_cfg.pop_top_percent,
            num_gaps,
            il_cfg.missing_gap_value,
            label_mode=il_cfg.label_mode,
            capacity_objects=capacity_objects,
            label_capacity_gamma=il_cfg.label_capacity_gamma,
            use_freq_feature=il_cfg.use_freq_feature,
            freq_feature_mode=il_cfg.freq_feature_mode,
            freq_feature_source=freq_feature_source,
            prev_slot_freq=prev_slot_freq,
            slot_size=slot_size,
        )

        if slot_log_mode == "full":
            dt_summary, sample_objs = summarize_slot_dataset(
                D_t, n_features, sample_k=max(slot_log_sample_k, 0)
            )
        else:
            dt_summary, sample_objs = {}, []

        slot_index += 1
        class1_metrics = compute_class1_metrics(D_t, il_model)
        update_start = time.perf_counter()
        slot_info = il_model.update_slot(D_t)
        update_model_time_sec = time.perf_counter() - update_start
        cls1_after = compute_class1_metrics(D_t, il_model)
        slot_cache_score_refresh_count, _ = _refresh_cache_scores(slot_index)

        admit_tau_used = admit_tau_current
        gamma_used = float(tau_capacity_gamma_current)
        target_pos_rate = label_info.get("pos_ratio")
        if target_pos_rate is None or target_pos_rate != target_pos_rate:
            target_pos_rate = float(il_cfg.pop_top_percent)
        n_objects_slot = float(label_info.get("n_objects", 0.0))
        cap_ratio = float(capacity_objects) / n_objects_slot if n_objects_slot > 0 else 0.0
        cap_ratio_current = cap_ratio
        if n_objects_slot > 0:
            target_pos_rate = min(float(target_pos_rate), gamma_used * cap_ratio)
        tau_pos_rate_min_eff = float(tau_pos_rate_min)
        target_pos_rate = float(np.clip(target_pos_rate, tau_pos_rate_min_eff, tau_pos_rate_max))

        admit_score_start = time.perf_counter()
        admit_tau_raw: Optional[float] = None
        admit_score_stats: Dict[str, Any] = {}
        tau_source = "D_t"
        reservoir_n = int(len(cand_score_reservoir))
        use_reservoir = bool(tau_use_candidate_reservoir and reservoir_n >= tau_reservoir_min_samples)

        if use_reservoir:
            tau_cand = compute_tau_from_scores(cand_score_reservoir, float(target_pos_rate))
            if tau_cand is not None:
                admit_tau_raw = tau_cand
                tau_source = "eligible_reservoir"
                admit_score_stats = {
                    "score_p50": float(np.percentile(cand_score_reservoir, 50)),
                    "score_p90": float(np.percentile(cand_score_reservoir, 90)),
                    "score_p99": float(np.percentile(cand_score_reservoir, 99)),
                    "score_max": float(np.max(cand_score_reservoir)),
                    "reservoir_n": reservoir_n,
                    "reservoir_min_samples": int(tau_reservoir_min_samples),
                }
            elif tau_reservoir_fallback == "D_t":
                admit_tau_raw, admit_score_stats = compute_admit_tau(D_t, il_model, float(target_pos_rate))
                tau_source = "D_t_fallback"
            elif tau_reservoir_fallback == "ema" and admit_tau_current is not None:
                admit_tau_raw = admit_tau_current
                tau_source = "ema_fallback"
            elif tau_reservoir_fallback == "none":
                admit_tau_raw = None
                tau_source = "none_fallback"
        else:
            if tau_reservoir_fallback == "ema" and admit_tau_current is not None:
                admit_tau_raw = admit_tau_current
                tau_source = "ema_fallback"
                admit_score_stats = {
                    "reservoir_n": reservoir_n,
                    "reservoir_min_samples": int(tau_reservoir_min_samples),
                }
            elif tau_reservoir_fallback == "none":
                admit_tau_raw = None
                tau_source = "none_fallback"
            else:
                admit_tau_raw, admit_score_stats = compute_admit_tau(D_t, il_model, float(target_pos_rate))
                tau_source = "D_t"
        admit_score_time_sec = time.perf_counter() - admit_score_start
        if admit_score_stats is None:
            admit_score_stats = {}
        if "reservoir_n" not in admit_score_stats:
            admit_score_stats["reservoir_n"] = reservoir_n
        if "reservoir_min_samples" not in admit_score_stats:
            admit_score_stats["reservoir_min_samples"] = int(tau_reservoir_min_samples)
        admit_score_stats["tau_source"] = tau_source
        if il_cfg.admit_use_tau and admit_tau_raw is not None:
            if admit_tau_current is None or admit_tau_ema_alpha <= 0.0:
                admit_tau_current = admit_tau_raw
            else:
                admit_tau_current = (
                    admit_tau_ema_alpha * admit_tau_raw
                    + (1.0 - admit_tau_ema_alpha) * admit_tau_current
                )
        elif not il_cfg.admit_use_tau:
            admit_tau_current = None

        start_req = (slot_index - 1) * slot_size
        end_req = end_req_idx
        phase = "warmup" if end_req < warmup_requests else "cache"

        prefill_count = 0
        if warmup_prefill and phase == "warmup" and (end_req + 1) == warmup_requests:
            prefill_count = _prefill_cache(D_t)

        admit_rate = (slot_admit / float(slot_admit_eligible)) if slot_admit_eligible > 0 else None
        churn_rate = (slot_evict / float(slot_cache_requests)) if slot_cache_requests > 0 else None
        slot_hit_ratio = (slot_cache_hits / float(slot_cache_requests)) if slot_cache_requests > 0 else None
        admit_tau_next = admit_tau_current
        total_slot_reqs = sum(int(stats["freq"]) for stats in slot_stats.values()) if slot_stats else 0
        admit_freq = sum(
            int(slot_stats[obj_id]["freq"])
            for obj_id in slot_admitted_objects
            if obj_id in slot_stats
        )
        admit_coverage = (admit_freq / float(total_slot_reqs)) if total_slot_reqs > 0 else None
        churn_per_capacity = (slot_evict / float(capacity_objects)) if capacity_objects > 0 else None
        cache_fill_ratio = (len(cache) / float(cache.capacity)) if cache.capacity > 0 else None

        slot_log: Dict[str, Any] = {
            "slot_index": slot_index,
            "phase": phase,
            "slot_size_requests": slot_req_count_local,
            "slot_start_request_idx": start_req,
            "slot_end_request_idx": end_req,
            "stats_total_requests": stats.total_requests,
            "stats_cache_hits": stats.cache_hits,
            "hit_ratio": stats.hit_ratio,
            "slot_cache_hits": slot_cache_hits,
            "slot_hit_ratio": slot_hit_ratio,
            "admit_coverage": admit_coverage,
            "admit_coverage_freq": admit_freq,
            "slot_total_freq": total_slot_reqs,
            "slot_compute_time_sec": float(time.perf_counter() - slot_compute_start),
            "update_model_time_sec": float(update_model_time_sec),
            "admit_score_time_sec": float(admit_score_time_sec),
            "admit_score_request_time_sec": float(slot_admit_score_time_sec),
            "score_calls_candidate": int(slot_score_candidate_calls),
            "score_calls_victim": int(slot_score_victim_calls),
            "score_calls_refresh": int(slot_score_refresh_calls),
            "score_time_candidate": float(slot_score_candidate_time_sec),
            "score_time_victim": float(slot_score_victim_time_sec),
            "score_time_refresh": float(slot_score_refresh_time_sec),
            "victim_select_calls": int(slot_victim_select_calls),
            "victim_scored_total": int(slot_victim_scored_total),
            "victim_scored_map": int(slot_victim_scored_map),
            "victim_scored_fresh": int(slot_victim_scored_fresh),
            "cache_score_refresh_count": int(slot_cache_score_refresh_count),
            "cache_score_map_size": int(len(cache_score_map)),
            "cache_fill_ratio": cache_fill_ratio,
            "cls1_metrics_before_update": class1_metrics,
            "cls1_metrics_after_update": cls1_after,
            "use_freq_feature": il_cfg.use_freq_feature,
            "freq_feature_mode": il_cfg.freq_feature_mode,
            "freq_feature_source": freq_feature_source,
            "label_info": label_info,
            "admit_use_tau": il_cfg.admit_use_tau,
            "admit_target_pos_rate": float(target_pos_rate),
            "admit_tau_used": admit_tau_used,
            "admit_tau_raw": admit_tau_raw,
            "admit_tau_next": admit_tau_next,
            "admit_tau_current": admit_tau_current,
            "admit_tau_ema_alpha": admit_tau_ema_alpha,
            "admit_score_stats": admit_score_stats,
            "tau_use_candidate_reservoir": tau_use_candidate_reservoir,
            "tau_score_sample_size": tau_score_sample_size,
            "tau_reservoir_min_samples": tau_reservoir_min_samples,
            "tau_reservoir_fallback": tau_reservoir_fallback,
            "tau_capacity_gamma": gamma_used,
            "tau_pos_rate_min": tau_pos_rate_min,
            "tau_pos_rate_min_eff": tau_pos_rate_min_eff,
            "tau_pos_rate_max": tau_pos_rate_max,
            "gate_min_hist_len": gate_min_hist_len,
            "warmup_prefill": warmup_prefill,
            "warmup_prefill_count": prefill_count,
            "slot_cache_requests": slot_cache_requests,
            "slot_admit_eligible": slot_admit_eligible,
            "slot_admit": slot_admit,
            "slot_admit_rate": admit_rate,
            "slot_reject_gate": slot_reject_gate,
            "slot_reject_tau": slot_reject_tau,
            "slot_reject_replace": slot_reject_replace,
            "slot_insert": slot_insert,
            "slot_evict": slot_evict,
            "slot_evict_score": slot_evict_score,
            "churn_rate": churn_rate,
            "churn_per_capacity": churn_per_capacity,
            "feature_stats": dt_summary.get("feature_stats", {}) if slot_log_mode == "full" else {},
            "freq_stats": dt_summary.get("freq_stats", {}) if slot_log_mode == "full" else {},
            "sample_objects": sample_objs if slot_log_mode == "full" else [],
        }
        slot_log.update(_compact_slot_info(slot_info))
        slot_logs.append(slot_log)

        prev_slot_freq = {obj_id: int(stats["freq"]) for obj_id, stats in slot_stats.items()}

        slot_cache_requests = 0
        slot_cache_hits = 0
        slot_admitted_objects = set()
        slot_admit_eligible = 0
        slot_admit = 0
        slot_reject_gate = 0
        slot_reject_tau = 0
        slot_reject_replace = 0
        slot_insert = 0
        slot_evict = 0
        slot_evict_score = 0
        slot_admit_score_time_sec = 0.0
        slot_score_candidate_calls = 0
        slot_score_candidate_time_sec = 0.0
        slot_score_victim_calls = 0
        slot_score_victim_time_sec = 0.0
        slot_score_refresh_calls = 0
        slot_score_refresh_time_sec = 0.0
        slot_victim_select_calls = 0
        slot_victim_scored_total = 0
        slot_victim_scored_map = 0
        slot_victim_scored_fresh = 0
        slot_cache_score_refresh_count = 0
        cand_score_reservoir = []
        cand_score_seen = 0

    while True:
        try:
            req = next(req_iter)
        except StopIteration:
            if slot_stats:
                end_req = min(slot_index * slot_size + slot_req_count, total_requests) - 1
                _finalize_slot(slot_req_count, end_req)
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
            if prev_ts is None:
                ts = ts_raw
            else:
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
                missing = il_cfg.missing_gap_value
                hist_len = sum(1 for g in gaps[:num_gaps] if g != missing)
                if freq_feature_source == "prev_slot":
                    freq_val = int(prev_slot_freq.get(obj_id, 0))
                else:
                    freq_val = int(slot_stats.get(obj_id, {}).get("freq", 0))

                features = build_features_from_gaps(
                    gaps=gaps,
                    freq=freq_val,
                    num_gaps=num_gaps,
                    use_freq_feature=il_cfg.use_freq_feature,
                    freq_feature_mode=il_cfg.freq_feature_mode,
                    slot_size=slot_size,
                    missing_gap_value=il_cfg.missing_gap_value,
                )

                if gate_min_hist_len > 0 and hist_len < gate_min_hist_len:
                    slot_reject_gate += 1
                    y_hat = 0
                else:
                    slot_admit_eligible += 1
                    current_slot = slot_index + 1
                    cand_score: Optional[float] = None
                    victim_id_for_admit: Optional[str] = None

                    if il_cfg.admit_use_tau and admit_tau_current is not None:
                        cand_score = _score_with_role(features, "candidate")
                        cand_score_seen = reservoir_update(
                            cand_score_reservoir,
                            cand_score_seen,
                            float(cand_score),
                            tau_score_sample_size,
                            rng,
                        )
                        tau_pass = cand_score >= admit_tau_current
                    else:
                        cand_score = _score_with_role(features, "candidate")
                        cand_score_seen = reservoir_update(
                            cand_score_reservoir,
                            cand_score_seen,
                            float(cand_score),
                            tau_score_sample_size,
                            rng,
                        )
                        tau_pass = True

                    if il_cfg.admit_use_tau and admit_tau_current is not None:
                        if tau_pass:
                            if cache.capacity > 0 and len(cache) >= cache.capacity:
                                victim_id_for_admit, victim_score = _select_victim_score(current_slot)
                                if victim_score is None or (cand_score is not None and cand_score > victim_score):
                                    y_hat = 1
                                else:
                                    slot_reject_replace += 1
                                    y_hat = 0
                            else:
                                y_hat = 1
                        else:
                            slot_reject_tau += 1
                            y_hat = 0
                    else:
                        score_start = time.perf_counter()
                        y_hat = int(il_model.predict(features))
                        slot_admit_score_time_sec += time.perf_counter() - score_start

                if y_hat == 1:
                    if cache.capacity > 0 and len(cache) >= cache.capacity and victim_id_for_admit is not None:
                        inserted, evicted_id, used_specific = _insert_with_victim(obj_id, victim_id_for_admit)
                    else:
                        inserted, evicted_id = cache.insert_with_eviction(obj_id)
                        used_specific = False
                    if inserted:
                        slot_admit += 1
                        slot_insert += 1
                        slot_admitted_objects.add(obj_id)
                        if evicted_id is not None:
                            slot_evict += 1
                            if used_specific:
                                slot_evict_score += 1
                            cache_score_map.pop(evicted_id, None)
                            cache_score_slot.pop(evicted_id, None)
                    if cand_score is not None:
                        cache_score_map[obj_id] = float(cand_score)
                        cache_score_slot[obj_id] = current_slot

        # 3) update counter & cek boundary slot
        global_idx += 1
        slot_req_count += 1

        if slot_req_count >= slot_size:
            end_req = (slot_index * slot_size) + slot_req_count - 1
            _finalize_slot(slot_req_count, end_req)

            pbar.update(1)
            current_hr = stats.hit_ratio if stats.total_requests > 0 else 0.0
            pbar.set_postfix(hr=f"{current_hr:.4f}")

            slot_stats = {}
            slot_req_count = 0

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

def get_next_run_id(results_root: str, dataset: str, model: str) -> Tuple[str, str]:
    """
    Menentukan ID run berikutnya (001, 002, ...) agar tidak overwrite.
    """
    dataset_dir = os.path.join(results_root, dataset)
    os.makedirs(dataset_dir, exist_ok=True)

    existing_ids: List[int] = []
    for fname in os.listdir(dataset_dir):
        if f"_{model}_" not in fname:
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
    ds_cfg,
    il_cfg,
    cache_cfg,
    model_name_base: str = "ilnse",
    results_root: str = "results",
) -> None:
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

    cache_eval = list(getattr(cache_cfg, "cache_size_percentages_eval", []))
    cache_percentages = cache_eval if cache_eval else list(cache_cfg.cache_size_percentages)

    # Hitung kapasitas dinamis sekali
    capacities_objects = get_dynamic_capacities(trace_path, total_requests, cache_percentages)

    print(f"=== IL-based Edge Cache Experiment ({dataset_name} trace) ===")
    print(f"Trace path        : {trace_path}")
    print(f"Total requests    : {total_requests}")
    print(f"Warm-up requests  : {warmup_requests}")
    print(f"Slot size         : {slot_size}")
    print(f"IL num_gaps       : {il_cfg.num_gaps}")
    print(f"IL top_percent    : {il_cfg.pop_top_percent}")
    print(f"IL label_mode     : {il_cfg.label_mode}")
    print(f"IL freq_feature   : {il_cfg.use_freq_feature} ({il_cfg.freq_feature_mode}, {il_cfg.freq_feature_source})")
    print(f"IL admit_tau      : {il_cfg.admit_use_tau} (ema={il_cfg.admit_tau_ema_alpha})")
    print(
        f"IL tau_reservoir  : {il_cfg.tau_use_candidate_reservoir} "
        f"(size={il_cfg.tau_score_sample_size}, min={il_cfg.tau_reservoir_min_samples}, fallback={il_cfg.tau_reservoir_fallback})"
    )
    print(f"IL tau_capacity   : gamma={il_cfg.tau_capacity_gamma}, clamp=[{il_cfg.tau_pos_rate_min}, {il_cfg.tau_pos_rate_max}]")
    print(f"IL victim_sample  : mode=tail, k={il_cfg.victim_sample_k}")
    print(f"IL score_refresh  : mode=tail, k={il_cfg.cache_score_refresh_k}, stale={il_cfg.cache_score_stale_slots}")
    print(f"IL slot_log_mode  : {il_cfg.slot_log_mode} (sample_k={il_cfg.slot_log_sample_k})")
    print(f"IL sigmoid (a, b) : ({il_cfg.sigmoid_a}, {il_cfg.sigmoid_b})")
    print(f"Max learners      : {il_cfg.max_classifiers}")
    print(f"Cache capacities  : {list(cache_percentages)}")
    print()

    model_name = model_name_base
    results: List[CacheStats] = []
    run_id, dataset_dir = get_next_run_id(results_root, dataset_name, model_name)

    for capacity in capacities_objects:
        print(f"[RUN] Capacity Objects={capacity}")

        run_start = time.perf_counter()
        stats, slot_logs = run_single_capacity(
            trace_path=trace_path,
            total_requests=total_requests,
            warmup_requests=warmup_requests,
            slot_size=slot_size,
            capacity_objects=capacity,
            il_cfg=il_cfg,
        )
        run_time_sec = time.perf_counter() - run_start
        results.append(stats)

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
            "total_runtime_sec": float(run_time_sec),
            "label_mode": il_cfg.label_mode,
            "label_capacity_gamma": il_cfg.label_capacity_gamma,
            "use_freq_feature": il_cfg.use_freq_feature,
            "freq_feature_mode": il_cfg.freq_feature_mode,
            "freq_feature_source": il_cfg.freq_feature_source,
            "admit_use_tau": il_cfg.admit_use_tau,
            "admit_tau_ema_alpha": il_cfg.admit_tau_ema_alpha,
            "tau_use_candidate_reservoir": il_cfg.tau_use_candidate_reservoir,
            "tau_score_sample_size": il_cfg.tau_score_sample_size,
            "tau_reservoir_min_samples": il_cfg.tau_reservoir_min_samples,
            "tau_reservoir_fallback": il_cfg.tau_reservoir_fallback,
            "tau_capacity_gamma": il_cfg.tau_capacity_gamma,
            "tau_pos_rate_min": il_cfg.tau_pos_rate_min,
            "tau_pos_rate_max": il_cfg.tau_pos_rate_max,
            "victim_sample_k": il_cfg.victim_sample_k,
            "cache_score_refresh_k": il_cfg.cache_score_refresh_k,
            "cache_score_stale_slots": il_cfg.cache_score_stale_slots,
            "seed": il_cfg.seed,
            "slot_log_mode": il_cfg.slot_log_mode,
            "slot_log_sample_k": il_cfg.slot_log_sample_k,
            "max_classifiers": il_cfg.max_classifiers,
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
