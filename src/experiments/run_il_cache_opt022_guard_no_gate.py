# src/experiments/run_il_cache.py

from __future__ import annotations

import json
import math
import os
import sys
import time
from collections import deque
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
 
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.trace_reader import TraceReader
from src.data.feature_table import FeatureTable
from src.ml.learn_nse_opt import LearnNSE
from src.cache.cache_simulator import CacheStats
from src.cache.lru import LRUCache

FEATURE_SETS = ("A0", "A1", "A2", "A3")
FEATURE_WINDOWS = {"short": 1, "mid": 7, "long": 30}
POLLUTION_WINDOW_SLOTS = 1

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

# Fixed experiment settings (no external config)
IL_NUM_GAPS = 6
IL_POP_TOP_PERCENT = 0.20
IL_LABEL_TOPK_ROUNDING = "floor"
IL_LABEL_TIE_BREAK = "none"
IL_SIGMOID_A = 0.5
IL_SIGMOID_B = 10.0
IL_MAX_CLASSIFIERS = 20
IL_MISSING_GAP_VALUE = 1e6
CACHE_SIZE_PERCENTAGES = (0.8, 1.0, 2.0, 3.0, 4.0, 5.0)
ADMISSION_CAPACITY_ALPHA = 0.08
DRIFT_ALPHA_MIN = 0.005
DRIFT_ALPHA_MAX = 0.2
DRIFT_SENSITIVITY = 1.0
DRIFT_EMA_ALPHA = 0.3
DRIFT_STATS_ALPHA = 0.1
DRIFT_NORM_EPS = 1e-6
DRIFT_Z_CLIP = 3.0
DRIFT_WEIGHT_JSD = 0.4
DRIFT_WEIGHT_OVERLAP = 0.6
# GUARD-ONLY: disable drift contribution to budget
DRIFT_GAIN = 0.0
DRIFT_ALPHA_FLOOR_MULT = 0.7
# GUARD (no-gate): disable hard gate by allowing full candidate set
SCORE_GATE_TOP_PERCENT = 1.0
FILL_RATIO = 0.9
FILL_RATE = 0.05
PRESSURE_MISS_GAMMA = 0.5
SCORE_SPREAD_Q = 0.9
SCORE_SPREAD_EMA_ALPHA = 0.2
SCORE_SPREAD_EPS = 1e-6
SCORE_QUALITY_MIN = 0.7
SCORE_QUALITY_MAX = 1.3
SCORE_QUALITY_MIN_BOOST = 0.1
ADMISSION_PRECISION_TARGET = 0.12
ADMISSION_PRECISION_SENSITIVITY = 1.0
CAPACITY_ALPHA_SCALE_MIN = 0.4

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


def _get_feature_dim(num_gaps: int, feature_set: str) -> int:
    if feature_set == "A0":
        return num_gaps
    if feature_set == "A1":
        return num_gaps + 1
    if feature_set == "A2":
        return num_gaps + 4
    if feature_set == "A3":
        return num_gaps + 5
    raise ValueError(f"Unknown feature_set: {feature_set}")


def _compute_admission_budget(
    miss_requests: int,
    capacity_objects: int,
    alpha: float,
    pressure_mult: float = 1.0,
) -> int:
    if miss_requests <= 0:
        return 0
    return int(math.ceil(capacity_objects * alpha * pressure_mult))


def _select_top_m(candidate_ids: List[str], candidate_scores: List[float], m: int) -> set[str]:
    if m <= 0 or not candidate_ids:
        return set()
    n = len(candidate_ids)
    if m >= n:
        return set(candidate_ids)
    scores = np.asarray(candidate_scores, dtype=float)
    kth = np.partition(scores, -m)[-m]
    greater_idx = np.where(scores > kth)[0]
    if len(greater_idx) >= m:
        top_idx = np.argpartition(scores, -m)[-m:]
        return {candidate_ids[i] for i in top_idx}
    remaining = m - len(greater_idx)
    eq_idx = np.where(scores == kth)[0]
    selected_idx = np.concatenate([greater_idx, eq_idx[:remaining]])
    return {candidate_ids[i] for i in selected_idx}


def _auc_roc(y_true: List[int], y_score: List[float]) -> float:
    y_true_arr = np.asarray(y_true, dtype=int)
    y_score_arr = np.asarray(y_score, dtype=float)
    pos = y_true_arr == 1
    n_pos = int(pos.sum())
    n_neg = int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(y_score_arr)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(y_score_arr) + 1, dtype=float)

    # average ranks for ties
    s = y_score_arr[order]
    i = 0
    while i < len(s):
        j = i + 1
        while j < len(s) and s[j] == s[i]:
            j += 1
        if j - i > 1:
            avg = ranks[order[i:j]].mean()
            ranks[order[i:j]] = avg
        i = j

    sum_ranks_pos = float(ranks[pos].sum())
    return float((sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _avg_precision(y_true: List[int], y_score: List[float]) -> float:
    y_true_arr = np.asarray(y_true, dtype=int)
    y_score_arr = np.asarray(y_score, dtype=float)
    n_pos = int((y_true_arr == 1).sum())
    if n_pos == 0:
        return float("nan")

    order = np.argsort(-y_score_arr)
    y_sorted = y_true_arr[order]
    tp = np.cumsum(y_sorted == 1)
    denom = np.arange(1, len(y_sorted) + 1)
    precision = tp / denom
    return float((precision[y_sorted == 1]).sum() / n_pos)


def _precision_at_k(y_true: List[int], y_score: List[float], k: int) -> float:
    y_true_arr = np.asarray(y_true, dtype=int)
    y_score_arr = np.asarray(y_score, dtype=float)
    if k <= 0:
        return float("nan")
    k = min(k, len(y_true_arr))
    idx = np.argsort(-y_score_arr)[:k]
    return float((y_true_arr[idx] == 1).mean())

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


def _weighted_jaccard(
    prev_dist: Optional[Dict[str, float]],
    curr_dist: Optional[Dict[str, float]],
) -> float:
    if not prev_dist or not curr_dist:
        return 0.0
    keys = set(prev_dist) | set(curr_dist)
    if not keys:
        return 0.0
    num = 0.0
    den = 0.0
    for k in keys:
        a = prev_dist.get(k, 0.0)
        b = curr_dist.get(k, 0.0)
        num += min(a, b)
        den += max(a, b)
    if den <= 0.0:
        return 0.0
    return float(num / den)


def _sum_history(window: int, slot_index: int, history: deque[Tuple[int, int]]) -> int:
    if slot_index <= 0 or window <= 0:
        return 0
    start_slot = max(1, slot_index - window + 1)
    return sum(count for slot_id, count in history if slot_id >= start_slot)


def _get_history_features(
    obj_id: str,
    slot_index: int,
    freq_history: Dict[str, deque[Tuple[int, int]]],
    cum_counts: Dict[str, int],
    windows: Dict[str, int],
) -> Tuple[int, int, int, int]:
    history = freq_history.get(obj_id)
    if history is None:
        return 0, 0, 0, int(cum_counts.get(obj_id, 0))
    f_short = _sum_history(windows["short"], slot_index, history)
    f_mid = _sum_history(windows["mid"], slot_index, history)
    f_long = _sum_history(windows["long"], slot_index, history)
    f_cum = int(cum_counts.get(obj_id, 0))
    return f_short, f_mid, f_long, f_cum


def _build_feature_vector(
    feature_set: str,
    gaps: List[float],
    history_features: Tuple[int, int, int, int],
) -> List[float]:
    if feature_set == "A0":
        return list(gaps)
    if feature_set == "A1":
        f_short, _, _, _ = history_features
        return list(gaps) + [float(f_short)]
    f_short, f_mid, f_long, f_cum = history_features
    if feature_set == "A2":
        return list(gaps) + [float(f_short), float(f_mid), float(f_long), float(f_cum)]
    if feature_set == "A3":
        return [float(f_short), float(f_mid), float(f_long), float(f_cum)]
    raise ValueError(f"Unknown feature_set: {feature_set}")


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
        "top_ratio": float(top_ratio),
        "rounding": label_topk_rounding,
        "tie_break": label_tie_break,
        "freq_at_k": None,
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


def _update_freq_history(
    slot_index: int,
    slot_stats: Dict[str, Dict[str, Any]],
    freq_history: Dict[str, deque[Tuple[int, int]]],
    cum_counts: Dict[str, int],
    long_window: int,
) -> None:
    cutoff = slot_index - long_window + 1
    for obj_id, stats in slot_stats.items():
        count = int(stats.get("freq", 0))
        if count <= 0:
            continue
        history = freq_history.setdefault(obj_id, deque())
        while history and history[0][0] < cutoff:
            history.popleft()
        history.append((slot_index, count))
        cum_counts[obj_id] = int(cum_counts.get(obj_id, 0)) + count


def build_slot_dataset_from_stats(
    slot_stats: Dict[str, Dict],
    top_ratio: float,
    num_gaps: int,
    missing_gap_value: float,
    feature_set: str,
    history_slot_index: int,
    freq_history: Dict[str, deque[Tuple[int, int]]],
    cum_counts: Dict[str, int],
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
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"Unknown feature_set: {feature_set}")
    num_features = _get_feature_dim(num_gaps, feature_set)

    label_info: Dict[str, Any] = {}
    if not slot_stats:
        return [], label_info

    items: List[Tuple[str, Dict]] = list(slot_stats.items())
    # urutkan berdasarkan freq menurun (popularitas per-slot)
    items.sort(key=lambda kv: kv[1]["freq"], reverse=True)

    top_ids, label_info = _select_top_ids_from_stats(
        slot_stats,
        top_ratio,
        label_topk_rounding,
        label_tie_break,
    )

    dataset: List[Dict] = []
    for obj_id, stats in items:
        gaps = stats["last_gaps"]
        if len(gaps) != num_gaps:
            if len(gaps) < num_gaps:
                gaps = list(gaps) + [missing_gap_value] * (num_gaps - len(gaps))
            else:
                gaps = list(gaps[:num_gaps])

        history_feats = _get_history_features(
            obj_id,
            history_slot_index,
            freq_history,
            cum_counts,
            FEATURE_WINDOWS,
        )
        features = _build_feature_vector(
            feature_set,
            gaps,
            history_feats,
        )
        if len(features) != num_features:
            raise ValueError(
                f"feature length mismatch: expected {num_features}, got {len(features)}"
            )
        hist_len = sum(1 for g in gaps if g != missing_gap_value)

        y = 1 if obj_id in top_ids else 0
        dataset.append(
            {
                "x": features,
                "y": y,
                "freq": stats["freq"],
                "object_id": obj_id,
                "hist_len": hist_len,
            }
        )

    return dataset, label_info


# ---------------------------------------------------------------------------
# Satu run IL+LRU untuk satu kapasitas cache
# ---------------------------------------------------------------------------

def run_single_capacity(
    trace_path: str,
    total_requests: int,
    warmup_requests: int,
    slot_size: int,
    capacity_objects: int,
    feature_set: str,
    slot_log_path: Optional[str] = None,
) -> Tuple[CacheStats, Dict[str, Any]]:
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

    if feature_set not in FEATURE_SETS:
        raise ValueError(f"Unknown feature_set: {feature_set}")

    num_gaps = IL_NUM_GAPS
    num_features = _get_feature_dim(num_gaps, feature_set)

    feature_table = FeatureTable(
        L=num_gaps,
        missing_gap_value=IL_MISSING_GAP_VALUE,
    )
    il_model = LearnNSE(
        n_features=num_features,
        a=IL_SIGMOID_A,
        b=IL_SIGMOID_B,
        max_learners=IL_MAX_CLASSIFIERS,
    )
    cache = LRUCache(capacity_objects=capacity_objects)
    stats = CacheStats(capacity_objects=capacity_objects)

    label_topk_rounding = IL_LABEL_TOPK_ROUNDING
    label_tie_break = IL_LABEL_TIE_BREAK
    admission_capacity_alpha = float(ADMISSION_CAPACITY_ALPHA)

    total_slots_expected = (total_requests + slot_size - 1) // slot_size
    pbar = tqdm(total=total_slots_expected, desc="Processing slots", unit="slot")

    global_idx = 0          # counter global request (0-based)
    slot_req_count = 0      # jumlah request dalam slot saat ini
    slot_index = 0          # 1-based untuk logging
    slot_stats: Dict[str, Dict] = {}  # per-slot: freq, last_gaps
    prev_ts: Optional[float] = None
    slot_cache_requests = 0
    slot_cache_hits = 0
    slot_cache_misses = 0
    total_hits_from_admitted = 0
    slot_admit_requests = 0
    slot_reject_requests = 0
    total_admit_requests = 0
    total_reject_requests = 0
    total_miss_requests = 0
    total_miss_candidates = 0
    total_admit_budget = 0
    total_admit_selected = 0
    total_admit_applied = 0
    total_admit_true_popular = 0
    total_pollution_count = 0
    total_pollution_total = 0
    total_update_time_s = 0.0
    total_update_slots = 0
    total_slots_processed = 0
    total_cache_slots = 0
    last_pollution_finalized = 0
    slot_log_file = open(slot_log_path, "w", encoding="utf-8") if slot_log_path else None
    prev_topk_dist: Optional[Dict[str, float]] = None
    drift_ema = 0.0
    drift_mean_ema = 0.0
    drift_var_ema = 0.0
    drift_stats_count = 0
    drift_raw_sum = 0.0
    drift_norm_sum = 0.0
    jsd_sum = 0.0
    overlap_sum = 0.0
    miss_rate_sum = 0.0
    alpha_sum = 0.0
    alpha_min_seen = None
    alpha_max_seen = None
    score_eval_slots = 0
    score_eval_n_sum = 0
    score_eval_pos_sum = 0
    score_auc_sum = 0.0
    score_ap_sum = 0.0
    score_precK_sum = 0.0
    score_auc_count = 0
    score_ap_count = 0
    score_precK_count = 0
    score_acc_sum = 0.0
    score_acc_count = 0
    score_precK_hr_next_n = 0
    score_precK_hr_next_sum_x = 0.0
    score_precK_hr_next_sum_y = 0.0
    score_precK_hr_next_sum_x2 = 0.0
    score_precK_hr_next_sum_y2 = 0.0
    score_precK_hr_next_sum_xy = 0.0
    prev_score_precK = None
    score_gate_k_sum = 0
    score_gate_applied_slots = 0
    fill_phase_slots = 0
    fill_min_budget_sum = 0
    pressure_mult_sum = 0.0
    pressure_mult_count = 0
    score_spread_ema = None
    score_spread_sum = 0.0
    score_spread_count = 0
    quality_mult_sum = 0.0
    quality_mult_count = 0
    precision_eff_sum = 0.0
    precision_eff_count = 0

    freq_history: Dict[str, deque[Tuple[int, int]]] = {}
    cum_counts: Dict[str, int] = {}

    applied_history: deque[Tuple[int, set[str]]] = deque(maxlen=3)

    admitted_by_slot: Dict[int, set[str]] = {}
    admitted_hit_by_slot: Dict[int, set[str]] = {}
    active_admit_slots_by_obj: Dict[str, set[int]] = {}
    slot_miss_obj_ids: List[str] = []
    slot_miss_features: List[List[float]] = []
    pending_admit_set: set[str] = set()
    pending_admit_source_slot: Optional[int] = None
    applied_admit_set: set[str] = set()
    applied_admit_source_slot: Optional[int] = None

    def _register_admission(obj_id: str, slot_num: int) -> None:
        admitted_by_slot.setdefault(slot_num, set()).add(obj_id)
        active_admit_slots_by_obj.setdefault(obj_id, set()).add(slot_num)

    def _mark_admit_hit(obj_id: str) -> None:
        slots = active_admit_slots_by_obj.get(obj_id)
        if not slots:
            return
        for slot_num in slots:
            admitted_hit_by_slot.setdefault(slot_num, set()).add(obj_id)

    def _finalize_pollution(slot_num: int) -> Tuple[int, int]:
        admitted = admitted_by_slot.get(slot_num, set())
        hits = admitted_hit_by_slot.get(slot_num, set())
        total = len(admitted)
        polluted = len(admitted - hits)
        return polluted, total

    def _prune_active_slots(expired_slot: int) -> None:
        for obj_id in admitted_by_slot.get(expired_slot, set()):
            active = active_admit_slots_by_obj.get(obj_id)
            if not active:
                continue
            active.discard(expired_slot)
            if not active:
                del active_admit_slots_by_obj[obj_id]

    def _apply_pending_admits(next_slot_num: int) -> None:
        nonlocal pending_admit_set
        nonlocal pending_admit_source_slot
        nonlocal applied_admit_set
        nonlocal applied_admit_source_slot
        nonlocal total_admit_applied
        if not pending_admit_set:
            applied_admit_set = set()
            applied_admit_source_slot = None
            return
        applied_admit_set = set(pending_admit_set)
        applied_admit_source_slot = pending_admit_source_slot
        pending_admit_set = set()
        pending_admit_source_slot = None
        total_admit_applied += len(applied_admit_set)
        if applied_admit_set:
            applied_history.append((next_slot_num, set(applied_admit_set)))
        for obj_id in applied_admit_set:
            cache.insert(obj_id)
            _register_admission(obj_id, next_slot_num)

    def _write_slot_log(payload: Dict[str, Any]) -> None:
        if slot_log_file is None:
            return
        slot_log_file.write(json.dumps(payload) + "\n")
        slot_log_file.flush()

    def _finalize_slot() -> None:
        nonlocal slot_index
        nonlocal slot_stats
        nonlocal slot_cache_requests
        nonlocal slot_cache_hits
        nonlocal slot_cache_misses
        nonlocal slot_miss_obj_ids
        nonlocal slot_miss_features
        nonlocal slot_admit_requests
        nonlocal slot_reject_requests
        nonlocal pending_admit_set
        nonlocal pending_admit_source_slot
        nonlocal applied_admit_set
        nonlocal applied_admit_source_slot
        nonlocal total_admit_selected
        nonlocal total_admit_requests
        nonlocal total_reject_requests
        nonlocal total_miss_requests
        nonlocal total_miss_candidates
        nonlocal total_admit_budget
        nonlocal total_admit_true_popular
        nonlocal total_pollution_count
        nonlocal total_pollution_total
        nonlocal total_update_time_s
        nonlocal total_update_slots
        nonlocal total_slots_processed
        nonlocal total_cache_slots
        nonlocal last_pollution_finalized
        nonlocal prev_topk_dist
        nonlocal drift_ema
        nonlocal drift_mean_ema
        nonlocal drift_var_ema
        nonlocal drift_stats_count
        nonlocal drift_raw_sum
        nonlocal drift_norm_sum
        nonlocal jsd_sum
        nonlocal overlap_sum
        nonlocal miss_rate_sum
        nonlocal alpha_sum
        nonlocal alpha_min_seen
        nonlocal alpha_max_seen
        nonlocal score_eval_slots
        nonlocal score_eval_n_sum
        nonlocal score_eval_pos_sum
        nonlocal score_auc_sum
        nonlocal score_ap_sum
        nonlocal score_precK_sum
        nonlocal score_auc_count
        nonlocal score_ap_count
        nonlocal score_precK_count
        nonlocal score_acc_sum
        nonlocal score_acc_count
        nonlocal score_precK_hr_next_n
        nonlocal score_precK_hr_next_sum_x
        nonlocal score_precK_hr_next_sum_y
        nonlocal score_precK_hr_next_sum_x2
        nonlocal score_precK_hr_next_sum_y2
        nonlocal score_precK_hr_next_sum_xy
        nonlocal prev_score_precK
        nonlocal score_gate_k_sum
        nonlocal score_gate_applied_slots
        nonlocal fill_phase_slots
        nonlocal fill_min_budget_sum
        nonlocal pressure_mult_sum
        nonlocal pressure_mult_count
        nonlocal score_spread_ema
        nonlocal score_spread_sum
        nonlocal score_spread_count
        nonlocal quality_mult_sum
        nonlocal quality_mult_count
        nonlocal precision_eff_sum
        nonlocal precision_eff_count
        nonlocal applied_history

        if not slot_stats:
            return

        D_t, label_info = build_slot_dataset_from_stats(
            slot_stats,
            IL_POP_TOP_PERCENT,
            num_gaps,
            IL_MISSING_GAP_VALUE,
            feature_set,
            slot_index,
            freq_history,
            cum_counts,
            label_topk_rounding,
            label_tie_break,
        )

        slot_index += 1
        slot_num = slot_index
        total_slots_processed += 1
        warmup_slots = warmup_requests // slot_size
        is_cache_slot = slot_num > warmup_slots

        # admission precision (applied set berasal dari slot sebelumnya)
        pos_ids: set[str] = set()
        admit_true_applied = None
        admit_precision = None
        precision_lag0 = None
        precision_lag1 = None
        precision_lag2 = None
        precision_eff = None
        if applied_admit_set:
            pos_ids = {row["object_id"] for row in D_t if row["y"] == 1}
            admit_true_applied = len(applied_admit_set & pos_ids)
            admit_precision = (
                float(admit_true_applied / len(applied_admit_set))
                if applied_admit_set
                else None
            )
        if not pos_ids:
            pos_ids = {row["object_id"] for row in D_t if row["y"] == 1}
        precision_by_lag: Dict[int, float] = {}
        for apply_slot, apply_set in applied_history:
            if not apply_set:
                continue
            lag = slot_num - apply_slot
            if lag < 0:
                continue
            precision_by_lag[lag] = float(len(apply_set & pos_ids) / len(apply_set))
        precision_lag0 = precision_by_lag.get(0)
        precision_lag1 = precision_by_lag.get(1)
        precision_lag2 = precision_by_lag.get(2)
        if precision_by_lag:
            precision_eff = max(precision_by_lag.values())
        admit_budget = 0
        admit_selected = 0
        miss_requests = int(slot_cache_misses)
        total_miss_requests += miss_requests
        miss_rate = float(miss_requests / slot_cache_requests) if slot_cache_requests > 0 else 0.0
        jsd_freq = 0.0
        overlap_w = 0.0
        drift_raw = 0.0
        drift_norm = 0.0
        alpha_for_slot = admission_capacity_alpha
        capacity_scale = 1.0
        fill_phase = False
        fill_min_budget = 0
        pressure_mult = 1.0
        quality_mult = 1.0
        score_spread = 0.0

        if is_cache_slot:
            total_cache_slots += 1
            top_k = int(label_info.get("k_actual", 0))
            curr_topk_dist: Optional[Dict[str, float]] = None
            if top_k > 0:
                top_k = min(top_k, len(D_t))
                top_freq_total = sum(int(row["freq"]) for row in D_t[:top_k])
                if top_freq_total > 0:
                    curr_topk_dist = {
                        row["object_id"]: float(row["freq"] / top_freq_total)
                        for row in D_t[:top_k]
                    }
            jsd_freq = _compute_jsd_identity(prev_topk_dist, curr_topk_dist)
            overlap_w = _weighted_jaccard(prev_topk_dist, curr_topk_dist)
            drift_raw = DRIFT_WEIGHT_JSD * jsd_freq + DRIFT_WEIGHT_OVERLAP * (1.0 - overlap_w)
            drift_ema = DRIFT_EMA_ALPHA * drift_raw + (1.0 - DRIFT_EMA_ALPHA) * drift_ema
            if drift_stats_count == 0:
                drift_mean_ema = drift_ema
                drift_var_ema = 0.0
            else:
                drift_mean_ema = (
                    (1.0 - DRIFT_STATS_ALPHA) * drift_mean_ema
                    + DRIFT_STATS_ALPHA * drift_ema
                )
                diff = drift_ema - drift_mean_ema
                drift_var_ema = (
                    (1.0 - DRIFT_STATS_ALPHA) * drift_var_ema
                    + DRIFT_STATS_ALPHA * diff * diff
                )
            drift_stats_count += 1
            drift_std = math.sqrt(drift_var_ema) if drift_var_ema > 0.0 else 0.0
            z = (drift_ema - drift_mean_ema) / (drift_std + DRIFT_NORM_EPS)
            if z < -DRIFT_Z_CLIP:
                z = -DRIFT_Z_CLIP
            elif z > DRIFT_Z_CLIP:
                z = DRIFT_Z_CLIP
            drift_norm = 1.0 / (1.0 + math.exp(-z))
            prev_topk_dist = curr_topk_dist
            capacity_scale = 1.0
            slot_unique = int(label_info.get("n_objects", 0))
            if slot_unique > 0:
                ratio = capacity_objects / slot_unique
                if ratio < CAPACITY_ALPHA_SCALE_MIN:
                    ratio = CAPACITY_ALPHA_SCALE_MIN
                elif ratio > 1.0:
                    ratio = 1.0
                capacity_scale = ratio
            fill_threshold = int(math.ceil(FILL_RATIO * capacity_objects))
            if len(cache) < fill_threshold:
                fill_phase = True
                fill_min_budget = int(math.ceil(FILL_RATE * capacity_objects))

            alpha_base = admission_capacity_alpha * (1.0 + DRIFT_GAIN * drift_norm) * capacity_scale
            if alpha_base < DRIFT_ALPHA_MIN:
                alpha_base = DRIFT_ALPHA_MIN
            elif alpha_base > DRIFT_ALPHA_MAX:
                alpha_base = DRIFT_ALPHA_MAX
            precision_adjust = 1.0
            guard_precision = precision_eff if precision_eff is not None else admit_precision
            if guard_precision is not None:
                precision_delta = (
                    (guard_precision - ADMISSION_PRECISION_TARGET)
                    / max(ADMISSION_PRECISION_TARGET, 1e-12)
                )
                if precision_delta > 0.0:
                    precision_adjust = 1.0 + ADMISSION_PRECISION_SENSITIVITY * precision_delta
            alpha_candidate = alpha_base * precision_adjust
            alpha_floor = alpha_base * DRIFT_ALPHA_FLOOR_MULT
            alpha_for_slot = alpha_candidate if alpha_candidate >= alpha_floor else alpha_floor
            if alpha_for_slot < DRIFT_ALPHA_MIN:
                alpha_for_slot = DRIFT_ALPHA_MIN
            elif alpha_for_slot > DRIFT_ALPHA_MAX:
                alpha_for_slot = DRIFT_ALPHA_MAX

        if is_cache_slot:
            drift_raw_sum += drift_raw
            drift_norm_sum += drift_norm
            jsd_sum += jsd_freq
            overlap_sum += overlap_w
            miss_rate_sum += miss_rate
            alpha_sum += alpha_for_slot
            if alpha_min_seen is None or alpha_for_slot < alpha_min_seen:
                alpha_min_seen = alpha_for_slot
            if alpha_max_seen is None or alpha_for_slot > alpha_max_seen:
                alpha_max_seen = alpha_for_slot
        candidate_scores: Dict[str, float] = {}
        candidate_ids: List[str] = []
        if slot_miss_features:
            X_miss = np.asarray(slot_miss_features, dtype=float)
            miss_scores = il_model.score_batch(X_miss)
            for obj_id, score in zip(slot_miss_obj_ids, miss_scores):
                prev = candidate_scores.get(obj_id)
                score_val = float(score)
                if prev is None:
                    candidate_scores[obj_id] = score_val
                    candidate_ids.append(obj_id)
                elif score_val > prev:
                    candidate_scores[obj_id] = score_val
        miss_candidates = int(len(candidate_ids))
        total_miss_candidates += miss_candidates

        candidate_score_list = [candidate_scores[obj_id] for obj_id in candidate_ids]
        if miss_candidates > 0:
            scores = np.asarray(candidate_score_list, dtype=float)
            p50 = float(np.quantile(scores, 0.5))
            pq = float(np.quantile(scores, SCORE_SPREAD_Q))
            score_spread = pq - p50
            if score_spread_ema is None:
                score_spread_ema = score_spread
            else:
                score_spread_ema = (
                    SCORE_SPREAD_EMA_ALPHA * score_spread
                    + (1.0 - SCORE_SPREAD_EMA_ALPHA) * score_spread_ema
                )
            denom = (score_spread_ema if score_spread_ema is not None else 0.0) + SCORE_SPREAD_EPS
            quality_mult = score_spread / denom
            effective_quality_min = SCORE_QUALITY_MIN
            if capacity_scale < 1.0:
                boost = (1.0 - capacity_scale) * SCORE_QUALITY_MIN_BOOST
                effective_quality_min = min(SCORE_QUALITY_MAX, SCORE_QUALITY_MIN + boost)
            if quality_mult < effective_quality_min:
                quality_mult = effective_quality_min
            elif quality_mult > SCORE_QUALITY_MAX:
                quality_mult = SCORE_QUALITY_MAX
            score_spread_sum += score_spread
            score_spread_count += 1
            quality_mult_sum += quality_mult
            quality_mult_count += 1
        score_eval_n = miss_candidates
        score_eval_pos = 0
        score_auc = float("nan")
        score_ap = float("nan")
        score_precK = float("nan")
        score_acc = float("nan")
        if miss_candidates > 0:
            label_map = {row["object_id"]: int(row["y"]) for row in D_t}
            y_miss = [label_map.get(obj_id, 0) for obj_id in candidate_ids]
            score_eval_pos = int(sum(y_miss))
            if score_eval_pos > 0 and score_eval_pos < miss_candidates:
                score_auc = _auc_roc(y_miss, candidate_score_list)
            if score_eval_pos > 0:
                score_ap = _avg_precision(y_miss, candidate_score_list)
                score_precK = _precision_at_k(y_miss, candidate_score_list, score_eval_pos)
            y_pred = [1 if s >= 0.5 else 0 for s in candidate_score_list]
            score_acc = float(
                sum(1 for yp, yt in zip(y_pred, y_miss) if yp == yt) / miss_candidates
            )
            if score_eval_pos > 0:
                score_eval_slots += 1
                score_eval_n_sum += miss_candidates
                score_eval_pos_sum += score_eval_pos
                if score_auc == score_auc:
                    score_auc_sum += score_auc
                    score_auc_count += 1
                if score_ap == score_ap:
                    score_ap_sum += score_ap
                    score_ap_count += 1
                if score_precK == score_precK:
                    score_precK_sum += score_precK
                    score_precK_count += 1
            if score_acc == score_acc:
                score_acc_sum += score_acc
                score_acc_count += 1

        if is_cache_slot:
            pressure_mult = (1.0 + PRESSURE_MISS_GAMMA * miss_rate) * quality_mult
            pressure_mult_sum += pressure_mult
            pressure_mult_count += 1
            if fill_phase:
                fill_phase_slots += 1
                fill_min_budget_sum += fill_min_budget
        admit_budget = _compute_admission_budget(
            slot_cache_misses,
            capacity_objects,
            alpha_for_slot,
            pressure_mult,
        )
        if fill_phase and admit_budget < fill_min_budget:
            admit_budget = fill_min_budget
        if miss_candidates <= 0:
            admit_budget = 0
        elif admit_budget > miss_candidates:
            admit_budget = miss_candidates
        score_gate_k = 0
        score_gate_applied = False
        if miss_candidates > 0:
            score_gate_k = max(1, int(math.ceil(SCORE_GATE_TOP_PERCENT * miss_candidates)))
            score_gate_k_sum += score_gate_k
            if not fill_phase and admit_budget > score_gate_k:
                admit_budget = score_gate_k
                score_gate_applied = True
                score_gate_applied_slots += 1
        total_admit_budget += admit_budget
        next_admit_set = _select_top_m(candidate_ids, candidate_score_list, admit_budget)
        admit_selected = len(next_admit_set)
        slot_admit_requests = admit_selected
        slot_reject_requests = max(miss_candidates - admit_selected, 0)
        total_admit_requests += slot_admit_requests
        total_reject_requests += slot_reject_requests
        total_admit_selected += admit_selected
        pending_admit_set = next_admit_set
        pending_admit_source_slot = slot_num

        if admit_true_applied is not None:
            total_admit_true_popular += admit_true_applied
        if precision_eff is not None:
            precision_eff_sum += precision_eff
            precision_eff_count += 1

        update_time_s = 0.0
        update_start = time.perf_counter()
        il_model.update_slot(D_t)
        update_time_s = time.perf_counter() - update_start
        total_update_time_s += update_time_s
        total_update_slots += 1

        warmup_slots = warmup_requests // slot_size
        phase = "warmup" if slot_num <= warmup_slots else "cache"
        current_slot_hr = (
            float(slot_cache_hits / slot_cache_requests)
            if slot_cache_requests > 0
            else None
        )
        if (
            is_cache_slot
            and prev_score_precK is not None
            and current_slot_hr is not None
            and prev_score_precK == prev_score_precK
        ):
            score_precK_hr_next_n += 1
            score_precK_hr_next_sum_x += float(prev_score_precK)
            score_precK_hr_next_sum_y += float(current_slot_hr)
            score_precK_hr_next_sum_x2 += float(prev_score_precK) ** 2
            score_precK_hr_next_sum_y2 += float(current_slot_hr) ** 2
            score_precK_hr_next_sum_xy += float(prev_score_precK) * float(current_slot_hr)
        if is_cache_slot:
            prev_score_precK = score_precK

        slot_log = {
            "slot_index": slot_num,
            "phase": phase,
            "slot_cache_requests": slot_cache_requests,
            "slot_cache_hits": slot_cache_hits,
            "slot_cache_misses": slot_cache_misses,
            "slot_hit_ratio": (
                current_slot_hr
            ),
            "hit_ratio": stats.hit_ratio,
            "miss_requests": miss_requests,
            "miss_rate": miss_rate,
            "miss_candidates": miss_candidates,
            "fill_phase": fill_phase,
            "fill_min_budget": fill_min_budget,
            "pressure_mult": pressure_mult,
            "score_spread": score_spread,
            "score_spread_ema": score_spread_ema,
            "quality_mult": quality_mult,
            "score_eval_n": score_eval_n,
            "score_eval_pos": score_eval_pos,
            "score_auc": score_auc,
            "score_ap": score_ap,
            "score_precK": score_precK,
            "score_acc": score_acc,
            "admission_precision_lag0": precision_lag0,
            "admission_precision_lag1": precision_lag1,
            "admission_precision_lag2": precision_lag2,
            "admission_precision_eff": precision_eff,
            "score_gate_k": score_gate_k,
            "score_gate_applied": score_gate_applied,
            "admit_budget": admit_budget,
            "admit_selected": admit_selected,
            "admit_applied": len(applied_admit_set),
            "admit_applied_from_slot": applied_admit_source_slot,
            "admit_true_from_applied": admit_true_applied,
            "admission_precision": admit_precision,
            "jsd_freq": jsd_freq,
            "topk_overlap_w": overlap_w,
            "drift_raw": drift_raw,
            "drift_ema": drift_ema,
            "drift_norm": drift_norm,
            "admission_alpha": alpha_for_slot,
            "capacity_scale": capacity_scale,
            "update_time_s": float(update_time_s),
        }
        _write_slot_log(slot_log)

        # update rolling history for next slot
        _update_freq_history(
            slot_num,
            slot_stats,
            freq_history,
            cum_counts,
            FEATURE_WINDOWS["long"],
        )

        # finalize pollution for expired slot
        expired = slot_num - POLLUTION_WINDOW_SLOTS
        if expired >= 1:
            polluted, total = _finalize_pollution(expired)
            total_pollution_count += polluted
            total_pollution_total += total
            _prune_active_slots(expired)
            last_pollution_finalized = expired

        # reset per-slot state
        slot_stats = {}
        slot_cache_requests = 0
        slot_cache_hits = 0
        slot_cache_misses = 0
        slot_admit_requests = 0
        slot_reject_requests = 0
        slot_miss_obj_ids = []
        slot_miss_features = []
        applied_admit_set = set()
        applied_admit_source_slot = None

    try:
        while True:
            try:
                req = next(req_iter)
            except StopIteration:
                # flush slot terakhir jika ada
                if slot_stats:
                    _finalize_slot()
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
                info = slot_stats[obj_id]
            else:
                info["freq"] += 1
                info["last_gaps"] = gaps

            history_feats = _get_history_features(
                obj_id,
                slot_index,
                freq_history,
                cum_counts,
                FEATURE_WINDOWS,
            )
            features = _build_feature_vector(
                feature_set,
                gaps,
                history_feats,
            )
            # features dipakai untuk keputusan admission saat ini (policy-time)

            # 2) fase caching (setelah warm-up selesai)
            if global_idx >= warmup_requests:
                stats.total_requests += 1
                slot_cache_requests += 1
                if cache.access(obj_id):
                    stats.cache_hits += 1
                    slot_cache_hits += 1
                    _mark_admit_hit(obj_id)
                    if obj_id in applied_admit_set:
                        total_hits_from_admitted += 1
                else:
                    slot_cache_misses += 1
                    slot_miss_obj_ids.append(obj_id)
                    slot_miss_features.append(features)

            # 3) update counter & cek boundary slot
            global_idx += 1
            slot_req_count += 1

            if slot_req_count >= slot_size:
                _finalize_slot()

                next_slot_start = slot_index * slot_size
                if next_slot_start < total_requests and next_slot_start >= warmup_requests:
                    _apply_pending_admits(slot_index + 1)
                else:
                    pending_admit_set = set()
                    pending_admit_source_slot = None

                pbar.update(1)
                current_hr = stats.hit_ratio if stats.total_requests > 0 else 0.0
                pbar.set_postfix(hr=f"{current_hr:.4f}")
                slot_req_count = 0
    finally:
        if slot_log_file is not None:
            slot_log_file.close()
        pbar.close()

    # finalize pollution for remaining slots
    for slot_num in range(last_pollution_finalized + 1, slot_index + 1):
        polluted, total = _finalize_pollution(slot_num)
        total_pollution_count += polluted
        total_pollution_total += total

    score_precK_hr_next_corr = None
    if score_precK_hr_next_n > 1:
        denom = (
            (score_precK_hr_next_n * score_precK_hr_next_sum_x2)
            - (score_precK_hr_next_sum_x ** 2)
        ) * (
            (score_precK_hr_next_n * score_precK_hr_next_sum_y2)
            - (score_precK_hr_next_sum_y ** 2)
        )
        if denom > 0.0:
            score_precK_hr_next_corr = (
                (score_precK_hr_next_n * score_precK_hr_next_sum_xy)
                - (score_precK_hr_next_sum_x * score_precK_hr_next_sum_y)
            ) / math.sqrt(denom)

    summary_metrics = {
        "feature_set": feature_set,
        "num_features": num_features,
        "slots_processed": total_slots_processed,
        "cache_slots_processed": total_cache_slots,
        "update_slots": total_update_slots,
        "miss_requests_total": total_miss_requests,
        "miss_candidates_total": total_miss_candidates,
        "score_eval_slots": score_eval_slots,
        "score_eval_n_avg": (
            float(score_eval_n_sum / score_eval_slots) if score_eval_slots > 0 else None
        ),
        "score_eval_pos_avg": (
            float(score_eval_pos_sum / score_eval_slots) if score_eval_slots > 0 else None
        ),
        "score_auc_avg": (
            float(score_auc_sum / score_auc_count) if score_auc_count > 0 else None
        ),
        "score_ap_avg": (
            float(score_ap_sum / score_ap_count) if score_ap_count > 0 else None
        ),
        "score_precK_avg": (
            float(score_precK_sum / score_precK_count) if score_precK_count > 0 else None
        ),
        "score_acc_avg": (
            float(score_acc_sum / score_acc_count) if score_acc_count > 0 else None
        ),
        "score_precK_hr_next_corr": score_precK_hr_next_corr,
        "score_precK_hr_next_n": score_precK_hr_next_n,
        "score_gate_k_avg": (
            float(score_gate_k_sum / total_cache_slots) if total_cache_slots > 0 else None
        ),
        "score_gate_applied_slots": score_gate_applied_slots,
        "fill_phase_slots": fill_phase_slots,
        "fill_min_budget_avg": (
            float(fill_min_budget_sum / fill_phase_slots) if fill_phase_slots > 0 else None
        ),
        "pressure_mult_avg": (
            float(pressure_mult_sum / pressure_mult_count)
            if pressure_mult_count > 0
            else None
        ),
        "score_spread_avg": (
            float(score_spread_sum / score_spread_count)
            if score_spread_count > 0
            else None
        ),
        "quality_mult_avg": (
            float(quality_mult_sum / quality_mult_count)
            if quality_mult_count > 0
            else None
        ),
        "admission_precision_eff_avg": (
            float(precision_eff_sum / precision_eff_count)
            if precision_eff_count > 0
            else None
        ),
        "score_auc_count": score_auc_count,
        "score_ap_count": score_ap_count,
        "score_precK_count": score_precK_count,
        "score_acc_count": score_acc_count,
        "admit_budget_total": total_admit_budget,
        "admit_requests_total": total_admit_requests,
        "reject_requests_total": total_reject_requests,
        "admit_selected_total": total_admit_selected,
        "admit_applied_total": total_admit_applied,
        "admit_true_popular_total": total_admit_true_popular,
        "hits_from_admitted_total": total_hits_from_admitted,
        "pollution_total": total_pollution_total,
        "pollution_count": total_pollution_count,
        "admission_precision_total": (
            float(total_admit_true_popular / total_admit_applied)
            if total_admit_applied > 0
            else None
        ),
        "hit_yield_total": (
            float(total_hits_from_admitted / total_admit_applied)
            if total_admit_applied > 0
            else None
        ),
        "pollution_rate_total": (
            float(total_pollution_count / total_pollution_total)
            if total_pollution_total > 0
            else None
        ),
        "miss_rate_avg": (
            float(miss_rate_sum / total_cache_slots) if total_cache_slots > 0 else None
        ),
        "jsd_freq_avg": (
            float(jsd_sum / total_cache_slots) if total_cache_slots > 0 else None
        ),
        "topk_overlap_w_avg": (
            float(overlap_sum / total_cache_slots) if total_cache_slots > 0 else None
        ),
        "drift_raw_avg": (
            float(drift_raw_sum / total_cache_slots) if total_cache_slots > 0 else None
        ),
        "drift_norm_avg": (
            float(drift_norm_sum / total_cache_slots) if total_cache_slots > 0 else None
        ),
        "admission_alpha_avg": (
            float(alpha_sum / total_cache_slots) if total_cache_slots > 0 else None
        ),
        "admission_alpha_min": alpha_min_seen,
        "admission_alpha_max": alpha_max_seen,
        "avg_update_time_s": (
            float(total_update_time_s / total_update_slots)
            if total_update_slots > 0
            else None
        ),
        "update_time_total_s": float(total_update_time_s),
    }

    return stats, summary_metrics


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
        if not fname.endswith(f"_{model}_{cache_size}.jsonl") and \
           not fname.endswith(f"_summary_{model}_{cache_size}.json"):
            continue
        prefix = fname.split("_", 1)[0]
        if len(prefix) == 3 and prefix.isdigit():
            existing_ids.append(int(prefix))

    next_id = 1 if not existing_ids else max(existing_ids) + 1
    run_id = f"{next_id:03d}"
    return run_id, dataset_dir


def get_next_group_id(results_root: str, dataset: str, model: str) -> Tuple[str, str]:
    """
    Menentukan ID run berikutnya untuk summary lintas cache size.
    """
    dataset_dir = os.path.join(results_root, dataset)
    os.makedirs(dataset_dir, exist_ok=True)

    existing_ids: List[int] = []
    suffix = f"_summary_{model}_all_sizes.json"
    for fname in os.listdir(dataset_dir):
        if not fname.endswith(suffix):
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
# Main: TANPA argparse, semua pakai konstanta lokal
# ---------------------------------------------------------------------------

def run_experiment(
    ds_cfg: Dict[str, Any],
    feature_set: str = "A0",
    model_name: Optional[str] = None,
    cache_size_percentages: Tuple[float, ...] = CACHE_SIZE_PERCENTAGES,
) -> None:
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"Unknown feature_set: {feature_set}")
    if model_name is None:
        model_name = f"ilnse_{feature_set}"
    label_topk_rounding = IL_LABEL_TOPK_ROUNDING
    label_tie_break = IL_LABEL_TIE_BREAK

    trace_path = ds_cfg["path"]                # path ke .gz (seperti di debug)
    slot_size = ds_cfg["slot_size"]            # misal 100000
    warmup_requests = ds_cfg["num_warmup_requests"]  # misal 1_000_000
    # asumsi: total request juga sudah ada di config
    # kalau nama field beda (misal ds_cfg.total_requests), ganti baris ini
    total_requests = ds_cfg["num_total_requests"]

    if warmup_requests % slot_size != 0:
        raise ValueError(
            f"warmup_requests ({warmup_requests}) harus kelipatan slot_size ({slot_size}) "
            "agar sesuai dengan setup eksperimen Xu."
        )

    dataset_name = ds_cfg["name"]     # misal "WIKI2018"
    results_root = "results"

    print(f"=== IL-based Edge Cache Experiment ({dataset_name} trace) ===")
    print(f"Trace path        : {trace_path}")
    print(f"Total requests    : {total_requests}")
    print(f"Warm-up requests  : {warmup_requests}")
    print(f"Slot size         : {slot_size}")
    print(f"IL num_gaps       : {IL_NUM_GAPS}")
    print(f"Feature set       : {feature_set}")
    print(f"Num features      : {_get_feature_dim(IL_NUM_GAPS, feature_set)}")
    print(f"IL top_percent    : {IL_POP_TOP_PERCENT}")
    print(f"Label rounding    : {label_topk_rounding}")
    print(f"Label tie-break   : {label_tie_break}")
    print(f"IL sigmoid (a, b) : ({IL_SIGMOID_A}, {IL_SIGMOID_B})")
    print(f"Max learners      : {IL_MAX_CLASSIFIERS}")
    print("Admission policy  : top_capacity_rate")
    print(f"Admission alpha   : {ADMISSION_CAPACITY_ALPHA}")
    print(f"Fill ratio/rate   : {FILL_RATIO} / {FILL_RATE}")
    print(f"Pressure gamma   : {PRESSURE_MISS_GAMMA}")
    print(
        "Quality spread   : q="
        f"{SCORE_SPREAD_Q}, ema={SCORE_SPREAD_EMA_ALPHA}, "
        f"min={SCORE_QUALITY_MIN}, max={SCORE_QUALITY_MAX}, "
        f"min_boost={SCORE_QUALITY_MIN_BOOST}"
    )
    print(f"Drift alpha range : {DRIFT_ALPHA_MIN}–{DRIFT_ALPHA_MAX}")
    print(f"Drift sensitivity: {DRIFT_SENSITIVITY}")
    print(f"Drift EMA alpha  : {DRIFT_EMA_ALPHA}")
    print(
        f"Drift weights    : jsd={DRIFT_WEIGHT_JSD}, overlap={DRIFT_WEIGHT_OVERLAP}"
    )
    print(f"Precision target  : {ADMISSION_PRECISION_TARGET}")
    print(f"Cache capacities  : {list(cache_size_percentages)}")
    print()

    # Hitung kapasitas dinamis
    capacities_objects = get_dynamic_capacities(trace_path, total_requests, list(cache_size_percentages))

    results: List[CacheStats] = []

    for capacity in capacities_objects:
        print(f"[RUN] Capacity Objects={capacity}")

        run_id, dataset_dir = get_next_run_id(results_root, dataset_name, model_name, capacity)
        per_slot_path = os.path.join(
            dataset_dir,
            f"{run_id}_{model_name}_{capacity}.jsonl",
        )
        summary_path = os.path.join(
            dataset_dir,
            f"{run_id}_summary_{model_name}_{capacity}.json",
        )

        stats, summary_metrics = run_single_capacity(
            trace_path=trace_path,
            total_requests=total_requests,
            warmup_requests=warmup_requests,
            slot_size=slot_size,
            capacity_objects=capacity,
            feature_set=feature_set,
            slot_log_path=per_slot_path,
        )
        results.append(stats)

        slots_processed = int(summary_metrics.get("slots_processed", 0))
        warmup_slots = warmup_requests // slot_size
        cache_slots = max(0, slots_processed - warmup_slots)

        summary_payload = {
            "dataset": dataset_name,
            "model": model_name,
            "feature_set": feature_set,
            "cache_size_objects": capacity,
            "total_requests": total_requests,
            "warmup_requests": warmup_requests,
            "slot_size": slot_size,
            "slot_log_path": per_slot_path,
            "hit_ratio": stats.hit_ratio,
            "cache_hits": stats.cache_hits,
            "cache_requests": stats.total_requests,
            "num_slots": slots_processed,
            "warmup_slots": warmup_slots,
            "cache_slots": cache_slots,
            "pop_top_percent": IL_POP_TOP_PERCENT,
            "label_topk_rounding": label_topk_rounding,
            "label_tie_break": label_tie_break,
            "admission_policy": "top_capacity_rate",
            "admission_capacity_alpha": ADMISSION_CAPACITY_ALPHA,
            "drift_alpha_min": DRIFT_ALPHA_MIN,
            "drift_alpha_max": DRIFT_ALPHA_MAX,
            "drift_sensitivity": DRIFT_SENSITIVITY,
            "drift_ema_alpha": DRIFT_EMA_ALPHA,
            "drift_weight_jsd": DRIFT_WEIGHT_JSD,
            "drift_weight_overlap": DRIFT_WEIGHT_OVERLAP,
            "drift_gain": DRIFT_GAIN,
            "drift_alpha_floor_mult": DRIFT_ALPHA_FLOOR_MULT,
            "admission_precision_target": ADMISSION_PRECISION_TARGET,
            "admission_precision_sensitivity": ADMISSION_PRECISION_SENSITIVITY,
            "capacity_alpha_scale_min": CAPACITY_ALPHA_SCALE_MIN,
            "feature_windows": FEATURE_WINDOWS,
            "pollution_window_slots": POLLUTION_WINDOW_SLOTS,
            "hit_contribution": {
                "hits_from_admitted": summary_metrics.get("hits_from_admitted_total"),
                "reject_requests_total": summary_metrics.get("reject_requests_total"),
            },
        }
        summary_payload.update(summary_metrics)
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

    hr_curve = [
        {"cache_size_objects": cap, "hit_ratio": st.hit_ratio}
        for cap, st in zip(capacities_objects, results)
    ]
    avg_hr = float(np.mean([st.hit_ratio for st in results])) if results else None

    group_id, dataset_dir = get_next_group_id(results_root, dataset_name, model_name)
    overall_path = os.path.join(
        dataset_dir,
        f"{group_id}_summary_{model_name}_all_sizes.json",
    )
    overall_payload = {
        "dataset": dataset_name,
        "model": model_name,
        "feature_set": feature_set,
        "num_features": _get_feature_dim(IL_NUM_GAPS, feature_set),
        "cache_sizes": capacities_objects,
        "hr_curve": hr_curve,
        "avg_hr": avg_hr,
    }
    save_json(overall_path, overall_payload)


def main() -> None:
    run_experiment(WIKIPEDIA_SEPTEMBER_2007)


if __name__ == "__main__":
    main()
# 1185
