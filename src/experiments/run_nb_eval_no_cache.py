# src/experiments/run_nb_eval_no_cache.py

from __future__ import annotations

import os, sys, json, math
from collections import deque
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config.experiment_config import WIKIPEDIA_SEPTEMBER_2007, ILConfig
from src.data.trace_reader import TraceReader
from src.data.feature_table import FeatureTable
from src.ml.learn_nse import LearnNSE, GaussianNaiveBayes, GaussianNaiveBayesMissingAware

FEATURE_SETS = ("A0", "A1", "A2")
FEATURE_WINDOWS = {"short": 1, "mid": 7, "long": 30}

# -----------------------------
# Feature helpers (A0/A1/A2)
# -----------------------------
def _get_feature_dim(num_gaps: int, feature_set: str) -> int:
    if feature_set == "A0":
        return num_gaps
    if feature_set == "A1":
        return num_gaps + 1
    if feature_set == "A2":
        return num_gaps + 4
    raise ValueError(f"Unknown feature_set: {feature_set}")


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
    gaps: List[float],
    feature_set: str,
    num_gaps: int,
    missing_gap_value: float,
    f_short: int,
    f_mid: int,
    f_long: int,
    f_cum: int,
) -> List[float]:
    g = list(gaps)
    if len(g) < num_gaps:
        g = g + [missing_gap_value] * (num_gaps - len(g))
    else:
        g = g[:num_gaps]
    if feature_set == "A0":
        return g
    if feature_set == "A1":
        return g + [float(f_short)]
    if feature_set == "A2":
        return g + [float(f_short), float(f_mid), float(f_long), float(f_cum)]
    raise ValueError(f"Unknown feature_set: {feature_set}")


def _update_freq_history(
    slot_index: int,
    slot_stats: Dict[str, Dict[str, Any]],
    freq_history: Dict[str, deque[Tuple[int, int]]],
    cum_counts: Dict[str, int],
    long_window: int,
) -> None:
    for obj_id, stats in slot_stats.items():
        freq = int(stats.get("freq", 0))
        if freq <= 0:
            continue
        hist = freq_history.get(obj_id)
        if hist is None:
            hist = deque()
            freq_history[obj_id] = hist
        hist.append((slot_index, freq))
        # trim old history for efficiency
        min_slot = max(1, slot_index - long_window + 1)
        while hist and hist[0][0] < min_slot:
            hist.popleft()
        cum_counts[obj_id] = int(cum_counts.get(obj_id, 0)) + freq

# -----------------------------
# Dataset per slot (per objek)
# -----------------------------
def build_slot_dataset_from_stats(
    slot_stats: Dict[str, Dict[str, Any]],
    top_ratio: float,
    effective_num_gaps: int,
    missing_gap_value: float,
    label_mode: str,
    coverage_target: float,
    label_capacity_gamma: float,
    capacity_objects: Optional[int],
    slot_index: int,
    feature_set: str,
    freq_history: Dict[str, deque[Tuple[int, int]]],
    cum_counts: Dict[str, int],
    windows: Dict[str, int],
    slot_size: int,
    gaps_key: str,  # "first_gaps" atau "last_gaps"
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
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

    items: List[Tuple[str, Dict[str, Any]]] = list(slot_stats.items())
    items.sort(key=lambda kv: kv[1]["freq"], reverse=True)

    n_objects = len(items)
    label_info["n_objects"] = int(n_objects)
    total_reqs = sum(int(stats["freq"]) for _, stats in items)
    top_ids: set = set()

    if label_mode == "coverage":
        if coverage_target <= 0.0 or coverage_target > 1.0:
            raise ValueError("coverage_target must be in (0, 1].")
        cum = 0
        for obj_id, stats in items:
            cum += int(stats["freq"])
            top_ids.add(obj_id)
            if total_reqs > 0 and (cum / float(total_reqs)) >= coverage_target:
                break
    elif label_mode == "top_ratio":
        k = int(n_objects * top_ratio)
        k = max(1, min(k, n_objects))
        top_ids = {items[i][0] for i in range(k)}
    elif label_mode == "capacity":
        if capacity_objects is None or capacity_objects <= 0:
            capacity_objects = int(round(float(top_ratio) * float(n_objects)))
            capacity_objects = max(1, capacity_objects)
        k = int(round(float(capacity_objects) * float(label_capacity_gamma)))
        k = max(1, min(k, n_objects))
        label_info["capacity_k"] = int(k)
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

    dataset: List[Dict[str, Any]] = []
    for obj_id, stats in items:
        gaps = stats[gaps_key]
        freq = int(stats["freq"])
        f_short, f_mid, f_long, f_cum = _get_history_features(
            obj_id,
            slot_index,
            freq_history,
            cum_counts,
            windows,
        )
        x = _build_feature_vector(
            gaps=gaps,
            feature_set=feature_set,
            num_gaps=effective_num_gaps,
            missing_gap_value=missing_gap_value,
            f_short=f_short,
            f_mid=f_mid,
            f_long=f_long,
            f_cum=f_cum,
        )

        dataset.append(
            {
                "obj_id": obj_id,
                "x": x,
                "y": 1 if obj_id in top_ids else 0,
                "freq": freq,
                "hist_len": int(stats.get("hist_len", 0)),
            }
        )
    return dataset, label_info

def to_nan(X: np.ndarray, missing_gap_value: float) -> np.ndarray:
    X = np.asarray(X, dtype=float).copy()
    X[X == float(missing_gap_value)] = np.nan
    return X

def dataset_to_nan(dataset: List[Dict[str, Any]], missing_gap_value: float) -> List[Dict[str, Any]]:
    mg = float(missing_gap_value)
    out = []
    for r in dataset:
        x = np.asarray(r["x"], dtype=float)
        x[x == mg] = np.nan
        r2 = dict(r)
        r2["x"] = x.tolist()
        out.append(r2)
    return out

# -----------------------------
# Metrics (tanpa sklearn)
# -----------------------------
def auc_roc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    pos = y_true == 1
    n_pos = int(pos.sum())
    n_neg = int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(y_score) + 1, dtype=float)

    # average ranks for ties
    s = y_score[order]
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

def avg_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    n_pos = int((y_true == 1).sum())
    if n_pos == 0:
        return float("nan")

    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    tp = np.cumsum(y_sorted == 1)
    denom = np.arange(1, len(y_sorted) + 1)
    precision = tp / denom
    return float((precision[y_sorted == 1]).sum() / n_pos)

def recall_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    n_pos = int((y_true == 1).sum())
    if n_pos == 0 or k <= 0:
        return float("nan")
    k = min(k, len(y_true))
    idx = np.argsort(-y_score)[:k]
    return float((y_true[idx] == 1).sum() / n_pos)

def k_from_budget(n: int, budget) -> int:
    """
    budget bisa:
    - float (0..1): fraksi dari jumlah objek
    - int: jumlah kandidat fixed
    """
    if n <= 0:
        return 0
    if isinstance(budget, float):
        k = int(math.ceil(budget * n))
    else:
        k = int(budget)
    return max(1, min(k, n))

def precision_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    if k <= 0:
        return float("nan")
    k = min(k, len(y_true))
    idx = np.argsort(-y_score)[:k]
    return float((y_true[idx] == 1).mean())

def extract_obj_ts(req, prev_ts: Optional[float], global_idx: int) -> Tuple[str, float]:
    """
    Support TraceReader output:
    - dict: {"object_id": ..., "timestamp": ...} (atau tanpa timestamp)
    - tuple/list: (object_id, timestamp, ...)
    Timestamp dibuat non-decreasing agar konsisten dengan FeatureTable gap.
    """
    if isinstance(req, dict):
        obj_id = req["object_id"]
        ts_raw = req.get("timestamp", None)
        if ts_raw is None:
            ts = (prev_ts + 1.0) if prev_ts is not None else float(global_idx + 1)
        else:
            ts = float(ts_raw)
            if prev_ts is not None and ts < prev_ts:
                ts = prev_ts
        return obj_id, ts

    if isinstance(req, (list, tuple)):
        if len(req) < 1:
            raise ValueError(f"Bad request format (empty): {req}")
        obj_id = req[0]
        ts_raw = req[1] if len(req) > 1 else None
        if ts_raw is None:
            ts = (prev_ts + 1.0) if prev_ts is not None else float(global_idx + 1)
        else:
            ts = float(ts_raw)
            if prev_ts is not None and ts < prev_ts:
                ts = prev_ts
        return obj_id, ts

    raise TypeError(f"Unsupported request type: {type(req)} -> {req}")

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
    effective_num_gaps: int,
    use_freq_feature: bool,
    freq_feature_mode: str,
    slot_size: int,
    missing_gap_value: float,
) -> List[float]:
    # pastikan panjang gaps = effective_num_gaps (pad/truncate)
    g = list(gaps)
    if len(g) < effective_num_gaps:
        g = g + [missing_gap_value] * (effective_num_gaps - len(g))
    else:
        g = g[:effective_num_gaps]

    if use_freq_feature:
        g = g + [compute_freq_feature(freq, slot_size, freq_feature_mode)]
    return g

# -----------------------------
# Main eval loop (tanpa cache)
# -----------------------------
def run_nb_eval_no_cache(
    trace_path: str,
    total_requests: int,
    warmup_requests: int,
    slot_size: int,
    il_cfg: ILConfig,
    feature_set: str = "A0",
    results_root: str = "results",
    dataset_name: str = "dataset",
    dataset_base_name: Optional[str] = None,
):
    os.makedirs(results_root, exist_ok=True)
    base_name = dataset_base_name or dataset_name
    out_dir = os.path.join(results_root, f"{base_name}_evaluasi_no_cache")
    os.makedirs(out_dir, exist_ok=True)

    # run id sederhana
    existing = [f for f in os.listdir(out_dir) if f.endswith(".jsonl") and "_nbeval-" in f]
    run_id = f"{len(existing)+1:03d}"
    detail_path = os.path.join(out_dir, f"{run_id}_nbeval-{dataset_name}.jsonl")
    summary_path = os.path.join(out_dir, f"{run_id}_summary_nbeval-{dataset_name}.json")

    reader = TraceReader(path=trace_path, max_rows=total_requests)
    req_iter = reader.iter_requests()

    ft = FeatureTable(L=il_cfg.num_gaps, missing_gap_value=il_cfg.missing_gap_value)

    warmup_slots = (warmup_requests + slot_size - 1) // slot_size

    effective_num_gaps = int(getattr(il_cfg, "num_gaps", 10))
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"Unknown feature_set: {feature_set}")
    label_mode = str(getattr(il_cfg, "label_mode", "top_ratio"))
    coverage_target = float(getattr(il_cfg, "coverage_target", 0.6))
    label_capacity_gamma = float(getattr(il_cfg, "label_capacity_gamma", 1.0))
    admit_eval_h = int(getattr(il_cfg, "admit_eval_horizon_slots", 1))
    admit_eval_k = getattr(il_cfg, "admit_eval_k", None)
    admit_eval_scope = str(getattr(il_cfg, "admit_eval_scope", "all"))
    max_learners = int(getattr(il_cfg, "max_classifiers", 20))
    if max_learners != 20:
        print(f"[WARN] max_classifiers={max_learners} -> override ke 20 (prune ala Xu).")
        max_learners = 20
    n_features = _get_feature_dim(effective_num_gaps, feature_set)

    il_xu = LearnNSE(
        n_features=n_features,
        a=il_cfg.sigmoid_a,
        b=il_cfg.sigmoid_b,
        max_learners=max_learners,
        base_learner_factory=GaussianNaiveBayes,  # baseline Xu
    )

    il_ma = LearnNSE(
        n_features=n_features,
        a=il_cfg.sigmoid_a,
        b=il_cfg.sigmoid_b,
        max_learners=max_learners,
        base_learner_factory=GaussianNaiveBayesMissingAware,  # optim: NB skip NaN
    )


    slot_stats: Dict[str, Dict[str, Any]] = {}
    freq_history: Dict[str, deque[Tuple[int, int]]] = {}
    cum_counts: Dict[str, int] = {}
    slot_req_count = 0
    global_idx = 0
    slot_idx = 0

    nb_xu: Optional[GaussianNaiveBayes] = None
    nb_ma: Optional[GaussianNaiveBayesMissingAware] = None
    have_model = False

    def _empty_agg() -> Dict[str, Any]:
        return {
            "slots_eval": 0,
            "xu_ap": [], "ma_ap": [],
            "xu_auc": [], "ma_auc": [],
            "xu_precK": [], "ma_precK": [],
            "xu_ap_small": [], "ma_ap_small": [],
            "xu_precK_small": [], "ma_precK_small": [],
        }

    agg_main = _empty_agg()
    agg_warmup = _empty_agg()

    post_metrics = ["hit_share", "unique_hit_rate", "hits_per_admit", "admit_k"]

    def _empty_post_agg() -> Dict[str, Any]:
        return {
            "slots_eval": 0,
            "xu": {m: [] for m in post_metrics},
            "ma": {m: [] for m in post_metrics},
        }

    post_agg_main = _empty_post_agg()
    post_agg_warmup = _empty_post_agg()
    post_incomplete = 0
    pending_post: List[Dict[str, Any]] = []
    post_admit_path = os.path.join(out_dir, f"{run_id}_postadmit-{dataset_name}.jsonl")

    total_slots_expected = (total_requests + slot_size - 1) // slot_size
    pbar = tqdm(total=total_slots_expected, desc="NB-only eval", unit="slot")

    def _update_agg(agg: Dict[str, Any], eval_dict: Dict[str, Any]) -> None:
        if not eval_dict or eval_dict.get("n_objects", 0) == 0:
            return
        agg["slots_eval"] += 1
        xu = eval_dict.get("xu", {})
        ma = eval_dict.get("ma", {})
        agg["xu_ap"].append(xu.get("ap", float("nan")))
        agg["ma_ap"].append(ma.get("ap", float("nan")))
        agg["xu_auc"].append(xu.get("auc", float("nan")))
        agg["ma_auc"].append(ma.get("auc", float("nan")))
        agg["xu_precK"].append(xu.get("precK", float("nan")))
        agg["ma_precK"].append(ma.get("precK", float("nan")))

        seg = eval_dict.get("seg_hist_len<=1")
        if seg:
            xu_s = seg.get("xu", {})
            ma_s = seg.get("ma", {})
            agg["xu_ap_small"].append(xu_s.get("ap", float("nan")))
            agg["ma_ap_small"].append(ma_s.get("ap", float("nan")))
            agg["xu_precK_small"].append(xu_s.get("precK", float("nan")))
            agg["ma_precK_small"].append(ma_s.get("precK", float("nan")))

    def _score_stats(scores: np.ndarray) -> Dict[str, float]:
        if scores is None or len(scores) == 0:
            return {"mean": float("nan"), "std": float("nan"), "p10": float("nan"), "p50": float("nan"), "p90": float("nan")}
        scores = np.asarray(scores, dtype=float)
        return {
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "p10": float(np.percentile(scores, 10)),
            "p50": float(np.percentile(scores, 50)),
            "p90": float(np.percentile(scores, 90)),
        }

    def _weights_stats(weights: Optional[List[float]]) -> Dict[str, float]:
        if not weights:
            return {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
        w = np.asarray(weights, dtype=float)
        return {
            "mean": float(np.mean(w)),
            "std": float(np.std(w)),
            "min": float(np.min(w)),
            "max": float(np.max(w)),
        }

    def _scope_allows(is_warmup: bool) -> bool:
        if admit_eval_scope == "warmup_only":
            return is_warmup
        if admit_eval_scope in ("post_warmup", "after_warmup"):
            return not is_warmup
        return True

    def _update_post_agg(agg: Dict[str, Any], post_rec: Dict[str, Any]) -> None:
        agg["slots_eval"] += 1
        for model in ("xu", "ma"):
            md = post_rec.get(model, {})
            for m in post_metrics:
                agg[model][m].append(md.get(m, float("nan")))

    def _finalize_post_admit(rec: Dict[str, Any], horizon_complete: bool) -> None:
        nonlocal post_incomplete

        post_rec: Dict[str, Any] = {
            "slot_idx": int(rec["slot_idx"]),
            "is_warmup": bool(rec["is_warmup"]),
            "horizon_slots": int(admit_eval_h),
            "future_slots": int(rec["future_slots"]),
            "horizon_complete": bool(horizon_complete),
        }

        total_reqs = float(rec["future_total_reqs"])
        for model in ("xu", "ma"):
            admit_k = int(rec["admit_k"][model])
            hits = float(rec["future_hits"][model])
            unique_hits = int(len(rec["future_hit_objs"][model]))

            hit_share = hits / total_reqs if total_reqs > 0 else float("nan")
            unique_hit_rate = (unique_hits / float(admit_k)) if admit_k > 0 else float("nan")
            hits_per_admit = (hits / float(admit_k)) if admit_k > 0 else float("nan")

            post_rec[model] = {
                "admit_k": admit_k,
                "future_total_reqs": int(total_reqs),
                "future_hits": int(hits),
                "unique_hits": unique_hits,
                "hit_share": hit_share,
                "unique_hit_rate": unique_hit_rate,
                "hits_per_admit": hits_per_admit,
            }

        with open(post_admit_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(post_rec) + "\n")

        if horizon_complete:
            if rec["is_warmup"]:
                _update_post_agg(post_agg_warmup, post_rec)
            else:
                _update_post_agg(post_agg_main, post_rec)
        else:
            post_incomplete += 1

    def _update_pending_with_future(slot_stats: Dict[str, Dict[str, Any]]) -> None:
        nonlocal pending_post
        if not pending_post or admit_eval_h <= 0:
            return

        slot_freq = {obj_id: int(stats["freq"]) for obj_id, stats in slot_stats.items()}
        slot_total = sum(slot_freq.values())

        for rec in pending_post:
            rec["future_total_reqs"] += slot_total
            rec["future_slots"] += 1
            if slot_freq:
                slot_ids = set(slot_freq.keys())
                for model in ("xu", "ma"):
                    admit_set = rec["admit_sets"][model]
                    if not admit_set:
                        continue
                    hit_objs = admit_set & slot_ids
                    if hit_objs:
                        rec["future_hits"][model] += sum(slot_freq[obj] for obj in hit_objs)
                        rec["future_hit_objs"][model].update(hit_objs)
            rec["remaining"] -= 1

        done = [r for r in pending_post if r["remaining"] <= 0]
        pending_post = [r for r in pending_post if r["remaining"] > 0]
        for rec in done:
            _finalize_post_admit(rec, horizon_complete=True)

    def finalize_slot(
        slot_dataset_last: List[Dict[str, Any]],
        slot_dataset_first: List[Dict[str, Any]],
        is_warmup: bool,
        slot_stats: Dict[str, Dict[str, Any]],
        label_info_last: Dict[str, Any],
        label_info_first: Dict[str, Any],
    ):
        nonlocal slot_idx, global_idx

        if not slot_dataset_last:
            return

        _update_pending_with_future(slot_stats)

        def _eval_with_scores(view_name: str, slot_dataset: List[Dict[str, Any]]):
            if not slot_dataset:
                return {"view": view_name, "n_objects": 0}, [], None, None

            Xte_xu = np.asarray([r["x"] for r in slot_dataset], dtype=float)
            Xte_ma = to_nan(Xte_xu, il_cfg.missing_gap_value)
            yte = np.asarray([r["y"] for r in slot_dataset], dtype=int)
            hist = np.asarray([r["hist_len"] for r in slot_dataset], dtype=int)
            obj_ids = [r["obj_id"] for r in slot_dataset]

            p_xu = il_xu.score_batch(Xte_xu)
            p_ma = il_ma.score_batch(Xte_ma)

            k = int((yte == 1).sum())
            out = {
                "view": view_name,
                "n_objects": int(len(yte)),
                "n_pos": int(k),
                "xu": {"ap": avg_precision(yte, p_xu), "auc": auc_roc(yte, p_xu), "precK": precision_at_k(yte, p_xu, k)},
                "ma": {"ap": avg_precision(yte, p_ma), "auc": auc_roc(yte, p_ma), "precK": precision_at_k(yte, p_ma, k),
                    "nan_frac": float(np.isnan(Xte_ma).mean())},
                "score_stats": {
                    "xu": _score_stats(p_xu),
                    "ma": _score_stats(p_ma),
                },
            }

            small = hist <= 1
            if small.any():
                y_s = yte[small]
                k_s = int((y_s == 1).sum())
                out["seg_hist_len<=1"] = {
                    "n": int(small.sum()),
                    "n_pos": int(k_s),
                    "xu": {"ap": avg_precision(y_s, p_xu[small]), "precK": precision_at_k(y_s, p_xu[small], k_s)},
                    "ma": {"ap": avg_precision(y_s, p_ma[small]), "precK": precision_at_k(y_s, p_ma[small], k_s)},
                }
            return out, obj_ids, p_xu, p_ma

        # EVAL: last vs first (model state = hasil slot sebelumnya)
        eval_last, obj_ids, p_xu, p_ma = _eval_with_scores("last", slot_dataset_last)
        rec = {
            "slot_idx": int(slot_idx),
            "is_warmup": bool(is_warmup),
            "eval_last": eval_last,
            "eval_first": _eval_with_scores("first", slot_dataset_first)[0],
            "label_last": label_info_last,
            "label_first": label_info_first,
        }

        with open(detail_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

        if rec["eval_last"].get("n_objects", 0) > 0:
            if is_warmup:
                _update_agg(agg_warmup, rec["eval_last"])
            else:
                _update_agg(agg_main, rec["eval_last"])

        if (
            admit_eval_h > 0
            and obj_ids
            and p_xu is not None
            and p_ma is not None
            and _scope_allows(is_warmup)
        ):
            if admit_eval_k is None or (
                isinstance(admit_eval_k, (int, float)) and admit_eval_k <= 0
            ):
                admit_budget = il_cfg.pop_top_percent
            else:
                admit_budget = admit_eval_k

            k = k_from_budget(len(obj_ids), admit_budget)
            if k > 0:
                idx_xu = np.argsort(-p_xu)[:k]
                idx_ma = np.argsort(-p_ma)[:k]
                pending_post.append(
                    {
                        "slot_idx": int(slot_idx),
                        "is_warmup": bool(is_warmup),
                        "remaining": int(admit_eval_h),
                        "future_slots": 0,
                        "future_total_reqs": 0,
                        "admit_sets": {
                            "xu": {obj_ids[i] for i in idx_xu},
                            "ma": {obj_ids[i] for i in idx_ma},
                        },
                        "admit_k": {"xu": int(k), "ma": int(k)},
                        "future_hits": {"xu": 0, "ma": 0},
                        "future_hit_objs": {"xu": set(), "ma": set()},
                    }
                )

        # UPDATE model pakai LAST (selaras pola slot-level & konsisten dengan statistik slot)
        slot_dataset_last_ma = dataset_to_nan(slot_dataset_last, il_cfg.missing_gap_value)
        info_xu = il_xu.update_slot(slot_dataset_last)
        info_ma = il_ma.update_slot(slot_dataset_last_ma)
        rec["learnnse"] = {
            "xu": {
                "E_t": info_xu.get("E_t"),
                "acc_before": info_xu.get("acc_before"),
                "acc_after": info_xu.get("acc_after"),
                "num_learners": info_xu.get("num_learners_after"),
                "weights": _weights_stats(info_xu.get("weights")),
            },
            "ma": {
                "E_t": info_ma.get("E_t"),
                "acc_before": info_ma.get("acc_before"),
                "acc_after": info_ma.get("acc_after"),
                "num_learners": info_ma.get("num_learners_after"),
                "weights": _weights_stats(info_ma.get("weights")),
            },
        }



    prev_ts = None  # pastikan ada sebelum loop
    for req in req_iter:
        if req is None:
            break

        obj_id, ts = extract_obj_ts(req, prev_ts, global_idx)
        prev_ts = ts

        gaps = ft.update_and_get_gaps(obj_id, ts)

        info = slot_stats.get(obj_id)
        if info is None:
            # simpan BOTH: first_gaps (kemunculan pertama) dan last_gaps (update terakhir)
            slot_stats[obj_id] = {
                "freq": 1,
                "first_gaps": gaps,
                "last_gaps": gaps,
                # opsional: hist_len dari gaps (untuk segmentasi cold-start)
                "hist_len": sum(1 for g in gaps[:il_cfg.num_gaps] if g != il_cfg.missing_gap_value),
            }
        else:
            info["freq"] += 1
            info["last_gaps"] = gaps
            # hist_len boleh update ke versi terakhir (berguna untuk analisis)
            info["hist_len"] = sum(1 for g in gaps[:il_cfg.num_gaps] if g != il_cfg.missing_gap_value)

        global_idx += 1
        slot_req_count += 1

        if slot_req_count >= slot_size:
            slot_num = slot_idx + 1
            slot_dataset_last, label_info_last = build_slot_dataset_from_stats(
                slot_stats,
                il_cfg.pop_top_percent,
                effective_num_gaps,
                il_cfg.missing_gap_value,
                label_mode,
                coverage_target,
                label_capacity_gamma,
                None,
                slot_num,
                feature_set,
                freq_history,
                cum_counts,
                FEATURE_WINDOWS,
                slot_size,
                gaps_key="last_gaps",
            )

            slot_dataset_first, label_info_first = build_slot_dataset_from_stats(
                slot_stats,
                il_cfg.pop_top_percent,
                effective_num_gaps,
                il_cfg.missing_gap_value,
                label_mode,
                coverage_target,
                label_capacity_gamma,
                None,
                slot_num,
                feature_set,
                freq_history,
                cum_counts,
                FEATURE_WINDOWS,
                slot_size,
                gaps_key="first_gaps",
            )

            is_warmup = slot_idx < warmup_slots
            finalize_slot(
                slot_dataset_last,
                slot_dataset_first,
                is_warmup,
                slot_stats,
                label_info_last,
                label_info_first,
            )

            _update_freq_history(
                slot_num,
                slot_stats,
                freq_history,
                cum_counts,
                FEATURE_WINDOWS["long"],
            )
            slot_stats = {}
            slot_req_count = 0
            slot_idx += 1
            pbar.update(1)

        if global_idx >= total_requests:
            break


    if slot_req_count > 0:
        slot_num = slot_idx + 1
        slot_dataset_last, label_info_last = build_slot_dataset_from_stats(
            slot_stats,
            il_cfg.pop_top_percent,
            effective_num_gaps,
            il_cfg.missing_gap_value,
            label_mode,
            coverage_target,
            label_capacity_gamma,
            None,
            slot_num,
            feature_set,
            freq_history,
            cum_counts,
            FEATURE_WINDOWS,
            slot_size,
            gaps_key="last_gaps",
        )

        slot_dataset_first, label_info_first = build_slot_dataset_from_stats(
            slot_stats,
            il_cfg.pop_top_percent,
            effective_num_gaps,
            il_cfg.missing_gap_value,
            label_mode,
            coverage_target,
            label_capacity_gamma,
            None,
            slot_num,
            feature_set,
            freq_history,
            cum_counts,
            FEATURE_WINDOWS,
            slot_size,
            gaps_key="first_gaps",
        )

        is_warmup = slot_idx < warmup_slots
        finalize_slot(
            slot_dataset_last,
            slot_dataset_first,
            is_warmup,
            slot_stats,
            label_info_last,
            label_info_first,
        )
        _update_freq_history(
            slot_num,
            slot_stats,
            freq_history,
            cum_counts,
            FEATURE_WINDOWS["long"],
        )
        pbar.update(1)

    pbar.close()

    if pending_post:
        for rec in pending_post:
            _finalize_post_admit(rec, horizon_complete=False)
        pending_post.clear()

    def mean(xs: List[float]) -> float:
        xs2 = [v for v in xs if v == v]
        return float(np.mean(xs2)) if xs2 else float("nan")

    summary = {
        "trace_path": trace_path,
        "total_requests": int(total_requests),
        "warmup_requests": int(warmup_requests),
        "slot_size": int(slot_size),
        "num_gaps": int(il_cfg.num_gaps),
        "feature_set": str(feature_set),
        "feature_windows": dict(FEATURE_WINDOWS),
        "pop_top_percent": float(il_cfg.pop_top_percent),
        "label_mode": label_mode,
        "max_classifiers": int(max_learners),
        "warmup_slots": int(warmup_slots),
        "slots_eval": int(agg_main["slots_eval"]),
        "mean": {
            "xu": {"ap": mean(agg_main["xu_ap"]), "auc": mean(agg_main["xu_auc"]), "precK": mean(agg_main["xu_precK"])},
            "ma": {"ap": mean(agg_main["ma_ap"]), "auc": mean(agg_main["ma_auc"]), "precK": mean(agg_main["ma_precK"])},
            "hist_len<=1": {
                "xu_ap": mean(agg_main["xu_ap_small"]),
                "ma_ap": mean(agg_main["ma_ap_small"]),
                "xu_precK": mean(agg_main["xu_precK_small"]),
                "ma_precK": mean(agg_main["ma_precK_small"]),
            }
        },
        "warmup_mean": {
            "slots_eval": int(agg_warmup["slots_eval"]),
            "xu": {"ap": mean(agg_warmup["xu_ap"]), "auc": mean(agg_warmup["xu_auc"]), "precK": mean(agg_warmup["xu_precK"])},
            "ma": {"ap": mean(agg_warmup["ma_ap"]), "auc": mean(agg_warmup["ma_auc"]), "precK": mean(agg_warmup["ma_precK"])},
            "hist_len<=1": {
                "xu_ap": mean(agg_warmup["xu_ap_small"]),
                "ma_ap": mean(agg_warmup["ma_ap_small"]),
                "xu_precK": mean(agg_warmup["xu_precK_small"]),
                "ma_precK": mean(agg_warmup["ma_precK_small"]),
            },
        },
        "post_admit": {
            "horizon_slots": int(admit_eval_h),
            "admit_k": float(admit_eval_k) if admit_eval_k is not None else float(il_cfg.pop_top_percent),
            "scope": str(admit_eval_scope),
            "slots_complete": int(post_agg_main["slots_eval"]),
            "slots_incomplete": int(post_incomplete),
            "mean": {
                "xu": {m: mean(post_agg_main["xu"][m]) for m in post_metrics},
                "ma": {m: mean(post_agg_main["ma"][m]) for m in post_metrics},
            },
            "warmup_mean": {
                "slots_eval": int(post_agg_warmup["slots_eval"]),
                "xu": {m: mean(post_agg_warmup["xu"][m]) for m in post_metrics},
                "ma": {m: mean(post_agg_warmup["ma"][m]) for m in post_metrics},
            },
        },
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Wrote:", detail_path)
    print("Wrote:", summary_path)
    print("Wrote:", post_admit_path)

if __name__ == "__main__":
    ds = WIKIPEDIA_SEPTEMBER_2007
    base_il_cfg = ILConfig()
    for feature_set in FEATURE_SETS:
        run_nb_eval_no_cache(
            trace_path=ds.path,
            total_requests=ds.num_total_requests,
            warmup_requests=ds.num_warmup_requests,
            slot_size=ds.slot_size,
            il_cfg=base_il_cfg,
            feature_set=feature_set,
            results_root="results_nse",
            dataset_name=f"{ds.name}_{feature_set}",
            dataset_base_name=ds.name,
        )
