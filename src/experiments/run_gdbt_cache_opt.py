# src/experiments/run_gdbt_cache_opt.py

from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import lightgbm as lgb
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.trace_reader import TraceReader
from src.data.feature_table import FeatureTable
from src.cache.cache_simulator import CacheStats
from src.cache.lru import LRUCache

FEATURE_SETS = ("A0", "A1", "A2")
FEATURE_WINDOWS = {"short": 1, "mid": 7, "long": 30}

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

IL_NUM_GAPS = 6
IL_POP_TOP_PERCENT = 0.20
IL_LABEL_TOPK_ROUNDING = "floor"
IL_LABEL_TIE_BREAK = "none"
IL_MISSING_GAP_VALUE = 1e6

GDBT_N_ESTIMATORS = 30
GDBT_UPDATE_INTERVAL_REQUESTS = 1_000_000

CACHE_SIZE_PERCENTAGES = (0.8, 1.0, 2.0, 3.0, 4.0, 5.0)
DEFAULT_FEATURE_SET = "A2"


@dataclass
class ILConfig:
    num_gaps: int = IL_NUM_GAPS
    pop_top_percent: float = IL_POP_TOP_PERCENT
    missing_gap_value: float = IL_MISSING_GAP_VALUE


@dataclass
class GDBTConfig:
    n_estimators: int = GDBT_N_ESTIMATORS
    update_interval_requests: int = GDBT_UPDATE_INTERVAL_REQUESTS


@dataclass
class GDBTCachePredictorOpt:
    il_config: ILConfig
    gdbt_config: GDBTConfig
    n_features: int
    model: Optional[lgb.Booster] = None
    _X_buffer: List[List[float]] = None  # type: ignore
    _y_buffer: List[int] = None          # type: ignore
    num_rebuilds: int = 0
    total_training_samples: int = 0
    last_rebuild_info: Dict[str, Any] = None  # type: ignore

    def __post_init__(self) -> None:
        self._X_buffer = []
        self._y_buffer = []
        self.num_rebuilds = 0
        self.total_training_samples = 0
        self.last_rebuild_info = {}

    def add_training_sample(self, x: List[float], y: int) -> None:
        if len(x) != self.n_features:
            raise ValueError(
                f"n_features mismatch: expected {self.n_features}, got {len(x)}"
            )
        self._X_buffer.append(list(x))
        self._y_buffer.append(int(y))

    def add_training_batch(self, dataset: List[Dict[str, Any]]) -> None:
        for item in dataset:
            self.add_training_sample(item["x"], item["y"])

    def get_buffer_size(self) -> int:
        return len(self._X_buffer)

    def should_rebuild(self, requests_since_last_rebuild: int) -> bool:
        return requests_since_last_rebuild >= int(self.gdbt_config.update_interval_requests)

    def _create_lgb_params(self) -> Dict[str, Any]:
        return {
            "objective": "binary",
            "boosting_type": "gbdt",
            "verbose": -1,
        }

    def rebuild_model(self) -> Dict[str, Any]:
        rebuild_info: Dict[str, Any] = {
            "rebuild_number": self.num_rebuilds + 1,
            "buffer_size": len(self._X_buffer),
            "success": False,
        }
        if not self._X_buffer:
            rebuild_info["error"] = "Empty buffer, cannot rebuild"
            self.last_rebuild_info = rebuild_info
            return rebuild_info

        try:
            X = np.asarray(self._X_buffer, dtype=float)
            y = np.asarray(self._y_buffer, dtype=int)

            if X.ndim != 2 or X.shape[1] != self.n_features:
                raise ValueError(
                    f"n_features mismatch: expected {self.n_features}, got {X.shape[1]}"
                )

            dtrain = lgb.Dataset(X, label=y)
            params = self._create_lgb_params()
            self.model = lgb.train(params, dtrain, num_boost_round=self.gdbt_config.n_estimators)

            self.num_rebuilds += 1
            self.total_training_samples += len(self._X_buffer)
            rebuild_info["success"] = True
            rebuild_info.update(
                {
                    "n_estimators": int(self.gdbt_config.n_estimators),
                    "n_estimators_actual": int(self.model.num_trees()),
                }
            )
        except Exception as exc:
            rebuild_info["error"] = str(exc)

        self.last_rebuild_info = rebuild_info
        return rebuild_info

    def clear_buffer(self) -> None:
        self._X_buffer = []
        self._y_buffer = []

    def predict(self, x: List[float]) -> int:
        if self.model is None:
            return 1
        x_arr = np.asarray(x, dtype=float).reshape(1, -1)
        if x_arr.shape[1] != self.n_features:
            raise ValueError(
                f"n_features mismatch: expected {self.n_features}, got {x_arr.shape[1]}"
            )
        proba = float(self.model.predict(x_arr)[0])
        return int(proba >= 0.5)

    def get_feature_importances(self) -> Optional[np.ndarray]:
        if self.model is None:
            return None
        try:
            return self.model.feature_importance()
        except Exception:
            return None

    def get_stats(self) -> Dict[str, Any]:
        return {
            "num_rebuilds": self.num_rebuilds,
            "total_training_samples": self.total_training_samples,
            "current_buffer_size": self.get_buffer_size(),
            "has_model": self.model is not None,
            "last_rebuild_info": self.last_rebuild_info,
            "n_features": self.n_features,
            "pop_top_percent": self.il_config.pop_top_percent,
            "update_interval_requests": self.gdbt_config.update_interval_requests,
        }


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
    raise ValueError(f"Unknown feature_set: {feature_set}")


def _sum_history(window: int, slot_index: int, history: List[Tuple[int, int]]) -> int:
    if slot_index <= 0 or window <= 0:
        return 0
    start_slot = max(1, slot_index - window + 1)
    return sum(count for slot_id, count in history if slot_id >= start_slot)


def _get_history_features(
    obj_id: str,
    slot_index: int,
    freq_history: Dict[str, List[Tuple[int, int]]],
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
    raise ValueError(f"Unknown feature_set: {feature_set}")


def _select_top_ids_from_stats(
    slot_stats: Dict[str, Dict[str, Any]],
    top_ratio: float,
    label_topk_rounding: str,
    label_tie_break: str,
) -> Tuple[set[str], Dict[str, Any]]:
    label_info: Dict[str, Any] = {}
    if not slot_stats:
        return set(), label_info
    items: List[Tuple[str, Dict[str, Any]]] = list(slot_stats.items())
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
            "top_ratio": float(len(top_ids) / n_objects) if n_objects > 0 else 0.0,
            "freq_at_k": int(kth_freq),
            "tie_count": int(tie_count),
            "tie_rate": float(tie_count / n_objects) if n_objects > 0 else 0.0,
        }
    )
    return top_ids, label_info


def _update_freq_history(
    slot_index: int,
    slot_stats: Dict[str, Dict[str, Any]],
    freq_history: Dict[str, List[Tuple[int, int]]],
    cum_counts: Dict[str, int],
    long_window: int,
) -> None:
    cutoff = slot_index - long_window + 1
    for obj_id, stats in slot_stats.items():
        count = int(stats.get("freq", 0))
        if count <= 0:
            continue
        history = freq_history.setdefault(obj_id, [])
        history[:] = [(sid, cnt) for sid, cnt in history if sid >= cutoff]
        history.append((slot_index, count))
        cum_counts[obj_id] = int(cum_counts.get(obj_id, 0)) + count


def build_slot_dataset_from_stats(
    slot_stats: Dict[str, Dict[str, Any]],
    top_ratio: float,
    num_gaps: int,
    missing_gap_value: float,
    feature_set: str,
    history_slot_index: int,
    freq_history: Dict[str, List[Tuple[int, int]]],
    cum_counts: Dict[str, int],
    label_topk_rounding: str,
    label_tie_break: str,
) -> List[Dict[str, Any]]:
    if not slot_stats:
        return []
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"Unknown feature_set: {feature_set}")
    num_features = _get_feature_dim(num_gaps, feature_set)

    items: List[Tuple[str, Dict[str, Any]]] = list(slot_stats.items())
    items.sort(key=lambda kv: kv[1]["freq"], reverse=True)

    top_ids, _ = _select_top_ids_from_stats(
        slot_stats,
        top_ratio,
        label_topk_rounding,
        label_tie_break,
    )

    dataset: List[Dict[str, Any]] = []
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
        features = _build_feature_vector(feature_set, gaps, history_feats)
        if len(features) != num_features:
            raise ValueError(
                f"feature length mismatch: expected {num_features}, got {len(features)}"
            )
        y = 1 if obj_id in top_ids else 0
        dataset.append(
            {
                "x": features,
                "y": y,
                "freq": stats["freq"],
                "object_id": obj_id,
            }
        )
    return dataset


def count_distinct_objects(trace_path: str, total_requests: int) -> int:
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
    n_unique = count_distinct_objects(trace_path, total_requests)
    capacities = [max(1, int(n_unique * p / 100.0)) for p in percentages]
    print(f"Jumlah objek unik: {n_unique}")
    print(f"Kapasitas cache (objek): {capacities}")
    return capacities


def get_next_run_id(results_root: str, dataset: str, model: str, cache_size: int) -> Tuple[str, str]:
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


def get_next_group_id(results_root: str, dataset: str, model: str) -> Tuple[str, str]:
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


def run_single_capacity(
    trace_path: str,
    total_requests: int,
    warmup_requests: int,
    slot_size: int,
    capacity_objects: int,
    feature_set: str,
    il_cfg: ILConfig,
    gdbt_cfg: GDBTConfig,
) -> Tuple[CacheStats, List[Dict[str, Any]]]:
    reader = TraceReader(path=trace_path, max_rows=total_requests)
    req_iter = reader.iter_requests()

    num_gaps = il_cfg.num_gaps
    num_features = _get_feature_dim(num_gaps, feature_set)
    feature_table = FeatureTable(
        L=num_gaps,
        missing_gap_value=il_cfg.missing_gap_value,
    )
    gdbt_model = GDBTCachePredictorOpt(il_cfg, gdbt_cfg, num_features)
    cache = LRUCache(capacity_objects=capacity_objects)
    stats = CacheStats(capacity_objects=capacity_objects)

    rebuild_logs: List[Dict[str, Any]] = []
    global_idx = 0
    slot_req_count = 0
    slot_index = 0
    slot_stats: Dict[str, Dict[str, Any]] = {}
    prev_ts: Optional[float] = None

    freq_history: Dict[str, List[Tuple[int, int]]] = {}
    cum_counts: Dict[str, int] = {}

    eval_requests_planned = max(total_requests - warmup_requests, 0)
    pbar = tqdm(
        total=eval_requests_planned,
        desc=f"GDBT Eval (cap={capacity_objects})",
        unit="req",
    )

    warmup_slots = warmup_requests // slot_size if slot_size > 0 else 0
    slots_per_rebuild = (
        gdbt_cfg.update_interval_requests // slot_size if slot_size > 0 else 1
    )
    slots_since_rebuild = 0

    while True:
        try:
            req = next(req_iter)
        except StopIteration:
            if slot_stats:
                D_t = build_slot_dataset_from_stats(
                    slot_stats,
                    il_cfg.pop_top_percent,
                    num_gaps,
                    il_cfg.missing_gap_value,
                    feature_set,
                    slot_index,
                    freq_history,
                    cum_counts,
                    IL_LABEL_TOPK_ROUNDING,
                    IL_LABEL_TIE_BREAK,
                )
                gdbt_model.add_training_batch(D_t)
                _update_freq_history(
                    slot_index,
                    slot_stats,
                    freq_history,
                    cum_counts,
                    FEATURE_WINDOWS["long"],
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

        history_feats = _get_history_features(
            obj_id,
            slot_index,
            freq_history,
            cum_counts,
            FEATURE_WINDOWS,
        )
        features = _build_feature_vector(feature_set, gaps, history_feats)

        if global_idx >= warmup_requests:
            stats.total_requests += 1
            if cache.access(obj_id):
                stats.cache_hits += 1
            else:
                y_hat = gdbt_model.predict(features)
                if y_hat == 1:
                    cache.insert(obj_id)
            pbar.update(1)

        slot_req_count += 1
        if slot_req_count >= slot_size:
            D_t = build_slot_dataset_from_stats(
                slot_stats,
                il_cfg.pop_top_percent,
                num_gaps,
                il_cfg.missing_gap_value,
                feature_set,
                slot_index,
                freq_history,
                cum_counts,
                IL_LABEL_TOPK_ROUNDING,
                IL_LABEL_TIE_BREAK,
            )
            gdbt_model.add_training_batch(D_t)
            _update_freq_history(
                slot_index,
                slot_stats,
                freq_history,
                cum_counts,
                FEATURE_WINDOWS["long"],
            )

            slot_index += 1
            slot_stats = {}
            slot_req_count = 0

            if slot_index == warmup_slots:
                rebuild_info = gdbt_model.rebuild_model()
                rebuild_info["phase"] = "warmup_complete"
                rebuild_info["global_request_idx"] = global_idx
                rebuild_logs.append(rebuild_info)
                gdbt_model.clear_buffer()
                slots_since_rebuild = 0
            elif slot_index > warmup_slots and global_idx >= warmup_requests:
                slots_since_rebuild += 1
                if slots_since_rebuild >= max(1, slots_per_rebuild):
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

        global_idx += 1

    pbar.close()

    final_stats = gdbt_model.get_stats()
    fi = gdbt_model.get_feature_importances()
    final_log = {
        "final_model_stats": final_stats,
        "feature_importances": fi.tolist() if fi is not None else None,
    }
    rebuild_logs.append(final_log)

    return stats, rebuild_logs


def run_experiment(
    ds_cfg: Dict[str, Any],
    feature_set: str = DEFAULT_FEATURE_SET,
    model_name: Optional[str] = None,
    cache_size_percentages: Tuple[float, ...] = CACHE_SIZE_PERCENTAGES,
) -> None:
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"Unknown feature_set: {feature_set}")
    if model_name is None:
        model_name = f"gdbt_{feature_set}"

    trace_path = ds_cfg["path"]
    slot_size = ds_cfg["slot_size"]
    warmup_requests = ds_cfg["num_warmup_requests"]
    total_requests = ds_cfg["num_total_requests"]

    if warmup_requests % slot_size != 0:
        raise ValueError(
            f"warmup_requests ({warmup_requests}) harus kelipatan slot_size ({slot_size})."
        )

    dataset_name = ds_cfg["name"]
    results_root = "results"
    il_cfg = ILConfig()
    gdbt_cfg = GDBTConfig()

    print(f"=== GDBT-based Edge Cache Experiment ({dataset_name} trace) ===")
    print(f"Trace path          : {trace_path}")
    print(f"Total requests      : {total_requests}")
    print(f"Warm-up requests    : {warmup_requests}")
    print(f"Slot size           : {slot_size}")
    print(f"Feature set         : {feature_set}")
    print(f"Num gaps            : {il_cfg.num_gaps}")
    print(f"Num features        : {_get_feature_dim(il_cfg.num_gaps, feature_set)}")
    print(f"Top percent         : {il_cfg.pop_top_percent}")
    print(f"GDBT n_estimators   : {gdbt_cfg.n_estimators}")
    print(f"GDBT update_interval: {gdbt_cfg.update_interval_requests}")
    print(f"Cache size (%)      : {list(cache_size_percentages)}")
    print()

    capacities_objects = get_dynamic_capacities(
        trace_path,
        total_requests,
        list(cache_size_percentages),
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
            feature_set=feature_set,
            il_cfg=il_cfg,
            gdbt_cfg=gdbt_cfg,
        )
        results.append(stats)

        run_id, dataset_dir = get_next_run_id(
            results_root, dataset_name, model_name, capacity
        )

        rebuild_path = os.path.join(
            dataset_dir,
            f"{run_id}_{model_name}_{capacity}_rebuilds.json",
        )
        save_json(
            rebuild_path,
            {
                "dataset": dataset_name,
                "model": model_name,
                "feature_set": feature_set,
                "cache_size_objects": capacity,
                "rebuilds": rebuild_logs,
            },
        )

        summary_path = os.path.join(
            dataset_dir,
            f"{run_id}_summary_{model_name}_{capacity}.json",
        )
        summary_payload = {
            "dataset": dataset_name,
            "model": model_name,
            "feature_set": feature_set,
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
            "num_rebuilds": rebuild_logs[-1]["final_model_stats"]["num_rebuilds"],
        }
        save_json(summary_path, summary_payload)

        print(
            f"      Hit ratio = {stats.hit_ratio:.4f} "
            f"({stats.cache_hits}/{stats.total_requests})"
        )
        print(f"      [LOG] rebuilds -> {rebuild_path}")
        print(f"      [LOG] summary  -> {summary_path}")

    print("\n=== Summary ===")
    for capacity, stats in zip(capacities_objects, results):
        print(
            f"GDBT+LRU, cache_size={capacity}, "
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
        "num_features": _get_feature_dim(il_cfg.num_gaps, feature_set),
        "cache_sizes": capacities_objects,
        "hr_curve": hr_curve,
        "avg_hr": avg_hr,
    }
    save_json(overall_path, overall_payload)


def main() -> None:
    run_experiment(WIKIPEDIA_SEPTEMBER_2007)


if __name__ == "__main__":
    main()
