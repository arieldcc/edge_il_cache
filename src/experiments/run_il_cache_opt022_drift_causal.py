from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.experiments import run_il_cache_opt022_drift_only as drift
from src.data.trace_reader import TraceReader as BaseTraceReader


class ConcatTraceReader:
    """
    TraceReader wrapper untuk concatenation A||B dengan batas per-segmen.
    Format path: "pathA::N1||pathB::N2"
    """

    def __init__(self, path: str, max_rows: Optional[int] = None) -> None:
        self.path = path
        self.max_rows = max_rows
        self.segments: List[Tuple[str, Optional[int]]] = []
        for part in path.split("||"):
            if "::" in part:
                p, n = part.split("::", 1)
                n_val = int(n)
                self.segments.append((p, n_val))
            else:
                self.segments.append((part, None))

    def iter_requests(self):
        count = 0
        for seg_path, seg_limit in self.segments:
            reader = BaseTraceReader(path=seg_path, max_rows=seg_limit)
            for req in reader.iter_requests():
                yield req
                count += 1
                if self.max_rows is not None and count >= self.max_rows:
                    return


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Causal drift experiment: fixed vs adaptive vs permuted (iso-total).",
    )
    parser.add_argument(
        "--dataset-a",
        choices=("wikipedia_september_2007", "wiki2018", "wikipedia_oktober_2007"),
        default="wikipedia_september_2007",
    )
    parser.add_argument(
        "--dataset-b",
        choices=("wikipedia_september_2007", "wiki2018", "wikipedia_oktober_2007"),
        default="wiki2018",
    )
    parser.add_argument(
        "--seg-a",
        type=int,
        default=5_000_000,
        help="Jumlah request dari trace A (termasuk warmup).",
    )
    parser.add_argument(
        "--seg-b",
        type=int,
        default=4_000_000,
        help="Jumlah request dari trace B (setelah change-point).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1_000_000,
    )
    parser.add_argument(
        "--slot-size",
        type=int,
        default=100_000,
    )
    parser.add_argument(
        "--feature-set",
        choices=("A1", "A2", "A3"),
        default="A2",
    )
    parser.add_argument(
        "--drift-method",
        choices=("jsd", "adwin", "page_hinkley"),
        default="jsd",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
    )
    parser.add_argument(
        "--model-prefix",
        type=str,
        default="ilnse_A2_drift_causal",
    )
    parser.add_argument(
        "--cache-size-percentages",
        nargs="+",
        type=float,
        default=None,
        help="Override cache size percentages (e.g., 0.8 2.0 5.0).",
    )
    return parser.parse_args()


def _dataset_cfg(name: str) -> Dict[str, object]:
    if name == "wikipedia_september_2007":
        return drift.WIKIPEDIA_SEPTEMBER_2007
    if name == "wikipedia_oktober_2007":
        return drift.WIKIPEDIA_OKTOBER_2007
    if name == "wiki2018":
        return drift.WIKI2018
    raise ValueError(name)


def _load_budget_schedule(slot_log_path: str) -> List[int]:
    budgets: List[int] = []
    with open(slot_log_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("phase") != "cache":
                continue
            budgets.append(int(row.get("admit_budget", 0)))
    return budgets


def _fixed_schedule(budgets: List[int]) -> List[int]:
    if not budgets:
        return []
    total = int(sum(budgets))
    n = len(budgets)
    mean = int(round(total / n))
    schedule = [mean for _ in range(n)]
    schedule[-1] = max(0, total - mean * (n - 1))
    return schedule


def _permuted_schedule(budgets: List[int], rng: np.random.Generator) -> List[int]:
    if not budgets:
        return []
    perm = budgets.copy()
    rng.shuffle(perm)
    return perm


def _run_variant(
    ds_cfg: Dict[str, object],
    trace_path: str,
    total_requests: int,
    warmup_requests: int,
    slot_size: int,
    feature_set: str,
    drift_method: str,
    model_name: str,
    budget_schedule: Optional[List[int]] = None,
    budget_schedule_map: Optional[Dict[int, List[int]]] = None,
    capacities_override: Optional[List[int]] = None,
    cache_size_percentages: Optional[List[float]] = None,
) -> Tuple[List[int], List[dict]]:
    capacities = (
        capacities_override
        if capacities_override is not None
        else drift.get_dynamic_capacities(
            trace_path,
            total_requests,
            cache_size_percentages if cache_size_percentages is not None else list(drift.CACHE_SIZE_PERCENTAGES),
        )
    )
    results: List[dict] = []
    for capacity in tqdm(capacities, desc=f"Capacities [{model_name}]", unit="cap"):
        schedule = budget_schedule_map.get(capacity) if budget_schedule_map else budget_schedule
        run_id, dataset_dir = drift.get_next_run_id("results", ds_cfg["name"], model_name, capacity)
        per_slot_path = os.path.join(dataset_dir, f"{run_id}_{model_name}_{capacity}.jsonl")
        summary_path = os.path.join(dataset_dir, f"{run_id}_summary_{model_name}_{capacity}.json")
        stats, summary_metrics = drift.run_single_capacity(
            trace_path=trace_path,
            total_requests=total_requests,
            warmup_requests=warmup_requests,
            slot_size=slot_size,
            capacity_objects=capacity,
            feature_set=feature_set,
            drift_method=drift_method,
            slot_log_path=per_slot_path,
            budget_schedule=schedule,
        )
        slots_processed = int(summary_metrics.get("slots_processed", 0))
        warmup_slots = warmup_requests // slot_size
        cache_slots = max(0, slots_processed - warmup_slots)
        summary_payload = {
            "dataset": ds_cfg["name"],
            "model": model_name,
            "feature_set": feature_set,
            "drift_method": drift_method,
            "drift_control_mode": drift.DRIFT_CONTROL_MODE,
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
            "pop_top_percent": drift.IL_POP_TOP_PERCENT,
            "label_topk_rounding": drift.IL_LABEL_TOPK_ROUNDING,
            "label_tie_break": drift.IL_LABEL_TIE_BREAK,
            "admission_policy": "top_capacity_rate",
            "admission_capacity_alpha": drift.ADMISSION_CAPACITY_ALPHA,
        }
        summary_payload.update(summary_metrics)
        drift.save_json(summary_path, summary_payload)
        results.append(summary_payload)
        print(f"      [LOG] per-slot  -> {per_slot_path}")
        print(f"      [LOG] summary   -> {summary_path}")
    return capacities, results


def main() -> None:
    args = _parse_args()
    rng = np.random.default_rng(args.seed)

    ds_a = _dataset_cfg(args.dataset_a)
    ds_b = _dataset_cfg(args.dataset_b)
    concat_path = f"{ds_a['path']}::{args.seg_a}||{ds_b['path']}::{args.seg_b}"
    total_requests = int(args.seg_a + args.seg_b)
    warmup_requests = int(args.warmup)
    slot_size = int(args.slot_size)

    if warmup_requests % slot_size != 0:
        raise ValueError("warmup_requests harus kelipatan slot_size")

    # Override reader to support concatenation
    drift.TraceReader = ConcatTraceReader  # type: ignore

    # disable fill phase to keep budget schedule clean
    drift.FILL_RATIO = 0.0
    drift.FILL_RATE = 0.0

    ds_cfg = {
        "name": f"causal_{args.dataset_a}_to_{args.dataset_b}",
        "path": concat_path,
        "num_total_requests": total_requests,
        "num_warmup_requests": warmup_requests,
        "slot_size": slot_size,
    }

    print(f"=== CAUSAL DRIFT EXPERIMENT ({ds_cfg['name']}) ===")
    print(f"Trace A: {ds_a['path']} ({args.seg_a} reqs)")
    print(f"Trace B: {ds_b['path']} ({args.seg_b} reqs)")
    print(f"Change-point at request: {args.seg_a}")
    print(f"Total requests: {total_requests}, warmup: {warmup_requests}, slot_size: {slot_size}")
    print()

    # Adaptive run to extract schedule
    drift.DRIFT_CONTROL_MODE = "scaled"
    adaptive_model = f"{args.model_prefix}_adaptive"
    capacities, _adaptive_summaries = _run_variant(
        ds_cfg,
        concat_path,
        total_requests,
        warmup_requests,
        slot_size,
        args.feature_set,
        args.drift_method,
        adaptive_model,
        budget_schedule=None,
        cache_size_percentages=args.cache_size_percentages,
    )

    # Build per-capacity schedules from adaptive logs
    dataset_dir = os.path.join("results", ds_cfg["name"])
    schedule_map: Dict[int, List[int]] = {}
    for capacity in capacities:
        sched_logs: List[str] = []
        for fname in os.listdir(dataset_dir):
            if fname.endswith(f"_{adaptive_model}_{capacity}.jsonl"):
                sched_logs.append(os.path.join(dataset_dir, fname))
        if not sched_logs:
            raise FileNotFoundError(f"Adaptive per-slot log not found for capacity {capacity}")
        sched_log_path = sorted(sched_logs)[-1]
        schedule_map[capacity] = _load_budget_schedule(sched_log_path)

    fixed_schedule_map = {cap: _fixed_schedule(schedule_map[cap]) for cap in capacities}
    permuted_schedule_map = {cap: _permuted_schedule(schedule_map[cap], rng) for cap in capacities}

    # Fixed schedule run
    drift.DRIFT_CONTROL_MODE = "fixed"
    fixed_model = f"{args.model_prefix}_fixed_iso"
    _run_variant(
        ds_cfg,
        concat_path,
        total_requests,
        warmup_requests,
        slot_size,
        args.feature_set,
        args.drift_method,
        fixed_model,
        budget_schedule_map=fixed_schedule_map,
        capacities_override=capacities,
    )

    # Permuted schedule run
    perm_model = f"{args.model_prefix}_permuted_iso"
    _run_variant(
        ds_cfg,
        concat_path,
        total_requests,
        warmup_requests,
        slot_size,
        args.feature_set,
        args.drift_method,
        perm_model,
        budget_schedule_map=permuted_schedule_map,
        capacities_override=capacities,
    )


if __name__ == "__main__":
    main()
