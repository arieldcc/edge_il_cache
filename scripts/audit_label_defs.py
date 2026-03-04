#!/usr/bin/env python3
"""
Audit label definitions: Xu (top-K% objects) vs Coverage-based.

Example:
  python scripts/audit_label_defs.py --dataset WIKIPEDIA_SEPTEMBER_2007 --slots 90 --coverage 0.6
  python scripts/audit_label_defs.py --dataset WIKI2018 --slots 90 --coverage 0.6
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Dict, List, Optional, Set, Tuple

# --- ensure repo root in sys.path ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import experiment_config as EC
from src.data.slot_iterator import iter_slots_from_trace


def jaccard(a: Set[str], b: Set[str]) -> Optional[float]:
    if not a and not b:
        return None
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union > 0 else 0.0


def mean(vals: List[float]) -> Optional[float]:
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def percentile(sorted_vals: List[float], p: float) -> Optional[float]:
    if not sorted_vals:
        return None
    if p <= 0:
        return sorted_vals[0]
    if p >= 100:
        return sorted_vals[-1]
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = int(math.floor(k))
    c = int(math.ceil(k))
    if f == c:
        return sorted_vals[f]
    return sorted_vals[f] * (c - k) + sorted_vals[c] * (k - f)


def main() -> None:
    il_cfg = EC.ILConfig()

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="WIKI2018",
                    help="DatasetConfig name in experiment_config.py")
    ap.add_argument("--slots", type=int, default=30)
    ap.add_argument("--slot-size", type=int, default=None)
    ap.add_argument("--top-ratio", type=float, default=None,
                    help="Top-K% for Xu definition (default ILConfig.pop_top_percent)")
    ap.add_argument("--coverage", type=float, default=0.6,
                    help="Coverage target for coverage-based labels (fraction of requests).")
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--out", default="results/audit_label_defs.jsonl")
    ap.add_argument("--summary-out", default=None)
    args = ap.parse_args()

    if not hasattr(EC, args.dataset):
        raise SystemExit(f"Unknown dataset '{args.dataset}'")
    ds = getattr(EC, args.dataset)

    slot_size = args.slot_size if args.slot_size is not None else ds.slot_size
    top_ratio = args.top_ratio if args.top_ratio is not None else il_cfg.pop_top_percent
    coverage = float(args.coverage)
    if coverage <= 0.0 or coverage > 1.0:
        raise SystemExit("coverage must be in (0, 1].")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    prev_top_xu: Optional[Set[str]] = None
    prev_top_cov: Optional[Set[str]] = None

    jacc_xu: List[float] = []
    jacc_cov: List[float] = []
    size_xu_list: List[int] = []
    size_cov_list: List[int] = []
    share_xu_list: List[float] = []
    share_cov_list: List[float] = []
    gap_xu_list: List[float] = []

    with open(args.out, "w", encoding="utf-8") as f_out:
        for slot_idx, slot in enumerate(
            iter_slots_from_trace(path=ds.path, slot_size=slot_size, max_rows=args.max_rows),
            start=1,
        ):
            if slot_idx > args.slots:
                break

            freq: Dict[str, int] = {}
            for req in slot:
                obj = str(req.get("object_id"))
                freq[obj] = freq.get(obj, 0) + 1

            n_objects = len(freq)
            if n_objects == 0:
                continue

            total_reqs = len(slot)
            items = sorted(freq.items(), key=lambda kv: kv[1], reverse=True)

            # Xu: top-K% objects
            k = int(n_objects * top_ratio)
            if k < 1:
                k = 1
            if k > n_objects:
                k = n_objects
            top_xu = set(obj for obj, _ in items[:k])
            share_xu = sum(freq[obj] for obj in top_xu) / float(total_reqs)

            # Coverage: minimal set to reach target coverage
            cum = 0
            top_cov: List[str] = []
            for obj, f in items:
                cum += f
                top_cov.append(obj)
                if cum / float(total_reqs) >= coverage:
                    break
            top_cov_set = set(top_cov)
            share_cov = cum / float(total_reqs)

            # Overlap & churn
            j_xu = jaccard(prev_top_xu, top_xu) if prev_top_xu is not None else None
            j_cov = jaccard(prev_top_cov, top_cov_set) if prev_top_cov is not None else None

            rec = {
                "slot": slot_idx,
                "objects_in_slot": n_objects,
                "slot_size": total_reqs,
                "top_ratio": top_ratio,
                "coverage_target": coverage,
                "xu_size": len(top_xu),
                "cov_size": len(top_cov_set),
                "xu_share": float(share_xu),
                "cov_share": float(share_cov),
                "xu_cov_gap": float(share_xu - coverage),
                "jaccard_xu_prev": j_xu,
                "jaccard_cov_prev": j_cov,
                "size_ratio_cov_xu": (len(top_cov_set) / len(top_xu)) if len(top_xu) > 0 else None,
            }
            f_out.write(json.dumps(rec) + "\n")

            size_xu_list.append(len(top_xu))
            size_cov_list.append(len(top_cov_set))
            share_xu_list.append(float(share_xu))
            share_cov_list.append(float(share_cov))
            gap_xu_list.append(float(share_xu - coverage))
            if j_xu is not None:
                jacc_xu.append(float(j_xu))
            if j_cov is not None:
                jacc_cov.append(float(j_cov))

            prev_top_xu = top_xu
            prev_top_cov = top_cov_set

    summary = {
        "dataset": args.dataset,
        "slots": args.slots,
        "top_ratio": top_ratio,
        "coverage_target": coverage,
        "avg_xu_size": mean(size_xu_list),
        "avg_cov_size": mean(size_cov_list),
        "avg_xu_share": mean(share_xu_list),
        "avg_cov_share": mean(share_cov_list),
        "avg_xu_cov_gap": mean(gap_xu_list),
        "p50_xu_share": percentile(sorted(share_xu_list), 50),
        "p90_xu_share": percentile(sorted(share_xu_list), 90),
        "avg_jaccard_xu": mean(jacc_xu),
        "avg_jaccard_cov": mean(jacc_cov),
    }

    if args.summary_out:
        with open(args.summary_out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"[audit] summary -> {args.summary_out}")
    else:
        print(json.dumps(summary, indent=2))

    print(f"[audit] written -> {args.out}")


if __name__ == "__main__":
    main()
