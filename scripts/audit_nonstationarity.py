#!/usr/bin/env python3
"""
Audit non-stationarity of per-slot popularity (top-K churn, persistence, re-entry).

Example:
  python scripts/audit_nonstationarity.py --dataset WIKIPEDIA_SEPTEMBER_2007 --slots 90 --out results/audit_nonstat_2007.jsonl
  python scripts/audit_nonstationarity.py --dataset WIKI2018 --slots 90 --out results/audit_nonstat_wiki2018.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Dict, Iterable, List, Optional

# --- ensure repo root in sys.path ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import experiment_config as EC
from src.data.slot_iterator import iter_slots_from_trace


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


def mean(vals: Iterable[float]) -> Optional[float]:
    vals = list(vals)
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def main() -> None:
    il_cfg = EC.ILConfig()

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="WIKI2018",
                    help="DatasetConfig name in experiment_config.py (e.g., WIKI2018, WIKIPEDIA_SEPTEMBER_2007)")
    ap.add_argument("--slots", type=int, default=30)
    ap.add_argument("--slot-size", type=int, default=None)
    ap.add_argument("--top-ratio", type=float, default=None,
                    help="Top-K ratio per slot (default ILConfig.pop_top_percent)")
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--out", default="results/audit_nonstationarity.jsonl")
    args = ap.parse_args()

    if not hasattr(EC, args.dataset):
        raise SystemExit(f"Unknown dataset '{args.dataset}'")
    ds = getattr(EC, args.dataset)

    slot_size = args.slot_size if args.slot_size is not None else ds.slot_size
    top_ratio = args.top_ratio if args.top_ratio is not None else il_cfg.pop_top_percent

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    prev_top: Optional[set] = None
    last_top_slot: Dict[str, int] = {}
    last_streak: Dict[str, int] = {}

    jaccards: List[float] = []
    retentions: List[float] = []
    new_ratios: List[float] = []
    drop_ratios: List[float] = []
    top_shares: List[float] = []
    reentry_gaps_all: List[int] = []
    streaks_all: List[int] = []

    with open(args.out, "w", encoding="utf-8") as f_out:
        for slot_idx, slot in enumerate(
            iter_slots_from_trace(path=ds.path, slot_size=slot_size, max_rows=args.max_rows),
            start=1,
        ):
            if slot_idx > args.slots:
                break

            # frequency per object in this slot
            freq: Dict[str, int] = {}
            for req in slot:
                obj = str(req.get("object_id"))
                freq[obj] = freq.get(obj, 0) + 1

            n_objects = len(freq)
            if n_objects == 0:
                continue

            k = int(n_objects * top_ratio)
            if k < 1:
                k = 1
            if k > n_objects:
                k = n_objects

            # top-K by freq
            items = sorted(freq.items(), key=lambda kv: kv[1], reverse=True)
            top_ids = [obj for obj, _ in items[:k]]
            top_set = set(top_ids)

            # overlap stats
            if prev_top is None:
                jaccard = None
                retention = None
                new_ratio = None
                drop_ratio = None
            else:
                inter = len(top_set & prev_top)
                union = len(top_set | prev_top)
                jaccard = (inter / union) if union > 0 else 0.0
                retention = inter / len(prev_top) if prev_top else 0.0
                new_ratio = (len(top_set) - inter) / len(top_set) if top_set else 0.0
                drop_ratio = (len(prev_top) - inter) / len(prev_top) if prev_top else 0.0

            # re-entry & streak stats
            reentry_gaps: List[int] = []
            streaks: List[int] = []
            for obj in top_set:
                prev = last_top_slot.get(obj)
                if prev is not None:
                    gap = slot_idx - prev - 1
                    if gap > 0:
                        reentry_gaps.append(gap)
                if prev == slot_idx - 1:
                    streak = last_streak.get(obj, 0) + 1
                else:
                    streak = 1
                last_streak[obj] = streak
                last_top_slot[obj] = slot_idx
                streaks.append(streak)

            # top-K frequency stats
            top_freqs = [freq[obj] for obj in top_ids]
            top_freqs_sorted = sorted(top_freqs)
            top_share = sum(top_freqs) / float(len(slot)) if slot else 0.0

            record = {
                "slot": slot_idx,
                "slot_size": len(slot),
                "objects_in_slot": n_objects,
                "top_k": k,
                "top_share": float(top_share),
                "jaccard_prev": jaccard,
                "retention_prev": retention,
                "new_ratio": new_ratio,
                "drop_ratio": drop_ratio,
                "top_freq_mean": float(sum(top_freqs) / len(top_freqs)) if top_freqs else None,
                "top_freq_p50": percentile(top_freqs_sorted, 50),
                "top_freq_p90": percentile(top_freqs_sorted, 90),
                "reentry_count": len(reentry_gaps),
                "reentry_gap_mean": mean(reentry_gaps),
                "streak_mean": mean(streaks),
                "streak_max": max(streaks) if streaks else None,
            }
            f_out.write(json.dumps(record) + "\n")

            if jaccard is not None:
                jaccards.append(float(jaccard))
            if retention is not None:
                retentions.append(float(retention))
            if new_ratio is not None:
                new_ratios.append(float(new_ratio))
            if drop_ratio is not None:
                drop_ratios.append(float(drop_ratio))
            top_shares.append(float(top_share))
            reentry_gaps_all.extend(reentry_gaps)
            streaks_all.extend(streaks)

            prev_top = top_set

    summary = {
        "dataset": args.dataset,
        "slots": args.slots,
        "top_ratio": top_ratio,
        "avg_jaccard": mean(jaccards),
        "avg_retention": mean(retentions),
        "avg_new_ratio": mean(new_ratios),
        "avg_drop_ratio": mean(drop_ratios),
        "avg_top_share": mean(top_shares),
        "avg_reentry_gap": mean(reentry_gaps_all),
        "avg_streak": mean(streaks_all),
        "max_streak": max(streaks_all) if streaks_all else None,
        "total_reentries": len(reentry_gaps_all),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
