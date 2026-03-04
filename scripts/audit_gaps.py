#!/usr/bin/env python3
"""
Audit gap features using parameters from src/config/experiment_config.py.

- Automatically uses:
  - L = ILConfig().num_gaps
  - missing_gap_value = ILConfig().missing_gap_value
  - slot_size = DatasetConfig.slot_size (unless overridden)
  - dataset path from WIKI2018 / WIKIPEDIA_* objects

Run:
  python scripts/audit_gaps_from_config.py --dataset WIKI2018 --slots 30
  python scripts/audit_gaps_from_config.py --dataset WIKIPEDIA_SEPTEMBER_2007 --slots 30
"""

from __future__ import annotations
import argparse, json, math, os, sys
from typing import List, Dict

# --- ensure repo root in sys.path (same style as your other scripts) ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import experiment_config as EC
from src.data.feature_table import FeatureTable
from src.data.slot_iterator import iter_slots_from_trace


def percentile(sorted_vals, p: float):
    if not sorted_vals:
        return None
    if p <= 0: return sorted_vals[0]
    if p >= 100: return sorted_vals[-1]
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = int(math.floor(k)); c = int(math.ceil(k))
    if f == c:
        return sorted_vals[f]
    return sorted_vals[f] * (c - k) + sorted_vals[c] * (k - f)


def compute_order_violation_rate(gaps: List[float], missing: float) -> bool:
    """
    Xu gap monotonicity should hold on REAL history entries.
    We ignore padded tail entries equal to `missing`.
    """
    # keep only real gaps (not padded)
    real = [g for g in gaps if g != missing]
    # if 0 or 1 real entries, can't violate
    if len(real) <= 1:
        return False
    # violation if any decreasing step among real entries
    return any(real[i] > real[i+1] for i in range(len(real)-1))


def slot_dim_stats(values_by_dim: List[List[float]], missing: float):
    out = []
    for dim_idx, vals in enumerate(values_by_dim, start=1):
        if not vals:
            out.append({"dim": dim_idx, "n": 0})
            continue
        s = sorted(vals)
        miss = sum(1 for v in vals if v == missing)
        neg = sum(1 for v in vals if v < 0)
        real = [v for v in vals if v != missing]
        real_sorted = sorted(real)
        out.append({
            "dim": dim_idx,
            "n": len(vals),
            "neg": neg,
            "missing": miss,
            "missing_ratio": miss / len(vals) if len(vals) > 0 else None,
            "min": s[0],
            "p50": percentile(s, 50),
            "p95": percentile(s, 95),
            "p99": percentile(s, 99),
            "max": s[-1],
            "real_n": len(real),
            "real_min": real_sorted[0] if real_sorted else None,
            "real_p50": percentile(real_sorted, 50) if real_sorted else None,
            "real_p95": percentile(real_sorted, 95) if real_sorted else None,
            "real_p99": percentile(real_sorted, 99) if real_sorted else None,
            "real_max": real_sorted[-1] if real_sorted else None,
        })
    return out


def main():
    il_cfg = EC.ILConfig()

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="WIKI2018",
                    help="Name of DatasetConfig variable in experiment_config.py, e.g. WIKI2018, WIKIPEDIA_SEPTEMBER_2007")
    ap.add_argument("--slots", type=int, default=30)
    ap.add_argument("--slot-size", type=int, default=None, help="Override slot size (default from dataset config).")
    ap.add_argument("--L", type=int, default=None, help="Override num gaps (default from ILConfig).")
    ap.add_argument("--missing-gap-value", type=float, default=None, help="Override missing gap value (default from ILConfig).")
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--out", default="results/gap_audit_from_config.jsonl")
    ap.add_argument("--store-last-ts", action="store_true",
                    help="Track per-object timestamp inversions (uses RAM ~ O(#objects)).")
    ap.add_argument(
        "--ts-mode",
        choices=["raw", "counter", "relative-log", "monotonic"],
        default="raw",
        help=(
            "How to derive timestamps for gap computation: "
            "raw = use trace timestamps as-is; "
            "counter = use monotonically increasing counter (global); "
            "relative-log = log1p(max(0, ts - min_ts_slot)); "
            "monotonic = make raw timestamps non-decreasing."
        ),
    )
    ap.add_argument(
        "--summary-out",
        default=None,
        help="Optional path to write summary JSON.",
    )
    args = ap.parse_args()

    # resolve dataset config
    if not hasattr(EC, args.dataset):
        raise SystemExit(f"Unknown dataset '{args.dataset}'. Check names in src/config/experiment_config.py")
    ds = getattr(EC, args.dataset)

    L = args.L if args.L is not None else il_cfg.num_gaps
    missing = args.missing_gap_value if args.missing_gap_value is not None else il_cfg.missing_gap_value
    slot_size = args.slot_size if args.slot_size is not None else ds.slot_size

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    ft = FeatureTable(L=L, missing_gap_value=missing)

    last_ts = {} if args.store_last_ts else None
    inv_examples: List[Dict] = []
    MAX_EX = 30

    global_counter = 0
    prev_mono_ts: Optional[float] = None

    summary_slots = 0
    agg = {
        "gap_vec_neg_rate": 0.0,
        "gap_vec_order_viol_rate": 0.0,
        "end_lastgaps_neg_rate": 0.0,
        "end_lastgaps_order_viol_rate": 0.0,
        "max_end_lastgaps_neg_rate": 0.0,
        "max_end_lastgaps_order_viol_rate": 0.0,
    }

    with open(args.out, "w", encoding="utf-8") as f_out:
        for slot_idx, slot in enumerate(iter_slots_from_trace(path=ds.path, slot_size=slot_size, max_rows=args.max_rows), start=1):
            if slot_idx > args.slots:
                break

            # pre-compute slot-wide stats for ts normalization if needed
            if args.ts_mode == "relative-log":
                slot_ts_values = [float(req["timestamp"]) for req in slot if req.get("timestamp") is not None]
                slot_ts_min = min(slot_ts_values) if slot_ts_values else 0.0
            else:
                slot_ts_min = 0.0

            ts_none = 0
            global_back = 0
            prev_global_ts = None

            vec_total = 0
            vec_neg = 0
            vec_ord_viol = 0

            per_obj_seen = set()

            for req in slot:
                obj = str(req.get("object_id"))
                if args.ts_mode == "counter":
                    global_counter += 1
                    ts = float(global_counter)
                else:
                    ts_raw = req.get("timestamp", None)
                    if ts_raw is None:
                        ts_none += 1
                        continue
                    ts_raw = float(ts_raw)

                    if args.ts_mode == "relative-log":
                        ts = math.log1p(max(0.0, ts_raw - slot_ts_min))
                    elif args.ts_mode == "monotonic":
                        if prev_mono_ts is None:
                            ts = ts_raw
                        else:
                            ts = ts_raw if ts_raw >= prev_mono_ts else prev_mono_ts
                        prev_mono_ts = ts
                    else:
                        ts = ts_raw

                if prev_global_ts is not None and ts < prev_global_ts:
                    global_back += 1
                prev_global_ts = ts

                if last_ts is not None:
                    p = last_ts.get(obj)
                    if p is not None and ts < p and len(inv_examples) < MAX_EX:
                        inv_examples.append({"slot": slot_idx, "object_id": obj, "prev_ts": p, "ts": ts})
                    last_ts[obj] = ts

                gaps = ft.update_and_get_gaps(obj, ts)
                vec_total += 1
                if any(g < 0 for g in gaps):
                    vec_neg += 1
                if compute_order_violation_rate(gaps, missing):
                    vec_ord_viol += 1

                per_obj_seen.add(obj)

            # end-of-slot: check last_gaps used for training
            end_total = 0
            end_neg = 0
            end_ord_viol = 0
            values_by_dim = [[] for _ in range(L)]

            for obj in per_obj_seen:
                gaps = ft.get_last_gaps(obj)
                if gaps is None:
                    continue
                end_total += 1
                if any(g < 0 for g in gaps):
                    end_neg += 1
                if compute_order_violation_rate(gaps, missing):
                    end_ord_viol += 1
                for i in range(L):
                    values_by_dim[i].append(float(gaps[i]))

            rec = {
                "slot": slot_idx,
                "slot_size": len(slot),
                "objects_in_slot": len(per_obj_seen),
                "params": {"L": L, "missing_gap_value": missing, "slot_size_used": slot_size, "dataset": args.dataset},
                "ts_none": ts_none,
                "global_ts_backward": global_back,
                "gap_vec_total": vec_total,
                "gap_vec_neg_rate": (vec_neg / vec_total) if vec_total else None,
                "gap_vec_order_viol_rate": (vec_ord_viol / vec_total) if vec_total else None,
                "end_lastgaps_total": end_total,
                "end_lastgaps_neg_rate": (end_neg / end_total) if end_total else None,
                "end_lastgaps_order_viol_rate": (end_ord_viol / end_total) if end_total else None,
                "per_dim": slot_dim_stats(values_by_dim, missing),
            }
            f_out.write(json.dumps(rec) + "\n")
            summary_slots += 1
            for key in ("gap_vec_neg_rate", "gap_vec_order_viol_rate",
                        "end_lastgaps_neg_rate", "end_lastgaps_order_viol_rate"):
                agg[key] += float(rec[key] or 0.0)
            agg["max_end_lastgaps_neg_rate"] = max(
                agg["max_end_lastgaps_neg_rate"], float(rec["end_lastgaps_neg_rate"] or 0.0)
            )
            agg["max_end_lastgaps_order_viol_rate"] = max(
                agg["max_end_lastgaps_order_viol_rate"], float(rec["end_lastgaps_order_viol_rate"] or 0.0)
            )

            print(
                f"[slot {slot_idx:03d}] objs={len(per_obj_seen):6d} "
                f"neg(vec)={(rec['gap_vec_neg_rate'] or 0):.4f} "
                f"ord(vec)={(rec['gap_vec_order_viol_rate'] or 0):.4f} "
                f"neg(end)={(rec['end_lastgaps_neg_rate'] or 0):.4f} "
                f"ord(end)={(rec['end_lastgaps_order_viol_rate'] or 0):.4f} "
                f"ts_back(global)={global_back}"
            )

    if args.store_last_ts and inv_examples:
        ex_path = os.path.splitext(args.out)[0] + "_ts_inversions.json"
        with open(ex_path, "w", encoding="utf-8") as f:
            json.dump(inv_examples, f, indent=2)
        print(f"[audit] per-object timestamp inversions -> {ex_path}")

    if summary_slots > 0:
        summary = {
            "dataset": args.dataset,
            "ts_mode": args.ts_mode,
            "slots": summary_slots,
            "avg_gap_vec_neg_rate": agg["gap_vec_neg_rate"] / summary_slots,
            "avg_gap_vec_order_viol_rate": agg["gap_vec_order_viol_rate"] / summary_slots,
            "avg_end_lastgaps_neg_rate": agg["end_lastgaps_neg_rate"] / summary_slots,
            "avg_end_lastgaps_order_viol_rate": agg["end_lastgaps_order_viol_rate"] / summary_slots,
            "max_end_lastgaps_neg_rate": agg["max_end_lastgaps_neg_rate"],
            "max_end_lastgaps_order_viol_rate": agg["max_end_lastgaps_order_viol_rate"],
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
