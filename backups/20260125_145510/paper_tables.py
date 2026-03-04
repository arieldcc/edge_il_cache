#!/usr/bin/env python3
"""Generate dataset and model parameter tables for the paper.

Usage examples:
  python scripts/paper_tables.py --distinct --format md
  python scripts/paper_tables.py --distinct --out-dir results/paper_tables
"""

from __future__ import annotations

import argparse
import ast
import gzip
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _format_value(val: Any) -> str:
    if isinstance(val, float):
        # keep consistent, but avoid trailing zeros noise
        return f"{val:.6g}"
    if isinstance(val, (list, tuple)):
        return ", ".join(_format_value(v) for v in val)
    if isinstance(val, dict):
        # compact dict: key=value
        return ", ".join(f"{k}={_format_value(v)}" for k, v in val.items())
    return str(val)


def _md_table(headers: List[str], rows: List[List[Any]]) -> str:
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(_format_value(v) for v in row) + " |")
    return "\n".join(lines)


def _csv_table(headers: List[str], rows: List[List[Any]]) -> str:
    lines = []
    lines.append(",".join(headers))
    for row in rows:
        lines.append(",".join(_format_value(v) for v in row))
    return "\n".join(lines)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _load_opt_constants() -> Dict[str, Any]:
    """Parse constants from src/experiments/run_il_cache_opt.py without importing it."""
    project_root = Path(__file__).resolve().parents[1]
    opt_path = project_root / "src" / "experiments" / "run_il_cache_opt.py"
    source = opt_path.read_text(encoding="utf-8")
    tree = ast.parse(source)

    wanted = {
        "FEATURE_SETS",
        "FEATURE_WINDOWS",
        "POLLUTION_WINDOW_SLOTS",
        "WIKIPEDIA_SEPTEMBER_2007",
        "WIKIPEDIA_OKTOBER_2007",
        "WIKI2018",
        "IL_NUM_GAPS",
        "IL_POP_TOP_PERCENT",
        "IL_LABEL_TOPK_ROUNDING",
        "IL_LABEL_TIE_BREAK",
        "IL_SIGMOID_A",
        "IL_SIGMOID_B",
        "IL_MAX_CLASSIFIERS",
        "IL_MISSING_GAP_VALUE",
        "CACHE_SIZE_PERCENTAGES",
        "ADMISSION_CAPACITY_ALPHA",
        "DRIFT_ALPHA_MIN",
        "DRIFT_ALPHA_MAX",
        "DRIFT_SENSITIVITY",
        "DRIFT_EMA_ALPHA",
        "DRIFT_STATS_ALPHA",
        "DRIFT_NORM_EPS",
        "DRIFT_Z_CLIP",
        "DRIFT_WEIGHT_JSD",
        "DRIFT_WEIGHT_OVERLAP",
        "DRIFT_GAIN",
        "DRIFT_ALPHA_FLOOR_MULT",
        "SCORE_GATE_TOP_PERCENT",
        "FILL_RATIO",
        "FILL_RATE",
        "PRESSURE_MISS_GAMMA",
        "SCORE_SPREAD_Q",
        "SCORE_SPREAD_EMA_ALPHA",
        "SCORE_SPREAD_EPS",
        "SCORE_QUALITY_MIN",
        "SCORE_QUALITY_MAX",
        "SCORE_QUALITY_MIN_BOOST",
        "ADMISSION_PRECISION_TARGET",
        "ADMISSION_PRECISION_SENSITIVITY",
        "CAPACITY_ALPHA_SCALE_MIN",
    }

    consts: Dict[str, Any] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            if len(node.targets) != 1:
                continue
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in wanted:
                try:
                    consts[target.id] = ast.literal_eval(node.value)
                except Exception:
                    # Skip non-literal values
                    pass
    return consts


def _dataset_map(consts: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {
        "september2007": consts["WIKIPEDIA_SEPTEMBER_2007"],
        "oktober2007": consts["WIKIPEDIA_OKTOBER_2007"],
        "wiki2018": consts["WIKI2018"],
    }


def _parse_raw_wikibench_line(line: str) -> Optional[str]:
    parts = line.split()
    if len(parts) < 4:
        return None
    # parts[2:-1] may contain URL tokens
    return " ".join(parts[2:-1])


def _count_distinct(path: str, total_requests: int) -> Optional[int]:
    if not os.path.exists(path):
        return None
    count = 0
    seen: set[str] = set()
    try:
        with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if count >= total_requests:
                    break
                line = line.strip()
                if not line:
                    continue
                obj_id = None
                if path.endswith("wiki2018.gz"):
                    parts = line.split()
                    if len(parts) >= 2:
                        obj_id = parts[1]
                else:
                    obj_id = _parse_raw_wikibench_line(line)
                if obj_id is None:
                    continue
                seen.add(obj_id)
                count += 1
    except Exception:
        return None
    return len(seen)


def _dataset_rows(consts: Dict[str, Any], names: List[str], compute_distinct: bool) -> Tuple[List[str], List[List[Any]]]:
    headers = [
        "Trace",
        "Requests",
        "DistinctObjects",
    ]
    rows: List[List[Any]] = []
    ds_map = _dataset_map(consts)
    for name in names:
        if name not in ds_map:
            raise ValueError(f"Unknown dataset key: {name}. Use one of {list(ds_map.keys())}.")
        cfg = ds_map[name]
        total_requests = int(cfg["num_total_requests"])
        distinct = _count_distinct(cfg["path"], total_requests) if compute_distinct else None
        rows.append(
            [
                cfg["name"],
                total_requests,
                distinct if distinct is not None else "(not computed)",
            ]
        )
    return headers, rows


def _feature_dim(num_gaps: int, feature_set: str) -> int:
    if feature_set == "A0":
        return num_gaps
    if feature_set == "A1":
        return num_gaps + 1
    if feature_set == "A2":
        return num_gaps + 4
    if feature_set == "A3":
        return num_gaps + 5
    raise ValueError(f"Unknown feature_set: {feature_set}")


def _param_rows(consts: Dict[str, Any]) -> Tuple[List[str], List[List[Any]]]:
    headers = ["Symbol", "Description", "Value"]

    L = consts["IL_NUM_GAPS"]
    rho = consts["IL_POP_TOP_PERCENT"]
    windows = consts["FEATURE_WINDOWS"]

    rows: List[List[Any]] = [
        ["L", "Number of gap/recency features", L],
        ["ρ", "Top-K labeling ratio (K = floor(ρ·N_t))", rho],
        ["a", "Learn++.NSE sigmoid slope", consts["IL_SIGMOID_A"]],
        ["b", "Learn++.NSE sigmoid shift", consts["IL_SIGMOID_B"]],
        ["H_max", "Max learners (prune oldest)", consts["IL_MAX_CLASSIFIERS"]],
        ["g_miss", "Missing-gap padding value", consts["IL_MISSING_GAP_VALUE"]],
        ["w_short", "Short frequency window (slots)", windows["short"]],
        ["w_mid", "Mid frequency window (slots)", windows["mid"]],
        ["w_long", "Long frequency window (slots)", windows["long"]],
        ["w_cum", "Cumulative frequency window", "all past slots"],
        ["C (%)", "Cache sizes as % of distinct objects", list(consts["CACHE_SIZE_PERCENTAGES"])],
        ["α_0", "Base admission rate", consts["ADMISSION_CAPACITY_ALPHA"]],
        ["α_min", "Min admission rate", consts["DRIFT_ALPHA_MIN"]],
        ["α_max", "Max admission rate", consts["DRIFT_ALPHA_MAX"]],
        ["β_d", "Drift EMA (d̄_t)", consts["DRIFT_EMA_ALPHA"]],
        ["β_stats", "Drift mean/var EMA", consts["DRIFT_STATS_ALPHA"]],
        ["w_JSD", "Drift weight for JSD", consts["DRIFT_WEIGHT_JSD"]],
        ["w_OV", "Drift weight for overlap", consts["DRIFT_WEIGHT_OVERLAP"]],
        ["g_d", "Drift gain", consts["DRIFT_GAIN"]],
        ["α_floor", "Alpha floor multiplier", consts["DRIFT_ALPHA_FLOOR_MULT"]],
        ["γ_miss", "Miss-rate pressure coefficient", consts["PRESSURE_MISS_GAMMA"]],
        ["q", "Score-spread quantile", consts["SCORE_SPREAD_Q"]],
        ["β_s", "Score-spread EMA", consts["SCORE_SPREAD_EMA_ALPHA"]],
        ["q_min", "Min quality multiplier", consts["SCORE_QUALITY_MIN"]],
        ["q_max", "Max quality multiplier", consts["SCORE_QUALITY_MAX"]],
        ["δ_q", "Quality min-boost (capacity effect)", consts["SCORE_QUALITY_MIN_BOOST"]],
        ["p_target", "Admission precision target", consts["ADMISSION_PRECISION_TARGET"]],
        ["κ_p", "Precision sensitivity", consts["ADMISSION_PRECISION_SENSITIVITY"]],
        ["κ_c,min", "Min capacity scale", consts["CAPACITY_ALPHA_SCALE_MIN"]],
        ["ϕ_gate", "Score-gate top percentile", consts["SCORE_GATE_TOP_PERCENT"]],
        ["ϕ_fill", "Fill-phase ratio", consts["FILL_RATIO"]],
        ["r_fill", "Fill-phase minimum rate", consts["FILL_RATE"]],
        ["W_poll", "Pollution window (slots)", consts["POLLUTION_WINDOW_SLOTS"]],
        ["Base learner", "Classifier used in ensemble", "GaussianNaiveBayes"],
        [
            "Feature set (gap-only)",
            "x = [gap1..gapL], dim L (A0 ablation)",
            _feature_dim(L, "A0"),
        ],
        [
            "Feature set (gap + multi-timescale freq)",
            "x = [gap1..gapL, f_short, f_mid, f_long, f_cum], dim L+4 (A2 ablation)",
            _feature_dim(L, "A2"),
        ],
    ]

    return headers, rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate tables for paper.")
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=["september2007", "wiki2018"],
        help="Datasets to include: september2007, oktober2007, wiki2018",
    )
    parser.add_argument(
        "--distinct",
        action="store_true",
        help="Compute distinct object counts by scanning trace.",
    )
    parser.add_argument(
        "--format",
        choices=["md", "csv", "json"],
        default="md",
        help="Output format for stdout.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Optional output directory to write table files.",
    )
    args = parser.parse_args()

    consts = _load_opt_constants()

    ds_headers, ds_rows = _dataset_rows(consts, args.datasets, args.distinct)
    param_headers, param_rows = _param_rows(consts)

    if args.format == "md":
        print("## Table 1. Trace Summary")
        print(_md_table(ds_headers, ds_rows))
        print("\n## Table 2. Model/Control Parameters")
        print(_md_table(param_headers, param_rows))
    elif args.format == "csv":
        print("# Table 1. Trace Summary")
        print(_csv_table(ds_headers, ds_rows))
        print("\n# Table 2. Model/Control Parameters")
        print(_csv_table(param_headers, param_rows))
    else:
        payload = {
            "table1_trace_summary": {
                "headers": ds_headers,
                "rows": ds_rows,
            },
            "table2_model_params": {
                "headers": param_headers,
                "rows": param_rows,
            },
        }
        print(json.dumps(payload, indent=2))

    if args.out_dir:
        _ensure_dir(args.out_dir)
        if args.format == "json":
            out_path = os.path.join(args.out_dir, "paper_tables.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "table1_trace_summary": {
                            "headers": ds_headers,
                            "rows": ds_rows,
                        },
                        "table2_model_params": {
                            "headers": param_headers,
                            "rows": param_rows,
                        },
                    },
                    f,
                    indent=2,
                )
        else:
            t1_path = os.path.join(args.out_dir, "table1_trace_summary." + args.format)
            t2_path = os.path.join(args.out_dir, "table2_model_params." + args.format)
            with open(t1_path, "w", encoding="utf-8") as f:
                if args.format == "md":
                    f.write(_md_table(ds_headers, ds_rows) + "\n")
                else:
                    f.write(_csv_table(ds_headers, ds_rows) + "\n")
            with open(t2_path, "w", encoding="utf-8") as f:
                if args.format == "md":
                    f.write(_md_table(param_headers, param_rows) + "\n")
                else:
                    f.write(_csv_table(param_headers, param_rows) + "\n")


if __name__ == "__main__":
    main()
