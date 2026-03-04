from __future__ import annotations

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.experiments import run_il_cache_opt022_guard_only as il

# Small, controlled sweep (edit if needed)
QUALITY_CONFIGS = [
    # baseline (current)
    {"q": 0.9, "ema": 0.2, "qmin": 0.7, "qmax": 1.3, "qboost": 0.1},
    # more aggressive down-weight when scores are weak
    {"q": 0.9, "ema": 0.2, "qmin": 0.5, "qmax": 1.3, "qboost": 0.1},
    # wider range (allows stronger damping)
    {"q": 0.9, "ema": 0.2, "qmin": 0.5, "qmax": 1.5, "qboost": 0.1},
    # smoother EMA (less jitter)
    {"q": 0.9, "ema": 0.1, "qmin": 0.7, "qmax": 1.3, "qboost": 0.1},
    # more reactive EMA (more jitter, stronger effect)
    {"q": 0.9, "ema": 0.3, "qmin": 0.7, "qmax": 1.3, "qboost": 0.1},
]

# Gate toggle: set to 0.05 (on) or 1.0 (off)
GATE_TOP_PERCENT = 0.05


def run_one(cfg: dict) -> None:
    il.SCORE_SPREAD_Q = cfg["q"]
    il.SCORE_SPREAD_EMA_ALPHA = cfg["ema"]
    il.SCORE_QUALITY_MIN = cfg["qmin"]
    il.SCORE_QUALITY_MAX = cfg["qmax"]
    il.SCORE_QUALITY_MIN_BOOST = cfg["qboost"]
    il.SCORE_GATE_TOP_PERCENT = GATE_TOP_PERCENT

    tag = (
        f"q{cfg['q']}_ema{cfg['ema']}"
        f"_min{cfg['qmin']}_max{cfg['qmax']}_boost{cfg['qboost']}"
    )
    model_name = f"ilnse_A2_guard_qsweep_{tag}_gate{GATE_TOP_PERCENT}"

    for ds in (il.WIKIPEDIA_SEPTEMBER_2007, il.WIKI2018):
        il.run_experiment(ds, feature_set="A2", model_name=model_name)



def main() -> None:
    for cfg in QUALITY_CONFIGS:
        run_one(cfg)


if __name__ == "__main__":
    main()
