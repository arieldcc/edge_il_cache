# src/config/experiment_config.py

from dataclasses import dataclass
from typing import List

@dataclass
class DatasetConfig:
    name: str
    path: str
    num_total_requests: int = 9_000_000
    num_warmup_requests: int = 1_000_000
    slot_size: int = 100_000

@dataclass
class ILConfig:
    num_gaps: int = 6       # Gap1..Gap6 (default)
    gap_truncation_mode: str = "off"  # "off", "capacity_aware"
    gap_truncation_small_ratio: float = 0.015
    gap_truncation_medium_ratio: float = 0.035
    gap_truncation_small_k: int = 2
    gap_truncation_medium_k: int = 3
    gap_truncation_large_k: int = 6
    pop_top_percent: float = 0.20
    sigmoid_a: float = 0.5
    sigmoid_b: float = 10.0
    sigmoid_b_wma: float = 20.0
    max_classifiers: int = 20  # implikasi dari b
    missing_gap_value: float = 1e6 # 1e6 / 2_592_000.0
    lambda_freq: float = 0.0
    lambda_freq_cap: float = 3.0
    use_freq_feature: bool = True
    freq_feature_mode: str = "log_norm"  # "log_norm" atau "log"
    freq_feature_source: str = "prev_slot"  # "prev_slot" atau "slot_total"
    label_mode: str = "capacity"  # "coverage", "top_ratio", "capacity"
    label_capacity_gamma: float = 1.2
    coverage_target: float = 0.6
    coverage_mode: str = "capacity_aware"  # "fixed" atau "capacity_aware"
    coverage_min: float = 0.55
    coverage_max: float = 0.75
    coverage_midpoint: float = 0.02
    coverage_slope: float = 0.01
    coverage_concentration_mode: str = "two_sided"  # "off", "one_sided", "two_sided"
    concentration_top_k_ratio: float = 0.01
    concentration_alpha: float = -1.0  # < 0 berarti auto (berdasarkan statistik dataset)
    concentration_min_objects: int = 0  # <= 0 berarti auto (berdasarkan statistik dataset)
    coverage_floor: float = 0.0  # <= 0 berarti auto (berdasarkan statistik dataset)
    coverage_ceiling: float = 0.0  # <= 0 berarti auto (berdasarkan statistik dataset)
    coverage_ema_alpha: float = 0.2
    gate_mode: str = "adaptive_coverage"  # "off", "fixed", "adaptive_coverage", "score_budget"
    gate_k_fixed: int = 1
    gate_ema_alpha: float = 0.0
    gate_target_coverage: float = -1.0  # <= 0 berarti pakai target coverage per slot
    gate_min_k: int = 0
    gate_max_k: int = 0  # <= 0 berarti pakai num_gaps
    score_budget_alpha: float = 1.0
    balance_capacity_adaptive: bool = True
    balance_ema_alpha: float = 0.2
    balance_ratio_cap: float = 3.0
    admit_budget_enabled: bool = True
    admit_rate_scale: float = 4.0
    admit_pos_rate_min: float = 0.005
    admit_pos_rate_max: float = 0.25
    admit_tau_sample_k: int = 5000
    admit_tau_ema_alpha: float = 0.3
    admit_guardrail_enabled: bool = True
    admit_guardrail_max_rate: float = 0.6
    admit_guardrail_k: float = 4.0
    admit_guardrail_floor: float = 0.20
    admit_guardrail_ceiling: float = 0.80
    admit_guardrail_action: str = "dampen"  # "dampen"
    admit_guardrail_dampen: float = 0.4
    admit_guardrail_signal: str = "churn"  # "admit_stream", "churn", "both"
    admit_guardrail_churn_k: float = 4.0
    admit_guardrail_churn_floor: float = 0.02
    admit_guardrail_churn_ceiling: float = 0.12
    admit_victim_enabled: bool = False
    admit_victim_margin: float = 0.0
    admit_eval_horizon_slots: int = 1
    admit_eval_k: float = -1.0  # <= 0 berarti pakai pop_top_percent
    admit_eval_scope: str = "all"  # "all", "warmup_only", "post_warmup"

@dataclass
class CacheConfig:
    # Mengikuti Xu et al.: cache capacity sebagai % dari jumlah objek distinc
    cache_size_percentages: List[float] = (0.8, 1.0, 2.0, 3.0, 4.0, 5.0)  # % dari total unique objects

@dataclass
class GDBTConfig:
    # Jumlah tree (iterations) = 30, sesuai [9] dan Xu et al.
    n_estimators: int = 30
    # Setiap 1M request, rebuild model dari scratch
    update_interval_requests: int = 1_000_000  # rebuild setiap 1M requests
    # tidak ada dituliskan di Xu et al maupun 
    # learning_rate: float = 0.1
    # max_depth: int = 3
    # min_samples_split: int = 2
    # min_samples_leaf: int = 1
    # random_state: int = 42

# === DATASET YANG DIPAKAI SAAT INI (RAW WIKIPEDIA) ===

WIKIPEDIA_SEPTEMBER_2007 = DatasetConfig(
    name="wikipedia_september_2007",
    path="data/raw/wikipedia_september_2007/wiki.1190153705.gz",
)

WIKIPEDIA_OKTOBER_2007 = DatasetConfig(
    name="wikipedia_oktober_2007",
    path="data/raw/wikipedia_oktober_2007/wiki.1191201596.gz",
)

WIKI2018 = DatasetConfig(
    name="wiki2018",
    path="data/raw/wiki2018/wiki2018.gz"
)
