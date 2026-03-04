# src/config/experiment_config_all_ds.py
#
# Minimal Xu-style NSE config + tau gate (online-valid).
# Keep only parameters that are explicitly used by the Xu model or tau gating.

from dataclasses import dataclass, field
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
    # Xu-style core
    num_gaps: int = 6  # Gap1..Gap6
    sigmoid_a: float = 0.5
    sigmoid_b: float = 10.0
    max_classifiers: int = 20
    missing_gap_value: float = 1e6
    pop_top_percent: float = 0.20
    label_mode: str = "top_ratio"  # "coverage", "top_ratio", "capacity"
    label_capacity_gamma: float = 1.2

    # Online-valid feature source
    use_freq_feature: bool = True
    freq_feature_mode: str = "log_norm"  # "log_norm" atau "log"
    freq_feature_source: str = "prev_slot"  # "prev_slot" atau "slot_total"

    # Tau gate (single control mechanism)
    admit_use_tau: bool = True
    admit_tau_ema_alpha: float = 0.2
    tau_use_candidate_reservoir: bool = True
    tau_score_sample_size: int = 4096
    tau_reservoir_min_samples: int = 512
    tau_reservoir_fallback: str = "ema"  # "ema", "D_t", atau "none"
    tau_capacity_gamma: float = 1.0
    tau_pos_rate_min: float = 0.002
    tau_pos_rate_max: float = 0.30

    # Score-aware eviction (NSE signal) + score refresh budget
    victim_sample_k: int = 32
    cache_score_refresh_k: int = 256
    cache_score_stale_slots: int = 1

    # misc / logging
    seed: int = 42
    slot_log_mode: str = "full"  # "light" atau "full"
    slot_log_sample_k: int = 0

@dataclass
class CacheConfig:
    # Mengikuti Xu et al.: cache capacity sebagai % dari jumlah objek distinc
    cache_size_percentages: List[float] = (0.8, 1.0, 2.0, 3.0, 4.0, 5.0)  # % dari total unique objects
    cache_size_percentages_eval: List[float] = field(default_factory=list)

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
    num_total_requests=9_000_000,
)

WIKIPEDIA_OKTOBER_2007 = DatasetConfig(
    name="wikipedia_oktober_2007",
    path="data/raw/wikipedia_oktober_2007/wiki.1191201596.gz",
    num_total_requests=9_000_000,
)

WIKI2018 = DatasetConfig(
    name="wiki2018",
    path="data/raw/wiki2018/wiki2018.gz",
    num_total_requests=10_000_000,
)

# 85
