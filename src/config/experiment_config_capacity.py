# src/config/experiment_config.py

from dataclasses import dataclass
from typing import List

@dataclass
class DatasetConfig:
    name: str
    path: str
    num_total_requests: int = 10_000_000
    num_warmup_requests: int = 1_000_000
    slot_size: int = 100_000

@dataclass
class ILConfig:
    num_gaps: int = 6       # Gap1..Gap6 (default)
    pop_top_percent: float = 0.20
    sigmoid_a: float = 0.5
    sigmoid_b: float = 10.0
    sigmoid_b_wma: float = 20.0
    max_classifiers: int = 20  # implikasi dari b
    missing_gap_value: float = 1e6 # 1e6 / 2_592_000.0
    use_freq_feature: bool = True
    freq_feature_mode: str = "log_norm"  # "log_norm" atau "log"
    freq_feature_source: str = "prev_slot"  # "prev_slot" atau "slot_total"
    label_mode: str = "capacity"  # "coverage", "top_ratio", "capacity"
    label_capacity_gamma: float = 1.2
    lambda_freq: float = 1.0
    lambda_freq_cap: float = 3.0
    class_mass_balance: bool = True
    balance_ema_alpha: float = 0.2
    balance_ratio_cap: float = 3.0
    admit_use_tau: bool = True
    admit_tau_ema_alpha: float = 0.2
    admit_slot_max_ratio: float = 0.2  # 0.0 = off
    admit_slot_max_abs: int = 0  # >0 = hard cap
    admit_budget_mode: str = "soft"  # "soft" atau "hard"
    admit_budget_margin: float = 0.0  # margin score untuk soft budget
    gate_min_hist_len: int = 0  # 0 berarti tanpa gate
    warmup_prefill: bool = True
    tau_use_candidate_reservoir: bool = True
    tau_score_sample_size: int = 4096
    tau_capacity_gamma: float = 1.0
    tau_pos_rate_min: float = 0.002
    tau_pos_rate_max: float = 0.30
    seed: int = 42
    slot_log_mode: str = "light"  # "light" atau "full"
    slot_log_sample_k: int = 0

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
