# src/config/experiment_config.py

from dataclasses import dataclass, replace
from typing import List, Tuple

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
    pop_top_percent: float = 0.20
    label_topk_rounding: str = "floor"  # "floor" atau "ceil"
    label_tie_break: str = "none"  # "none" atau "include_ties"
    sigmoid_a: float = 0.5
    sigmoid_b: float = 10.0
    sigmoid_b_wma: float = 20.0
    max_classifiers: int = 20  # implikasi dari b
    missing_gap_value: float = 1e6 # 1e6 / 2_592_000.0
    
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
    

# === DATASET YANG DIPAKAI SAAT INI (RAW WIKIPEDIA) ===

WIKIPEDIA_SEPTEMBER_2007 = DatasetConfig(
    name="wikipedia_september_2007",
    path="data/raw/wikipedia_september_2007/wiki.1190153705.gz",
    num_total_requests=9_000_000,
    num_warmup_requests=1_000_000,
    slot_size=100_000,
)

WIKIPEDIA_OKTOBER_2007 = DatasetConfig(
    name="wikipedia_oktober_2007",
    path="data/raw/wikipedia_oktober_2007/wiki.1191201596.gz",
    num_total_requests=8_000_000,
    num_warmup_requests=1_000_000,
    slot_size=100_000,
)

WIKI2018 = DatasetConfig(
    name="wiki2018",
    path="data/raw/wiki2018/wiki2018.gz",
    num_total_requests=10_000_000,
    num_warmup_requests=1_000_000,
    slot_size=100_000,
)
