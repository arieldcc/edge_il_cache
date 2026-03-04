# src/cache/cache_simulator.py

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CacheStats:
    """
    Menyimpan statistik dasar cache untuk satu percobaan.
    """
    capacity_objects: int
    total_requests: int = 0
    cache_hits: int = 0

    @property
    def hit_ratio(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.cache_hits / float(self.total_requests)