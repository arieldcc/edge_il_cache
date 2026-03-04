# src/cache/lru2.py

from collections import OrderedDict
from typing import Any, Dict


class LRU2Cache:
    """
    Implementasi sederhana LRU-K dengan K=2.

    Ide:
    - Underlying eviction tetap LRU (OrderedDict).
    - Admission rule: objek baru hanya masuk cache setelah referensi ke-2 (K=2).

    Asumsi:
    - freq[obj_id] adalah counter global (tidak direset per slot).
    """

    def __init__(self, capacity_objects: int) -> None:
        self.capacity = max(1, int(capacity_objects))
        self.data: "OrderedDict[str, Any]" = OrderedDict()
        self.freq: Dict[str, int] = {}

    def __len__(self) -> int:
        return len(self.data)

    def contains(self, obj_id: str) -> bool:
        return obj_id in self.data

    def access(self, obj_id: str) -> bool:
        """
        Akses cache:
        - Hit: True + update urutan LRU.
        - Miss: False, cache tidak diubah (insert dipanggil terpisah).
        """
        if obj_id in self.data:
            # Jadikan paling recently used.
            self.data.move_to_end(obj_id, last=True)
            return True
        return False

    def insert(self, obj_id: str) -> None:
        """
        Admission rule:
        - freq[obj_id] di-increment.
        - Hanya jika freq >= 2 objek benar-benar dimasukkan ke cache.
        - Eviction tetap LRU (popitem(last=False)).
        """
        f = self.freq.get(obj_id, 0) + 1
        self.freq[obj_id] = f

        # Belum mencapai referensi ke-2 → jangan masuk cache dulu.
        if f < 2:
            return

        # Jika sudah ada di cache, cukup refresh posisi LRU.
        if obj_id in self.data:
            self.data.move_to_end(obj_id, last=True)
            return

        # Jika penuh, evict LRU.
        if len(self.data) >= self.capacity:
            self.data.popitem(last=False)

        self.data[obj_id] = None
