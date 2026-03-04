# src/cache/lfuda.py

import heapq
import itertools
from typing import Dict, Optional


class LFUDACache:
    """
    LFU dengan Dynamic Aging (LFU-DA) sesuai HPL / Squid (tanpa faktor size).

    - Setiap objek:
        freq: frekuensi akses
        key : freq + age

    - Global:
        age : nilai aging global

    - Aturan:
        - Eviction: pilih objek dengan key terkecil; set age = key_evict.
        - Objek baru: freq = 1, key = age + 1.
        - Hit: freq++, key = age + freq.

    Implementasi:
        - self.data: obj_id -> {"freq": int, "key": float}
        - self.heap: min-heap of (key, counter, obj_id) dengan lazy deletion
    """

    def __init__(self, capacity_objects: int) -> None:
        self.capacity = max(1, int(capacity_objects))
        self.age: float = 0.0
        self.data: Dict[str, Dict[str, float]] = {}
        self.heap = []  # list of (key, counter, obj_id)
        self._counter = itertools.count()  # tie-breaker agar heap stabil

    def __len__(self) -> int:
        return len(self.data)

    def contains(self, obj_id: str) -> bool:
        return obj_id in self.data

    def access(self, obj_id: str) -> bool:
        """
        Hit/miss:
        - jika hit: update freq & key, return True.
        - jika miss: return False (insert dipanggil terpisah).
        """
        entry = self.data.get(obj_id)
        if entry is None:
            return False

        entry["freq"] += 1
        entry["key"] = self.age + entry["freq"]
        # masukkan entri baru ke heap (lazy deletion untuk entri lama)
        heapq.heappush(
            self.heap,
            (entry["key"], next(self._counter), obj_id),
        )
        return True

    def _evict_one(self) -> None:
        """
        Evict satu objek dengan key terkecil (valid), update self.age.
        Menggunakan lazy deletion:
        - heappop sampai menemukan (obj_id, key) yang masih cocok dengan self.data.
        """
        while self.heap:
            key, _, obj_id = heapq.heappop(self.heap)
            entry = self.data.get(obj_id)
            if entry is None:
                # objek sudah tidak ada di data -> entri heap kadaluarsa
                continue
            if entry["key"] != key:
                # key di heap sudah lama (objek sudah punya key baru)
                continue

            # ini korban yang sah
            self.age = key
            del self.data[obj_id]
            return

        # kalau sampai sini, heap kosong atau semua entri kadaluarsa
        # berarti tidak ada yang bisa di-evict (kasus edge)
        return

    def insert(self, obj_id: str) -> None:
        """
        Insert objek baru:
        - jika sudah ada, treat as access (freq++, key update).
        - jika penuh, evict dulu.
        """
        entry = self.data.get(obj_id)
        if entry is not None:
            # treat as access
            entry["freq"] += 1
            entry["key"] = self.age + entry["freq"]
            heapq.heappush(
                self.heap,
                (entry["key"], next(self._counter), obj_id),
            )
            return

        if len(self.data) >= self.capacity:
            self._evict_one()

        new_entry = {"freq": 1, "key": self.age + 1.0}
        self.data[obj_id] = new_entry
        heapq.heappush(
            self.heap,
            (new_entry["key"], next(self._counter), obj_id),
        )
