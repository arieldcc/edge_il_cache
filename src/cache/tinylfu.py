# src/cache/tinylfu.py

from __future__ import annotations

from collections import OrderedDict


class TinyLFUCache:
    """
    TinyLFU-style admission + LRU eviction.

    - Maintains a Count-Min Sketch (CMS) for approximate frequency.
    - Optional doorkeeper (1-bit bloom) to filter one-hit wonders.
    - Admission: admit candidate only if freq(candidate) >= freq(LRU victim).
    - Eviction: LRU among admitted objects.

    Note: access() updates the frequency. insert() performs admission/eviction.
    """

    def __init__(
        self,
        capacity_objects: int,
        cms_width: int = 16384,
        cms_depth: int = 4,
        doorkeeper_bits: int = 65536,
        reset_interval: int | None = None,
    ) -> None:
        if capacity_objects < 0:
            raise ValueError("capacity_objects must be non-negative")
        self.capacity = int(capacity_objects)
        self._data: "OrderedDict[str, None]" = OrderedDict()

        self.cms_width = int(cms_width)
        self.cms_depth = int(cms_depth)
        self._cms = [
            [0 for _ in range(self.cms_width)] for _ in range(self.cms_depth)
        ]

        self.doorkeeper_bits = int(doorkeeper_bits)
        self._doorkeeper = bytearray((self.doorkeeper_bits + 7) // 8)

        if reset_interval is None:
            reset_interval = max(100_000, self.capacity * 10)
        self.reset_interval = int(reset_interval)
        self._access_count = 0

    def __len__(self) -> int:
        return len(self._data)

    def contains(self, obj_id: str) -> bool:
        return obj_id in self._data

    def access(self, obj_id: str) -> bool:
        """
        Update frequency. If hit, refresh LRU order.
        Returns True on hit, False on miss.
        """
        self._record(obj_id)
        if obj_id in self._data:
            self._data.move_to_end(obj_id, last=True)
            return True
        return False

    def insert(self, obj_id: str) -> None:
        """
        Admission + LRU eviction:
        - If already in cache, refresh.
        - If doorkeeper rejects (first-time), skip admission.
        - If cache full, compare CMS freq(candidate) vs CMS freq(victim).
        """
        if self.capacity <= 0:
            return

        if obj_id in self._data:
            self._data.move_to_end(obj_id, last=True)
            return

        if not self._doorkeeper_test_and_set(obj_id):
            return

        if len(self._data) < self.capacity:
            self._data[obj_id] = None
            return

        victim_id = next(iter(self._data))
        if self._estimate(obj_id) >= self._estimate(victim_id):
            self._data.popitem(last=False)
            self._data[obj_id] = None

    def peek_lru(self) -> str | None:
        if not self._data:
            return None
        return next(iter(self._data))

    def _record(self, obj_id: str) -> None:
        h = self._hash64(obj_id)
        for i in range(self.cms_depth):
            idx = (h + i * 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
            pos = idx % self.cms_width
            self._cms[i][pos] += 1

        self._access_count += 1
        if self.reset_interval > 0 and self._access_count >= self.reset_interval:
            self._decay()
            self._access_count = 0

    def _estimate(self, obj_id: str) -> int:
        h = self._hash64(obj_id)
        mins = None
        for i in range(self.cms_depth):
            idx = (h + i * 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
            pos = idx % self.cms_width
            val = self._cms[i][pos]
            mins = val if mins is None else min(mins, val)
        return int(mins or 0)

    def _doorkeeper_test_and_set(self, obj_id: str) -> bool:
        h = self._hash64(obj_id)
        bit_idx = h % self.doorkeeper_bits
        byte_idx = bit_idx >> 3
        mask = 1 << (bit_idx & 7)
        if self._doorkeeper[byte_idx] & mask:
            return True
        self._doorkeeper[byte_idx] |= mask
        return False

    def _decay(self) -> None:
        for row in self._cms:
            for i in range(len(row)):
                row[i] >>= 1
        self._doorkeeper = bytearray(len(self._doorkeeper))

    @staticmethod
    def _hash64(obj_id: str) -> int:
        data = str(obj_id).encode("utf-8", errors="ignore")
        h = 1469598103934665603  # FNV-1a 64-bit offset basis
        for b in data:
            h ^= b
            h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
        return h
