# src/cache/lru.py

from collections import OrderedDict


class LRUCache:
    """
    Implementasi LRU cache sederhana berbasis OrderedDict.

    Asumsi:
    - Kapasitas dinyatakan dalam jumlah objek (semua objek dianggap sama ukuran).
    - Key: object_id (string dari trace).
    - Value: tidak dipakai (hanya presence/absence), bisa None.

    Kebijakan:
    - Hit: objek dipindah ke posisi paling baru (MRU).
    - Miss: jika penuh, evict objek yang paling lama tidak diakses (LRU).
    """

    def __init__(self, capacity_objects: int) -> None:
        if capacity_objects < 0:
            raise ValueError("capacity_objects must be non-negative")
        self.capacity = int(capacity_objects)
        self._data: "OrderedDict[str, None]" = OrderedDict()

    def __len__(self) -> int:
        return len(self._data)

    def contains(self, obj_id: str) -> bool:
        return obj_id in self._data

    def access(self, obj_id: str) -> bool:
        """
        Akses objek di cache.
        - Jika hit: kembalikan True dan update LRU (move_to_end).
        - Jika miss: kembalikan False, cache tidak diubah.
        """
        if obj_id in self._data:
            # Jadikan paling recently used.
            self._data.move_to_end(obj_id, last=True)
            return True
        return False

    def insert(self, obj_id: str) -> None:
        """
        Masukkan objek ke cache dengan kebijakan LRU:
        - Jika kapasitas 0 → tidak melakukan apa-apa.
        - Jika obj_id sudah ada → hanya menyegarkan posisi LRU.
        - Jika penuh → evict entry paling lama (first).
        """
        if self.capacity <= 0:
            return

        if obj_id in self._data:
            # Refresh recency.
            self._data.move_to_end(obj_id, last=True)
            return

        if len(self._data) >= self.capacity:
            # Evict LRU (paling kiri).
            self._data.popitem(last=False)

        self._data[obj_id] = None

    def peek_lru(self) -> str | None:
        """
        Mengembalikan object_id LRU tanpa mengubah urutan cache.
        """
        if not self._data:
            return None
        return next(iter(self._data))

    def insert_with_eviction(self, obj_id: str) -> tuple[bool, str | None]:
        """
        Masukkan objek dan kembalikan apakah insert terjadi + id yang dievict.
        """
        if self.capacity <= 0:
            return False, None

        if obj_id in self._data:
            self._data.move_to_end(obj_id, last=True)
            return False, None

        evicted_id = None
        if len(self._data) >= self.capacity:
            evicted_id, _ = self._data.popitem(last=False)

        self._data[obj_id] = None
        return True, evicted_id
