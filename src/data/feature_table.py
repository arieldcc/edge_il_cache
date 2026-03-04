# src/data/feature_table.py

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Tuple, Iterable, Optional


@dataclass
class ObjectStats:
    """
    Menyimpan informasi per object_id:
      - timestamps: riwayat timestamp terakhir (paling baru di depan)
      - freq: frekuensi di slot saat ini
      - last_gaps: vektor Gap1..GapL dari request terakhir di slot ini
    """
    timestamps: Deque[float] = field(default_factory=deque)
    freq: int = 0
    last_gaps: Optional[List[float]] = None


class FeatureTable:
    """
    Feature table seperti dijelaskan di artikel IL-based edge caching.

    Untuk setiap object_id, kita simpan:
      - circular queue berisi L timestamp terakhir
      - frequency counter per slot
      - last_gaps: fitur Gap1..GapL yang dihitung saat request terakhir

    Desain ini:
      - L = batas MAKSIMAL jumlah gap yang dipakai.
      - Setiap request SELALU menghasilkan vektor fitur panjang L:
          Gap_i = t_now - t_i  untuk semua riwayat yang ada (<= L),
          sisanya dipad dengan missing_gap_value.
    """

    def __init__(self, L: int = 6, missing_gap_value: float = 0.0) -> None:
        """
        :param L: jumlah gap (L = 6 di eksperimen utama artikel)
        :param missing_gap_value:
            nilai pengganti jika riwayat timestamp kurang dari L
            (bisa 0.0 atau nilai besar, sesuai konfigurasi eksperimen).
        """
        self.L = L
        self.missing_gap_value = missing_gap_value
        self._table: Dict[str, ObjectStats] = {}

    def _get_or_create(self, object_id: str) -> ObjectStats:
        if object_id not in self._table:
            # timestamps disusun dengan yang terbaru di depan (left side deque)
            self._table[object_id] = ObjectStats(
                timestamps=deque(maxlen=self.L),
                freq=0,
                last_gaps=None,
            )
        return self._table[object_id]

    def update_and_get_gaps(self, object_id: str, timestamp: float) -> List[float]:
        """
        Dipanggil setiap ada request (object_id, timestamp).

        Langkah:
          1. Ambil riwayat timestamps lama (sebelum request ini).
          2. Hitung Gap1..GapL dari timestamp sekarang ke riwayat lama
             (maksimal L riwayat).
          3. Jika riwayat < L, sisa posisi dipad dengan missing_gap_value.
          4. Simpan gaps sebagai last_gaps.
          5. Update riwayat timestamps dengan timestamp sekarang.
          6. Tambah freq untuk slot sekarang.

        :return: list panjang L berisi Gap1..GapL (float).
        """
        stats = self._get_or_create(object_id)

        gaps: List[float] = []

        # timestamps di stats.timestamps berisi riwayat sebelum request ini
        # diasumsikan urut: [t_prev1 (terbaru), t_prev2, ..., t_prevK]
        for t_prev in stats.timestamps:
            gaps.append(timestamp - t_prev)
            if len(gaps) >= self.L:
                break

        # Jika riwayat kurang dari L, pad dengan missing_gap_value
        if len(gaps) < self.L:
            gaps.extend([self.missing_gap_value] * (self.L - len(gaps)))

        # Simpan sebagai last_gaps (dipakai nanti saat labeling)
        stats.last_gaps = gaps

        # Update riwayat timestamps: masukkan timestamp terbaru di depan
        # deque dengan maxlen=L akan otomatis menghapus yang paling lama jika penuh
        stats.timestamps.appendleft(timestamp)

        # Update frekuensi di slot ini
        stats.freq += 1

        return gaps

    def get_last_gaps(self, object_id: str) -> Optional[List[float]]:
        """
        Mengambil Gap1..GapL terakhir yang dihitung untuk object_id ini.
        Biasanya dipanggil di akhir slot saat membentuk D_t dan label popularitas.

        :return: list panjang L atau None jika object belum pernah muncul.
        """
        stats = self._table.get(object_id)
        if stats is None:
            return None
        return stats.last_gaps

    def get_freq(self, object_id: str) -> int:
        """
        Mengambil frekuensi object_id di slot saat ini.
        """
        stats = self._table.get(object_id)
        if stats is None:
            return 0
        return stats.freq

    def iter_freq_items(self) -> Iterable[Tuple[str, int]]:
        """
        Menghasilkan (object_id, freq) untuk semua objek
        yang pernah muncul (freq bisa 0 jika sudah di-reset).
        """
        for obj_id, stats in self._table.items():
            yield obj_id, stats.freq

    def reset_freqs(self) -> None:
        """
        Mereset frequency counter semua objek di akhir slot.
        (last_gaps dan timestamps TIDAK direset, karena itu memuat
        riwayat yang dipakai untuk slot berikutnya.)
        """
        for stats in self._table.values():
            stats.freq = 0

    def num_objects(self) -> int:
        """
        Berapa banyak object_id yang pernah muncul.
        """
        return len(self._table)
