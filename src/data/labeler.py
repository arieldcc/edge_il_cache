# src/data/labeler.py

from typing import Dict, List, Tuple
from .feature_table import FeatureTable


def build_slot_dataset_topk(
    feature_table: FeatureTable,
    top_ratio: float = 0.2,
    min_freq: int = 1,
) -> List[Dict]:
    """
    Membangun dataset D_t untuk satu slot, dengan label top-K% popularitas.

    PENTING:
      - Fungsi ini TIDAK memproses request lagi.
      - Asumsinya: selama slot t, loop utama sudah memanggil
        feature_table.update_and_get_gaps(...) untuk setiap request.
      - Di sini kita hanya:
          * baca freq per objek,
          * tentukan top-K%,
          * ambil last_gaps,
          * bentuk sample (x, y, freq),
          * reset freq untuk slot berikutnya.

    Output:
      - dataset: list of dict:
          {
              "object_id": str,
              "x": List[float],   # Gap1..GapL (last_gaps)
              "y": int,           # 1 jika top-K%, 0 jika tidak
              "freq": int,        # frekuensi di slot ini
          }
    """

    # 1) Kumpulkan (obj_id, freq) dari FeatureTable
    freq_list: List[Tuple[str, int]] = []
    for obj_id, f in feature_table.iter_freq_items():
        if f >= min_freq:
            freq_list.append((obj_id, f))

    if not freq_list:
        # tidak ada objek dengan freq >= min_freq di slot ini
        feature_table.reset_freqs()
        return []

    # 2) Tentukan K = top_ratio * N_obj
    n_obj = len(freq_list)
    k = int(n_obj * top_ratio)
    if k < 1:
        k = 1  # minimal 1 objek populer

    # 3) Urutkan objek berdasarkan frekuensi (descending)
    freq_list.sort(key=lambda x: x[1], reverse=True)

    # ambil K object_id populer
    top_ids = set(obj_id for obj_id, _ in freq_list[:k])

    # 4) Bentuk dataset D_t
    dataset: List[Dict] = []
    for obj_id, f in freq_list:
        gaps = feature_table.get_last_gaps(obj_id)
        if gaps is None:
            # secara teoritis tidak boleh terjadi kalau FeatureTable dipakai konsisten,
            # tapi jaga-jaga saja
            continue
        y = 1 if obj_id in top_ids else 0
        dataset.append(
            {
                "object_id": obj_id,
                "x": gaps,
                "y": y,
                "freq": f,
            }
        )

    # 5) Reset frekuensi untuk slot berikutnya (riwayat timestamp tetap)
    feature_table.reset_freqs()

    return dataset
