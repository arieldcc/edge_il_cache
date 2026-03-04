# scripts/debug_feature_table_2slots.py

import os
import sys

# Tambahkan root project ke sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.feature_table import FeatureTable, ObjectStats


def dump_state(ft: FeatureTable, label: str) -> None:
    print(f"\n===== STATE SETELAH {label} =====")
    # tampilkan semua object_id yang pernah muncul, urutkan biar stabil
    for obj_id in sorted(ft._table.keys(), key=str):
        stats: ObjectStats = ft._table[obj_id]
        ts_list = list(stats.timestamps)
        gaps = stats.last_gaps
        print(f"Objek {obj_id}:")
        print(f"  timestamps (newest→oldest): {ts_list}")
        print(f"  freq (slot saat ini)      : {stats.freq}")
        print(f"  last_gaps (Gap1..L)       : {gaps}")


def build_training_for_slot(ft: FeatureTable, pop_top_percent: float = 0.5):
    """
    Bentuk D_t untuk satu slot:
      - hanya objek dengan freq > 0 di slot ini,
      - ranking berdasarkan freq desc,
      - top-% jadi label 1, lainnya 0,
      - fitur = last_gaps (Gap1..L).
    """
    # kumpulkan (obj_id, freq) hanya yang freq>0
    objs = [(obj_id, freq) for obj_id, freq in ft.iter_freq_items() if freq > 0]

    # sort desc berdasarkan freq
    objs.sort(key=lambda x: x[1], reverse=True)

    N_t = len(objs)
    if N_t == 0:
        print("  [WARNING] Slot kosong, tidak ada objek dengan freq>0")
        return []

    K = int(pop_top_percent * N_t)
    if K < 1:
        K = 1  # minimal 1 objek populer

    top_set = {obj_id for obj_id, _ in objs[:K]}

    D_t = []
    print(f"\n== DATA TRAINING SLOT (N_t={N_t}, top_percent={pop_top_percent}, K={K}) ==")
    for obj_id, freq in objs:
        gaps = ft.get_last_gaps(obj_id)  # list panjang L (sesuai FeatureTable baru)
        y = 1 if obj_id in top_set else 0
        print(f"  obj={obj_id}, freq={freq}, label={y}, gaps={gaps}")
        D_t.append((gaps, y))

    return D_t


if __name__ == "__main__":
    # L=3 sesuai ilustrasi Gap1..Gap3 di Figure 2
    L = 3
    ft = FeatureTable(L=L)  # pastikan versi FeatureTable yang SELALU hitung gap + pad

    print("Simulasi FeatureTable untuk 2 slot (Figure 2 + slot lanjutan)")
    print(f"L = {L}")

    # ----------------------
    # SLOT 1: Figure 2 Xu
    # ----------------------
    slot1 = [
        (1.0, "1"),  # T1: object 1
        (2.0, "2"),  # T2: object 2
        (3.0, "3"),  # T3: object 3
        (4.0, "4"),  # T4: object 4
        (5.0, "1"),  # T5: object 1
        (6.0, "2"),  # T6: object 2
        (7.0, "3"),  # T7: object 3
        (8.0, "1"),  # T8: object 1
        (9.0, "2"),  # T9: object 2
        (10.0, "1"), # T10: object 1
    ]

    print("\n================ SLOT 1 ================")
    for idx, (ts, obj) in enumerate(slot1, start=1):
        gaps = ft.update_and_get_gaps(obj, ts)
        print(f"\n[T{idx:02d}] request object {obj}, timestamp={ts}")
        print(f"  gaps yang DIKEMBALIKAN untuk objek {obj}: {gaps}")
        dump_state(ft, f"T{idx:02d}")

    # Bentuk D_1 (top 50%) dan TAMPILKAN
    D_1 = build_training_for_slot(ft, pop_top_percent=0.5)

    # Reset freq untuk pindah ke slot 2, tapi timestamps & last_gaps tetap
    ft.reset_freqs()

    # ----------------------
    # SLOT 2: urutan baru
    # data slot 2: 1, 2, 1, 3, 2, 1, 4, 3, 1, 4
    # timestamps lanjut 11..20
    # ----------------------
    slot2_objs = ["1", "2", "4", "3", "2", "4", "4", "3", "1", "4"]
    slot2 = [(10.0 + i, obj) for i, obj in enumerate(slot2_objs, start=1)]
    # → (11,1), (12,2), ..., (20,4)

    print("\n================ SLOT 2 ================")
    for idx2, (ts, obj) in enumerate(slot2, start=1):
        gaps = ft.update_and_get_gaps(obj, ts)
        print(f"\n[T{10+idx2:02d}] request object {obj}, timestamp={ts}")
        print(f"  gaps yang DIKEMBALIKAN untuk objek {obj}: {gaps}")
        dump_state(ft, f"T{10+idx2:02d}")

    # Bentuk D_2 (top 50%) dan TAMPILKAN
    D_2 = build_training_for_slot(ft, pop_top_percent=0.5)
