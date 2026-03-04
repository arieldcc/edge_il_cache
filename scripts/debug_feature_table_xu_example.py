# scripts/debug_feature_table_figure2.py

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


if __name__ == "__main__":
    # L=3 sesuai ilustrasi Gap1..Gap3 di Figure 2
    L = 3
    ft = FeatureTable(L=L)

    # Urutan request persis seperti Figure 2 Xu et al.
    # T1..T10 kita representasikan sebagai 1..10
    sequence = [
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

    print("Simulasi FeatureTable untuk urutan request Figure 2 (Xu et al.)")
    print(f"L = {L}")

    for idx, (ts, obj) in enumerate(sequence, start=1):
        gaps = ft.update_and_get_gaps(obj, ts)
        print(f"\nT{idx}: request object {obj}, timestamp={ts}")
        print(f"  gaps yang DIKEMBALIKAN untuk objek {obj}: {gaps}")
        # setelah memproses request ini, dump seluruh state feature table
        dump_state(ft, f"T{idx}")
