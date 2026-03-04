# scripts/debug_labeler_2slots.py

import os
import sys

# Tambahkan root project ke sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.feature_table import FeatureTable
from src.data.labeler import build_slot_dataset_topk


def make_req(ts: float, obj_id: str, uid: int) -> dict:
    """
    Bikin satu request dict dengan format yang sama seperti TraceReader.
    unique_id dibuat sintetis dari counter.
    """
    return {
        "unique_id": str(uid),
        "timestamp": ts,
        "object_id": obj_id,
        "is_update": "0",
    }


if __name__ == "__main__":
    # L=3 sesuai ilustrasi Figure 2
    L = 3
    ft = FeatureTable(L=L)

    print("=== DEBUG LABELER + FEATURE_TABLE (2 SLOT) ===")
    print(f"L = {L}\n")

    # ------------------------------------------------
    # SLOT 1: urutan T1..T10 seperti debug_feature_table
    # ------------------------------------------------
    slot1_raw = [
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

    slot1 = []
    uid = 0
    for ts, obj in slot1_raw:
        uid += 1
        slot1.append(make_req(ts, obj, uid))

    print("========== SLOT 1 (STREAM REQUEST) ==========")
    for i, req in enumerate(slot1, start=1):
        gaps = ft.update_and_get_gaps(req["object_id"], req["timestamp"])
        print(f"[T{i:02d}] obj={req['object_id']}, ts={req['timestamp']}, gaps={gaps}")

    # akhir slot 1 → bentuk D_1 dari FeatureTable
    print("\n========== DATASET D_1 (labeler, top_ratio=0.5) ==========")
    D_1 = build_slot_dataset_topk(
        feature_table=ft,
        top_ratio=0.5,  # top-50% objek di slot 1
        min_freq=1,
    )

    for row in D_1:
        print(
            f"obj={row['object_id']}, freq={row['freq']}, y={row['y']}, x={row['x']}"
        )

    # ------------------------------------------------
    # SLOT 2: pola baru yang kamu minta
    #   slot2_objs = ["1", "2", "4", "3", "2", "4", "4", "3", "1", "4"]
    # timestamps lanjut 11..20
    # ------------------------------------------------
    slot2_objs = ["1", "2", "4", "3", "2", "4", "4", "3", "1", "4"]
    slot2_raw = [(10.0 + i, obj) for i, obj in enumerate(slot2_objs, start=1)]

    slot2 = []
    for ts, obj in slot2_raw:
        uid += 1
        slot2.append(make_req(ts, obj, uid))

    print("\n========== SLOT 2 (STREAM REQUEST) ==========")
    for i, req in enumerate(slot2, start=1):
        gaps = ft.update_and_get_gaps(req["object_id"], req["timestamp"])
        print(f"[T{10+i:02d}] obj={req['object_id']}, ts={req['timestamp']}, gaps={gaps}")

    # akhir slot 2 → bentuk D_2 dari FeatureTable
    print("\n========== DATASET D_2 (labeler, top_ratio=0.5) ==========")
    D_2 = build_slot_dataset_topk(
        feature_table=ft,
        top_ratio=0.5,  # top-50% objek di slot 2
        min_freq=1,
    )

    for row in D_2:
        print(
            f"obj={row['object_id']}, freq={row['freq']}, y={row['y']}, x={row['x']}"
        )
