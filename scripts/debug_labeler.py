import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.slot_iterator import iter_slots_from_trace
from src.data.feature_table import FeatureTable
from src.data.labeler import build_slot_dataset_topk


if __name__ == "__main__":
    # 1 slot (100k request) pertama, sesuai artikel
    slots = iter_slots_from_trace(
        parquet_dir="data/parquet/wikipedia_2007/",
        slot_size=100_000,
        max_rows=100_000,
    )

    slot0 = next(slots)  # slot pertama

    ft = FeatureTable(L=6, missing_gap_value=0.0)

    dataset = build_slot_dataset_topk(
        slot_requests=slot0,
        feature_table=ft,
        top_ratio=0.2,   # top-20%
    )

    print(f"Total samples in D_t: {len(dataset)}")

    # cek distribusi label
    pos = sum(1 for d in dataset if d["y"] == 1)
    neg = sum(1 for d in dataset if d["y"] == 0)
    print(f"Label 1 (popular): {pos}")
    print(f"Label 0 (non-pop): {neg}")

    # tampilkan beberapa contoh
    print("\nContoh 5 sampel populer (y=1):")
    count = 0
    for d in dataset:
        if d["y"] == 1:
            print(f"object_id={d['object_id']}")
            print(f"  freq={d['freq']}")
            print(f"  gaps={d['x']}")
            count += 1
            if count >= 5:
                break

    print("\nContoh 5 sampel tidak populer (y=0):")
    count = 0
    for d in dataset:
        if d["y"] == 0:
            print(f"object_id={d['object_id']}")
            print(f"  freq={d['freq']}")
            print(f"  gaps={d['x']}")
            count += 1
            if count >= 5:
                break