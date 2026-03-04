# scripts/debug_slot_iterator.py

import os
import sys

# Tambahkan root project ke sys.path
# /.../edge_il_cache/
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.slot_iterator import iter_slots_from_trace


if __name__ == "__main__":
    # Pakai dataset asli .gz
    path = "data/raw/wikipedia_september_2007/wiki.1190153705.gz"
    # Atau ganti:
    # path = "data/raw/wikipedia_oktober_2007/wiki.1191201596.gz"

    slots = iter_slots_from_trace(
        path=path,
        slot_size=100_000,
        max_rows=300_000,   # misal 3 slot pertama
    )

    for i, slot in enumerate(slots):
        print(f"Slot {i}, size={len(slot)}")
        print("  First req:", slot[0])
        print("  Last  req:", slot[-1])
