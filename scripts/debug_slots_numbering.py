# scripts/debug_slots_numbering.py
import os, sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config.experiment_config import WIKIPEDIA_SEPTEMBER_2007 as DS
from src.data.slot_iterator import iter_slots_from_trace

if __name__ == "__main__":
    global_idx = 0
    for slot_idx, slot in enumerate(
        iter_slots_from_trace(
            path=DS.path,
            slot_size=10,   # kecil dulu untuk debug
            max_rows=50,
        ),
        start=1
    ):
        print(f"\nSLOT {slot_idx}, len={len(slot)}")
        for req in slot:
            global_idx += 1
            print(f"  global#{global_idx}: ts={req['timestamp']}, obj={req['object_id'][:40]}...")
