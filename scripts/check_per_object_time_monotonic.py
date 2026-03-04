# scripts/check_per_object_time_monotonic.py
import os, sys, collections
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config.experiment_config import WIKIPEDIA_SEPTEMBER_2007 as DS
from src.data.trace_reader import TraceReader

if __name__ == "__main__":
    reader = TraceReader(path=DS.path, max_rows=1_000_000)
    last_ts = {}
    bad_count = 0

    for idx, req in enumerate(reader.iter_requests(), start=1):
        oid = req["object_id"]
        ts  = req["timestamp"]
        if oid in last_ts and ts < last_ts[oid]:
            bad_count += 1
            if bad_count <= 20:
                print(f"[WARN] ts mundur untuk obj={oid!r}: {ts} < {last_ts[oid]} (req#{idx})")
        last_ts[oid] = ts

    print(f"\nTotal pelanggaran monotonicitas per-obj: {bad_count}")
