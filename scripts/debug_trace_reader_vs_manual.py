# scripts/debug_trace_reader_vs_manual.py

import os, sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.trace_reader import TraceReader
from src.config.experiment_config import WIKI2018 as DS

if __name__ == "__main__":
    reader = TraceReader(path=DS.path, max_rows=10)

    print("=== COMPARE TraceReader vs parsing manual (10 baris pertama) ===")
    with open(__file__, "r"):  # hanya untuk pastikan script jalan, tidak dipakai
        pass

    # baca lagi langsung dari gzip untuk dibandingkan
    import gzip
    with gzip.open(DS.path, "rt", encoding="utf-8", errors="ignore") as f_raw:
        raw_lines = [next(f_raw).strip() for _ in range(10)]

    for i, (req, line) in enumerate(zip(reader.iter_requests(), raw_lines)):
        parts = line.split()
        print(f"\nLINE {i}: {line!r}")
        print("  manual parts:", parts)
        print("  TraceReader :", req)
