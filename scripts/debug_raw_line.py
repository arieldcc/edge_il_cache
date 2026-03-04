# scripts/debug_raw_line.py

import gzip

path = "data/raw/wikipedia_september_2007/wiki.1190153705.gz"

with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as f:
    for i, line in enumerate(f):
        if i >= 10:
            break
        line = line.strip()
        print(f"LINE {i}: {line!r}")
        parts = line.split()
        print("  split:", parts)
