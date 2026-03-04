# scripts/debug_one_object_gaps.py

import os
import sys

# Tambahkan root project ke sys.path: /.../edge_il_cache/
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.trace_reader import TraceReader
from src.data.feature_table import FeatureTable

# Pilih satu objek populer yang tadi sering muncul di debug pipeline
OBJ_TARGET = (
    "http://en.wikipedia.org/w/index.php?"
    "title=MediaWiki:Monobook.css&usemsgcache=yes&action=raw&"
    "ctype=text/css&smaxage=2678400"
)

DATA_PATH = "data/raw/wikipedia_september_2007/wiki.1190153705.gz"


def main():
    # Baca sebagian trace (misal 200k request pertama)
    reader = TraceReader(DATA_PATH, max_rows=200_000)
    ft = FeatureTable(L=6)

    seen_target = 0

    for i, req in enumerate(reader.iter_requests(), start=1):
        obj_id = req["object_id"]
        ts = req["timestamp"]

        # Ambil riwayat SEBELUM update (untuk objek target saja)
        stats = ft._table.get(OBJ_TARGET)
        timestamps_before = list(stats.timestamps) if stats else []

        # Update feature table dengan request ini
        gaps = ft.update_and_get_gaps(obj_id, ts)

        # Kalau request ini untuk objek target → print detail
        if obj_id == OBJ_TARGET:
            seen_target += 1
            print(f"\n=== HIT #{seen_target} untuk OBJ TARGET pada request global #{i} ===")
            print(f"  timestamp_now   = {ts}")
            print(f"  timestamps_prev = {timestamps_before}")
            print(f"  gaps            = {gaps}")

            if gaps and any(g < 0 for g in gaps):
                print("  >>> TERDETEKSI GAP NEGATIF <<<")
                break

        # Batas aman supaya skrip debug tidak jalan terlalu lama
        if i >= 200_000:
            break


if __name__ == "__main__":
    main()
