# scripts/debug_trace_reader.py

import os
import sys

# Tentukan root project (misal: /.../edge_il_cache/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.trace_reader import TraceReader


def debug_parquet():
    """
    Debug untuk mode PARQUET:
    Membaca beberapa request pertama dari direktori Parquet.
    """
    path = "data/parquet/wikipedia_2007/"  # sesuaikan jika beda

    reader = TraceReader(
        path=path,
        max_rows=5,  # baca 5 baris pertama
    )

    print(f"== Debug PARQUET: {path} ==")
    for req in reader.iter_requests():
        print(req)


def debug_raw_gz():
    """
    Debug untuk mode RAW GZ:
    Membaca beberapa request pertama dari file .gz Wikipedia.
    """
    # Contoh: salah satu dari dua dataset baru
    # Sesuaikan path ini dengan struktur di mesinmu:
    #   data/raw/wikipedia_september_2007/wiki.1190153705.gz
    #   data/raw/wikipedia_oktober_2007/wiki.1191201596.gz
    path = "data/raw/wikipedia_oktober_2007/wiki.1191201596.gz"

    reader = TraceReader(
        path=path,
        max_rows=5,  # baca 5 baris pertama
    )

    print(f"== Debug RAW GZ: {path} ==")
    for req in reader.iter_requests():
        print(req)


if __name__ == "__main__":
    # Pilih mode debug dengan argumen:
    #   python scripts/debug_trace_reader.py parquet
    #   python scripts/debug_trace_reader.py gz
    mode = sys.argv[1] if len(sys.argv) > 1 else "parquet"

    if mode == "parquet":
        debug_parquet()
    elif mode in ("gz", "raw_gz"):
        debug_raw_gz()
    else:
        print(f"Mode tidak dikenal: {mode!r}")
        print("Gunakan: parquet | gz")
