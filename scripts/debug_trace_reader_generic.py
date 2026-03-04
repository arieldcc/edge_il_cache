# scripts/debug_trace_reader_generic.py

import os
import sys
import gzip
import tarfile
import io

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.trace_reader import TraceReader
from src.config.experiment_config import (
    WIKIPEDIA_SEPTEMBER_2007,
    WIKIPEDIA_OKTOBER_2007,
    WIKI2018,
)

# === PILIH DATASET UNTUK DEBUG DI SINI ===
DS = WIKI2018
# DS = WIKIPEDIA_SEPTEMBER_2007
# DS = WIKIPEDIA_OKTOBER_2007


def iter_manual_wikibench_gz(path, max_rows):
    """Parser manual untuk trace wikibench .gz lama."""
    with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if i >= max_rows:
                break
            yield line.strip()


def iter_manual_wiki2018_tr_tar(path, max_rows):
    """Parser manual untuk wiki2018.tr.tar.gz (format LRB)."""
    with gzip.open(path, "rb") as gz_f:
        with tarfile.open(fileobj=gz_f, mode="r:*") as tf:
            member = next(m for m in tf.getmembers() if m.name.endswith(".tr"))
            f_raw = tf.extractfile(member)
            if f_raw is None:
                raise RuntimeError("Tidak bisa extract *.tr dari tar")
            text_f = io.TextIOWrapper(f_raw, encoding="utf-8", errors="ignore")
            for i, line in enumerate(text_f):
                if i >= max_rows:
                    break
                yield line.strip()


if __name__ == "__main__":
    max_rows = 20
    reader = TraceReader(path=DS.path, max_rows=max_rows)

    print("=== COMPARE TraceReader vs PARSING MANUAL (10 baris pertama) ===")
    print(f"Dataset: {DS.name}")
    print(f"Path   : {DS.path}")
    print()

    # pilih parser manual berdasarkan ekstensi
    if DS.path.endswith(".tr.tar.gz"):
        manual_iter = iter_manual_wiki2018_tr_tar(DS.path, max_rows)
    elif DS.path.endswith(".gz"):
        manual_iter = iter_manual_wikibench_gz(DS.path, max_rows)
    else:
        raise ValueError(f"Path {DS.path} tidak dikenali untuk debug manual")

    for i, (req, line) in enumerate(zip(reader.iter_requests(), manual_iter)):
        print(f"\nLINE {i}: raw line = {line!r}")
        print("  TraceReader ->", req)
