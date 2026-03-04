# src/data/slot_iterator.py

from typing import Dict, Generator, List, Optional
from .trace_reader import TraceReader


def iter_slots_from_trace(
    path: str = "data/raw/wikipedia_september_2007/wiki.1190153705.gz",
    slot_size: int = 100_000,
    max_rows: Optional[int] = None,
) -> Generator[List[Dict], None, None]:
    """
    Menghasilkan slot berisi 'slot_size' request dari trace.

    `path` bisa:
      - file .gz (dataset asli Wikipedia 2007, misalnya:
          data/raw/wikipedia_september_2007/wiki.1190153705.gz
          data/raw/wikipedia_oktober_2007/wiki.1191201596.gz
        )
      - atau direktori berisi file .parquet (jika suatu saat masih dipakai).

    Setiap elemen slot adalah list of dict:
      {
          "unique_id": str,
          "timestamp": float,
          "object_id": str,
          "is_update": str,
      }

    :param path: path ke file .gz ATAU direktori Parquet.
    :param slot_size: jumlah request per slot (Xu et al.: 100k).
    :param max_rows: batas total request yang dibaca dari trace.
    """
    # TraceReader sekarang menerima 'path', bukan 'parquet_dir'
    reader = TraceReader(path=path, max_rows=max_rows)

    batch: List[Dict] = []
    for req in reader.iter_requests():
        batch.append(req)

        if len(batch) >= slot_size:
            yield batch
            batch = []

    # sisa yang kurang dari slot_size (misal di ekor trace)
    if batch:
        yield batch
