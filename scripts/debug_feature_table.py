import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.trace_reader import TraceReader
from src.data.feature_table import FeatureTable


if __name__ == "__main__":
    reader = TraceReader(
        parquet_dir="data/parquet/wikipedia_2007/",
        max_rows=20_000,  # misal 20k request pertama
    )
    ft = FeatureTable(L=6, missing_gap_value=0.0)

    # proses 1 slot kecil (misalnya 1000 request dulu, untuk uji awal)
    count = 0
    for req in reader.iter_requests():
        gaps = ft.update_and_get_gaps(req["object_id"], req["timestamp"])
        if count < 5:
            print(f"REQ {count}: object={req['object_id']}")
            print("  timestamp:", req["timestamp"])
            print("  gaps    :", gaps)
        count += 1
        if count >= 1000:
            break

    print("\nTotal objek yang terlihat:", ft.num_objects())
    # contoh cek freq dan last_gaps untuk 5 object pertama
    print("\nContoh freq & last_gaps 5 objek pertama:")
    i = 0
    for obj_id, freq in ft.iter_freq_items():
        print(f"  object_id={obj_id}")
        print(f"    freq     = {freq}")
        print(f"    last_gaps= {ft.get_last_gaps(obj_id)}")
        i += 1
        if i >= 5:
            break
