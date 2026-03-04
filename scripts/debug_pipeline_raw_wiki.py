# scripts/debug_pipeline_raw_wiki.py

import os
import sys

# Tambahkan root project ke sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config.experiment_config import (
    WIKIPEDIA_SEPTEMBER_2007,
    ILConfig,
)
from src.data.slot_iterator import iter_slots_from_trace
from src.data.feature_table import FeatureTable
from src.data.labeler import build_slot_dataset_topk


def main():
    ds_cfg = WIKIPEDIA_SEPTEMBER_2007
    il_cfg = ILConfig()

    # untuk DEBUG: batasi jumlah slot
    max_debug_slots = 10
    max_rows = ds_cfg.slot_size * max_debug_slots

    print("=== DEBUG PIPELINE: raw Wikipedia September 2007 ===")
    print(f"Dataset path   : {ds_cfg.path}")
    print(f"Slot size      : {ds_cfg.slot_size}")
    print(f"IL num_gaps    : {il_cfg.num_gaps}")
    print(f"pop_top_percent: {il_cfg.pop_top_percent}")
    print(f"max_rows debug : {max_rows}")
    print("NOTE: timestamp TIDAK pakai ts asli trace, "
          "tapi pakai nomor urut request global (1,2,3,...)")
    print()

    # Inisialisasi FeatureTable dengan L = num_gaps (Xu: 6)
    ft = FeatureTable(L=il_cfg.num_gaps)

    # counter global untuk request, dipakai sebagai "waktu" sintetis
    global_req_idx = 0

    # Iterasi slot dari trace asli (.gz)
    for slot_idx, slot in enumerate(
        iter_slots_from_trace(
            path=ds_cfg.path,
            slot_size=ds_cfg.slot_size,
            max_rows=max_rows,
        ),
        start=1,
    ):
        print(f"\n================ SLOT {slot_idx} ================")
        print(f"Jumlah request di slot ini: {len(slot)}")

        # 1) PROSES REQUEST DI SLOT: update FeatureTable
        for req in slot:
            obj_id = req["object_id"]

            # Abaikan timestamp asli; pakai nomor urut global sebagai timestamp
            global_req_idx += 1
            ts_seq = float(global_req_idx)

            # update gap & freq dengan ts_seq
            ft.update_and_get_gaps(obj_id, ts_seq)

        # 2) AKHIR SLOT: bangun D_t dengan label top-K% popularitas
        print(
            f"Bangun dataset D_{slot_idx} dengan top_ratio={il_cfg.pop_top_percent}"
        )
        D_t = build_slot_dataset_topk(
            feature_table=ft,
            top_ratio=il_cfg.pop_top_percent,
            min_freq=1,
        )

        # Ringkasan D_t
        num_samples = len(D_t)
        num_popular = sum(1 for row in D_t if row["y"] == 1)
        num_non_popular = num_samples - num_popular
        print(f"  #obj (sample)  : {num_samples}")
        print(f"    #popular (y=1): {num_popular}")
        print(f"    #non-pop (y=0): {num_non_popular}")

        # Tampilkan beberapa contoh baris (misalnya 5 pertama)
        print("\n  Contoh 5 baris pertama D_t:")
        for row in D_t[:5]:
            obj_id = row["object_id"]
            freq = row["freq"]
            y = row["y"]
            gaps = row["x"]
            print(
                f"    obj={obj_id!r}, freq={freq}, y={y}, "
                f"len_gaps={len(gaps)}, gaps={gaps}"
            )

        # batasi hanya beberapa slot untuk debug
        if slot_idx >= max_debug_slots:
            print("\n[DEBUG] Mencapai batas slot debug, berhenti di sini.")
            break


if __name__ == "__main__":
    main()
