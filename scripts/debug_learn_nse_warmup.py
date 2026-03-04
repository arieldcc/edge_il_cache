# scripts/debug_learn_nse.py

import os
import sys
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config.experiment_config import WIKIPEDIA_2007, ILConfig
from src.data.slot_iterator import iter_slots_from_trace
from src.data.feature_table import FeatureTable
from src.data.labeler import build_slot_dataset_topk
from src.ml.learn_nse import LearnNSE


if __name__ == "__main__":
    il_cfg = ILConfig()

    dataset_dir = "data/parquet/wikipedia_2007/"  # sesuaikan dengan struktur Anda

    feature_table = FeatureTable(
        L=il_cfg.num_gaps,
        missing_gap_value=0.0,
    )

    il_model = LearnNSE(
        n_features=il_cfg.num_gaps,
        a=il_cfg.sigmoid_a,
        b=il_cfg.sigmoid_b,
        max_learners=il_cfg.max_classifiers,
    )

    # ambil 10 slot pertama (1M request) sebagai warm-up seperti di artikel
    max_rows = WIKIPEDIA_2007.num_warmup_requests
    slot_size = WIKIPEDIA_2007.slot_size

    slots = iter_slots_from_trace(
        parquet_dir=dataset_dir,
        slot_size=slot_size,
        max_rows=max_rows,
    )

    for t, slot_reqs in enumerate(slots, start=1):
        print(f"\n=== WARM-UP SLOT {t} ===")
        print(f"Jumlah request di slot: {len(slot_reqs)}")

        # bangun D_t (Gap1..GapL + label top-20%)
        D_t = build_slot_dataset_topk(
            slot_requests=slot_reqs,
            feature_table=feature_table,
            top_ratio=il_cfg.pop_top_percent,
        )

        print(f"Jumlah sample objek unik (|D_t|): {len(D_t)}")
        if not D_t:
            print("[WARN] Slot kosong, lanjut.")
            continue

        # sedikit statistik label
        num_pos = sum(1 for d in D_t if d["y"] == 1)
        num_neg = len(D_t) - num_pos
        print(f"Label 1 (popular): {num_pos}, Label 0: {num_neg}")

        # evaluasi ensemble sebelum update (kalau sudah ada learner)
        if il_model.num_learners() > 0:
            X_before = [d["x"] for d in D_t]
            y_before = [d["y"] for d in D_t]
            import numpy as np
            X_arr = np.array(X_before, dtype=float)
            y_arr = np.array(y_before, dtype=int)
            y_hat_before = il_model._predict_batch(X_arr)
            acc_before = (y_hat_before == y_arr).mean()
            print(f"Akurasi ensemble sebelum update: {acc_before:.4f}")
        else:
            print("Belum ada learner (slot pertama).")

        # update Learn++.NSE dengan D_t
        il_model.update_slot(D_t)

        print(f"Jumlah learner sesudah update: {il_model.num_learners()}")
        print(f"E_t (slot error ensemble): {il_model.last_E_t:.4f}")
        print("epsilon_k^t (per learner):")
        print(" ", [f"{e:.4f}" for e in il_model.last_epsilons])
        print("beta_bar_k^t (per learner):")
        print(" ", [f"{b:.4f}" for b in il_model.last_beta_bars])
        print("W_k^t (per learner):")
        print(" ", [f"{w:.4f}" for w in il_model.last_weights])

        # evaluasi ensemble setelah update
        X_after = [d["x"] for d in D_t]
        y_after = [d["y"] for d in D_t]
        X_arr2 = np.array(X_after, dtype=float)
        y_arr2 = np.array(y_after, dtype=int)
        y_hat_after = il_model._predict_batch(X_arr2)
        acc_after = (y_hat_after == y_arr2).mean()
        print(f"Akurasi ensemble sesudah update: {acc_after:.4f}")

        if t >= 10:
            # sesuai paper: warm-up 1M (10 slot × 100k)
            break
