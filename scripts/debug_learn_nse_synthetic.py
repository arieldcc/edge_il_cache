import os
import sys
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.ml.learn_nse import LearnNSE


def make_batch(mean_pos, mean_neg, n_samples=200, noise=0.5):
    """
    Batch sintetis:
      - kelas 1 cluster di sekitar mean_pos (R^2),
      - kelas 0 cluster di sekitar mean_neg.
    """
    X_pos = mean_pos + noise * np.random.randn(n_samples // 2, 2)
    X_neg = mean_neg + noise * np.random.randn(n_samples // 2, 2)
    X = np.vstack([X_pos, X_neg])
    y = np.array([1] * (n_samples // 2) + [0] * (n_samples // 2), dtype=int)
    # shuffle
    idx = np.random.permutation(len(y))
    return X[idx], y[idx]


def batch_to_dataset(X, y):
    dataset = []
    for xi, yi in zip(X, y):
        dataset.append({"x": xi.tolist(), "y": int(yi), "freq": 1, "object_id": None})
    return dataset


if __name__ == "__main__":
    np.random.seed(42)

    # 2 fitur sederhana, bukan Gap tapi tujuannya tes algoritma Learn++.NSE
    il = LearnNSE(n_features=2, a=0.5, b=10.0, max_learners=20)

    # definisikan 3 environment / batch:
    #   t=1: kelas 1 di (2,2), kelas 0 di (-2,-2)
    #   t=2: geser sedikit
    #   t=3: tukar posisi (concept drift)
    envs = [
        {"mean_pos": np.array([2.0, 2.0]), "mean_neg": np.array([-2.0, -2.0])},
        {"mean_pos": np.array([3.0, 3.0]), "mean_neg": np.array([-3.0, -3.0])},
        {"mean_pos": np.array([-2.0, -2.0]), "mean_neg": np.array([2.0, 2.0])},
    ]

    for t, env in enumerate(envs, start=1):
        print(f"\n=== SLOT {t} ===")
        X, y = make_batch(env["mean_pos"], env["mean_neg"], n_samples=400, noise=0.5)
        dataset = batch_to_dataset(X, y)

        # eval ensemble sebelum update
        if il.num_learners() > 0:
            from src.ml.learn_nse import GaussianNaiveBayes  # hanya untuk type hint
            X_arr = X
            # prediksi batch sebelum update
            # gunakan API internal _predict_batch untuk diagnosa
            y_hat_before = il._predict_batch(X_arr)
            acc_before = (y_hat_before == y).mean()
            print(f"Acc sebelum update (dgn {il.num_learners()} learners): {acc_before:.4f}")
        else:
            print("Belum ada learner (warm start).")

        il.update_slot(dataset)

        print(f"Jumlah learner sesudah update: {il.num_learners()}")
        print(f"E_t (slot error ensemble): {il.last_E_t:.4f}")
        print("epsilon_k^t per learner:", [f"{e:.4f}" for e in il.last_epsilons])
        print("beta_bar_k^t per learner:", [f"{b:.4f}" for b in il.last_beta_bars])
        print("W_k^t per learner:", [f"{w:.4f}" for w in il.last_weights])

        # eval ensemble setelah update
        y_hat_after = il._predict_batch(X)
        acc_after = (y_hat_after == y).mean()
        print(f"Acc sesudah update: {acc_after:.4f}")
