# src/ml/learn_nse.py

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Any

import numpy as np


# ---------------------------------------------------------------------------
#  Naive Bayes sederhana untuk fitur numerik (Gaussian NB)
#  Dipakai sebagai base classifier h_k di Learn++.NSE
# ---------------------------------------------------------------------------

class GaussianNaiveBayes:
    """
    Gaussian Naive Bayes sederhana, dilatih sekali per slot dengan sample_weight.

    - Fitur: vektor numerik (Gap1..GapL)
    - Kelas: {0, 1}

    Ini *bukan* incremental antar-slot; setiap base classifier
    dilatih hanya dengan dataset slot D_t tertentu.
    """

    def __init__(self) -> None:
        self.classes_: Optional[np.ndarray] = None
        self.class_prior_: Optional[np.ndarray] = None
        self.theta_: Optional[np.ndarray] = None
        self.sigma_: Optional[np.ndarray] = None
        self.eps_: float = 1e-9

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> "GaussianNaiveBayes":
        """
        X: shape (n_samples, n_features)
        y: shape (n_samples,), nilai 0/1
        sample_weight: shape (n_samples,) atau None
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)

        if sample_weight is None:
            sample_weight = np.ones(X.shape[0], dtype=float)
        else:
            sample_weight = np.asarray(sample_weight, dtype=float)

        classes = np.array([0, 1], dtype=int)
        self.classes_ = classes
        n_classes = len(classes)
        n_features = X.shape[1]

        theta = np.zeros((n_classes, n_features), dtype=float)
        sigma = np.zeros((n_classes, n_features), dtype=float)
        class_weight_sum = np.zeros(n_classes, dtype=float)

        # hitung prior dan statistik Gaussian per kelas
        for idx, c in enumerate(classes):
            mask = (y == c)
            w = sample_weight[mask]
            if w.size == 0 or w.sum() == 0:
                # tidak ada sampel kelas ini; prior nol, var dummy
                class_weight_sum[idx] = 0.0
                theta[idx, :] = 0.0
                sigma[idx, :] = 1.0
                continue

            X_c = X[mask]
            w_sum = w.sum()
            class_weight_sum[idx] = w_sum

            # mean tertimbang
            mean = np.average(X_c, axis=0, weights=w)
            theta[idx, :] = mean

            # var tertimbang
            diff = X_c - mean
            var = np.average(diff**2, axis=0, weights=w)
            sigma[idx, :] = np.maximum(var, self.eps_)

        # prior kelas
        total_weight = class_weight_sum.sum()
        if total_weight == 0:
            # fallback: prior uniform
            self.class_prior_ = np.array([0.5, 0.5])
        else:
            self.class_prior_ = class_weight_sum / total_weight

        self.theta_ = theta
        self.sigma_ = sigma

        return self

    def _joint_log_likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Menghitung log P(X, y=c) untuk setiap c.
        X: shape (n_samples, n_features)
        return: shape (n_samples, n_classes)
        """
        if self.classes_ is None or self.theta_ is None or self.sigma_ is None:
            raise RuntimeError("Model belum dilatih.")

        X = np.asarray(X, dtype=float)
        n_samples, n_features = X.shape
        n_classes = len(self.classes_)

        jll = np.zeros((n_samples, n_classes), dtype=float)

        for idx in range(n_classes):
            # log prior
            log_prior = math.log(self.class_prior_[idx] + self.eps_)

            # Gaussian log-density per fitur
            mean = self.theta_[idx]
            var = self.sigma_[idx]
            # -0.5 * sum_j [ log(2pi*var_j) + (x_j-mean_j)^2 / var_j ]
            log_det = 0.5 * np.sum(np.log(2.0 * math.pi * var))
            inv_var = 1.0 / (2.0 * var)
            diff = X - mean
            quad = np.sum(diff * diff * inv_var, axis=1)
            jll[:, idx] = log_prior - log_det - quad

        return jll

    def predict(self, X: np.ndarray) -> np.ndarray:
        jll = self._joint_log_likelihood(X)
        idx_max = np.argmax(jll, axis=1)
        return self.classes_[idx_max]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Estimasi probabilitas kelas berdasarkan log-likelihood ter-normalisasi.
        return: shape (n_samples, n_classes)
        """
        jll = self._joint_log_likelihood(X)
        # logsumexp stabil untuk normalisasi probabilitas
        jll_max = np.max(jll, axis=1, keepdims=True)
        exp_jll = np.exp(jll - jll_max)
        denom = np.sum(exp_jll, axis=1, keepdims=True)
        denom = np.maximum(denom, self.eps_)
        return exp_jll / denom

    def predict_single(self, x: Sequence[float]) -> int:
        X = np.asarray(x, dtype=float).reshape(1, -1)
        return int(self.predict(X)[0])


class GaussianNaiveBayesMissingAware(GaussianNaiveBayes):
    """
    NB yang mengabaikan fitur missing (NaN) saat fit dan predict.
    Dipakai hanya untuk model optimasi. Baseline Xu tetap pakai GaussianNaiveBayes.
    """

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)

        if sample_weight is None:
            sample_weight = np.ones(X.shape[0], dtype=float)
        else:
            sample_weight = np.asarray(sample_weight, dtype=float)

        classes = np.array([0, 1], dtype=int)
        self.classes_ = classes
        n_classes = len(classes)
        n_features = X.shape[1]

        # fitur yang pernah terlihat (non-NaN) setidaknya sekali di training
        self.feature_seen_ = np.any(~np.isnan(X), axis=0)

        theta = np.zeros((n_classes, n_features), dtype=float)
        sigma = np.zeros((n_classes, n_features), dtype=float)
        class_wsum = np.zeros(n_classes, dtype=float)

        for ci, c in enumerate(classes):
            mask_c = (y == c)
            Xc = X[mask_c]
            wc = sample_weight[mask_c]
            wsum_c = float(wc.sum())
            class_wsum[ci] = wsum_c

            if Xc.shape[0] == 0 or wsum_c <= 0:
                theta[ci, :] = 0.0
                sigma[ci, :] = 1.0
                continue

            # NaN-aware mean/var per fitur (n_features kecil -> loop ini murah)
            for j in range(n_features):
                # kalau fitur tidak pernah terlihat sama sekali, biarkan dummy (tidak akan dipakai di jll)
                if not self.feature_seen_[j]:
                    theta[ci, j] = 0.0
                    sigma[ci, j] = 1.0
                    continue

                obs = ~np.isnan(Xc[:, j])
                if not np.any(obs):
                    theta[ci, j] = 0.0
                    sigma[ci, j] = 1.0
                    continue

                xj = Xc[obs, j]
                wj = wc[obs]
                wsum = float(wj.sum())
                if wsum <= 0:
                    theta[ci, j] = 0.0
                    sigma[ci, j] = 1.0
                    continue

                mean = float(np.average(xj, weights=wj))
                var = float(np.average((xj - mean) ** 2, weights=wj))
                theta[ci, j] = mean
                sigma[ci, j] = max(var, self.eps_)

        tot = float(class_wsum.sum())
        self.class_prior_ = (class_wsum / tot) if tot > 0 else np.array([0.5, 0.5], dtype=float)
        self.theta_ = theta
        self.sigma_ = sigma
        return self

    def _joint_log_likelihood(self, X: np.ndarray) -> np.ndarray:
        if self.classes_ is None or self.theta_ is None or self.sigma_ is None:
            raise RuntimeError("Model belum dilatih.")

        X = np.asarray(X, dtype=float)
        n_samples, n_features = X.shape
        n_classes = len(self.classes_)

        # observed hanya jika bukan NaN dan fitur tersebut pernah terlihat saat fit
        obs = ~np.isnan(X)
        if hasattr(self, "feature_seen_"):
            obs = obs & self.feature_seen_[None, :]

        obs_f = obs.astype(float)

        # isi NaN dengan 0 agar operasi aman; kontribusi tetap dimask oleh obs_f
        X0 = np.where(obs, X, 0.0)

        jll = np.zeros((n_samples, n_classes), dtype=float)
        for idx in range(n_classes):
            log_prior = math.log(float(self.class_prior_[idx]) + self.eps_)
            mean = self.theta_[idx]
            var = self.sigma_[idx]

            log_var_term = np.log(2.0 * math.pi * var)
            inv_var = 1.0 / (2.0 * var)

            # hanya fitur observed yang dihitung
            log_det = 0.5 * (obs_f @ log_var_term)
            diff = X0 - mean
            quad = (obs_f * (diff * diff)) @ inv_var

            jll[:, idx] = log_prior - log_det - quad

        return jll

# ---------------------------------------------------------------------------
#  Learn++.NSE
# ---------------------------------------------------------------------------

@dataclass
class BaseLearnerInfo:
    learner_id: int
    model: GaussianNaiveBayes
    created_at_slot: int
    # history of beta_k^tau per slot tau
    beta_history: Dict[int, float] = field(default_factory=dict)


class LearnNSE:
    """
    Implementasi Learn++.NSE untuk binary classification (kelas 0/1),
    mengikuti struktur di artikel IL-based edge caching:

      - Setiap slot t punya dataset D_t = {(x_i^t, y_i^t)}
      - Di tiap slot:
         * update bobot instance berdasarkan kinerja ensemble (E_t)
         * latih base classifier baru h_t dengan bobot instance
         * hitung error tiap h_k di slot t → epsilon_k^t
         * konversi ke beta_k^t = epsilon / (1 - epsilon)
         * gabungkan sejarah beta_k^tau (tau <= t) dengan fungsi recency-weighted
           untuk dapat beta_bar_k^t
         * bobot voting W_k^t = log(1 / beta_bar_k^t)
      - Maksimal jumlah base classifier dibatasi (default 20) → pruning yang tertua.

    Catatan:
      - Implementasi ini memakai GaussianNB sederhana sebagai base learner.
      - Fungsi recency memakai sigmoid dengan parameter a dan b.
    """

    def __init__(
        self,
        n_features: int,
        a: float = 0.5,
        b: float = 10.0,
        max_learners: int = 20,
        base_learner_factory=None,
    ) -> None:
        """
        :param n_features: jumlah fitur (L = 6 di eksperimen utama).
        :param a: parameter slope sigmoid untuk weighting recency.
        :param b: parameter shift sigmoid.
        :param max_learners: jumlah maksimum base classifiers yang dipertahankan.
        """
        self.n_features = n_features
        self.a = a
        self.b = b
        self.max_learners = max_learners

        if base_learner_factory is None:
            base_learner_factory = GaussianNaiveBayes
        self._base_learner_factory = base_learner_factory

        self._next_learner_id: int = 1

        self.learners: List[BaseLearnerInfo] = []
        self.current_slot: int = 0  # t
        # W_k^t terbaru (update setiap slot)
        self._weights: List[float] = []

        self.last_E_t: Optional[float] = None
        self.last_epsilons: Optional[List[float]] = None
        self.last_beta_bars: Optional[List[float]] = None
        self.last_weights: Optional[List[float]] = None

    # ----------------- API publik -----------------

    def num_learners(self) -> int:
        return len(self.learners)

    def predict(self, x: Sequence[float]) -> int:
        """
        Prediksi kelas (0/1) untuk satu sampel x menggunakan ensemble saat ini.
        Jika belum ada learner, default 1 (agresif admit) atau 0 (konservatif).
        Di sini kita pilih 1 agar IL benar-benar mempelajari kapan *tidak* cache.
        """
        if not self.learners:
            return 1

        x_arr = np.asarray(x, dtype=float).reshape(1, -1)
        if x_arr.shape[1] != self.n_features:
            raise ValueError(f"n_features mismatch: expected {self.n_features}, got {x_arr.shape[1]}")

        scores = {0: 0.0, 1: 0.0}
        for info, w in zip(self.learners, self._weights):
            y_hat = info.model.predict(x_arr)[0]
            scores[int(y_hat)] += w

        # pilih kelas dengan skor tertinggi
        return 1 if scores[1] >= scores[0] else 0

    def score_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Skor kontinu berbasis probabilitas kelas-1 (weighted average).
        return: shape (n_samples,)
        """
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("score_batch expects 2D array shape (n_samples, n_features).")

        n_samples = X.shape[0]
        if not self.learners:
            return np.full(n_samples, 0.5, dtype=float)

        total_w = float(np.sum(self._weights)) if self._weights else 0.0
        if total_w <= 0.0:
            return np.full(n_samples, 0.5, dtype=float)

        scores = np.zeros(n_samples, dtype=float)
        for info, w in zip(self.learners, self._weights):
            model = info.model
            if hasattr(model, "predict_proba"):
                p1 = model.predict_proba(X)[:, 1]
            else:
                y_hat = model.predict(X)
                p1 = (y_hat == 1).astype(float)
            scores += float(w) * p1

        scores = scores / total_w
        return np.clip(scores, 0.0, 1.0)

    def score_one(self, x: Sequence[float]) -> float:
        """
        Skor kontinu untuk satu sampel.
        """
        X = np.asarray(x, dtype=float).reshape(1, -1)
        return float(self.score_batch(X)[0])

    def update_slot(self, dataset: List[Dict], return_info: bool = False) -> Optional[Dict[str, Any]]:
        """
        Implementasi slot-level dari Learn++.NSE mengikuti Xu + Elwell:

        1. Hitung error ensemble E_t pada D_t.
        2. Bentuk bobot instance w_i^t dan distribusi F_i^t.
        3. Latih base classifier baru h_t dengan distribusi F_i^t.
        4. Hitung epsilon_k^t untuk semua learner (weighted error).
        5. Clamp semua epsilon_k^t ke (0, 0.5].
        6. Hitung beta_k^t, beta_bar_k^t memakai sigmoid recency.
        7. Hitung W_k^t = log(1 / beta_bar_k^t).
        8. Jika jumlah learner > max_learners, prune learner tertua.
        """

        if not dataset:
            # slot kosong; tidak ada update apa-apa selain majuin time index
            self.current_slot += 1
            if return_info:
                return {
                    "slot_index": self.current_slot,
                    "num_samples": 0,
                    "num_label_1": 0,
                    "num_label_0": 0,
                    "num_learners_before": self.num_learners(),
                    "num_learners_after": self.num_learners(),
                    "E_t": None,
                    "acc_before": None,
                    "acc_after": None,
                    "epsilons": [],
                    "beta_bars": [],
                    "weights": [],
                }
            return None

        self.current_slot += 1
        t = self.current_slot

        # 1) Siapkan X, y
        X, y = self._prepare_xy(dataset)
        n_samples = X.shape[0]
        num_label_1 = int((y == 1).sum())
        num_label_0 = n_samples - num_label_1
        num_learners_before = self.num_learners()

        # 2) Hitung E_t (error ensemble di slot t) dan bobot instance F_i^t
        if self.learners:
            ensemble_pred = self._predict_batch(X)
            misclassified = (ensemble_pred != y).astype(float)
            E_t = float(misclassified.mean())
            # hindari E_t = 0 atau 1
            E_t = min(max(E_t, 1e-6), 1.0 - 1e-6)
            acc_before = 1.0 - E_t

            # Step 2 Elwell: sampel yg benar → bobot E_t, salah → bobot 1
            w = np.where(misclassified == 1.0, 1.0, E_t)
        else:
            # belum ada learner → distribusi uniform
            E_t = 0.5
            acc_before = None
            w = np.ones(X.shape[0], dtype=float)

        w_sum = w.sum()
        if w_sum <= 0:
            F = np.ones_like(w) / len(w)
        else:
            F = w / w_sum  # D_t

        # 3) Latih base classifier baru h_t dengan distribusi F (Step 3)
        # Latih pertama kali
        # new_model = GaussianNaiveBayes().fit(X, y, sample_weight=F)
        new_model = self._base_learner_factory().fit(X, y, sample_weight=F)

        y_hat = new_model.predict(X)
        err = float(np.average((y_hat != y).astype(float), weights=F))

        # Jika error >= 0.5, latih ulang SEKALI (seperti tertulis di Algorithm 1)
        if err >= 0.5:
            # new_model = GaussianNaiveBayes().fit(X, y, sample_weight=F)
            new_model = self._base_learner_factory().fit(X, y, sample_weight=F)

        new_id = self._next_learner_id
        self._next_learner_id += 1

        new_info = BaseLearnerInfo(
            learner_id=new_id,
            model=new_model,
            created_at_slot=t,
            beta_history={},
        )
        self.learners.append(new_info)

        # new_model = GaussianNaiveBayes().fit(X, y, sample_weight=F)
        # new_info = BaseLearnerInfo(
        #     model=new_model,
        #     created_at_slot=t,
        #     beta_history={},
        # )
        # self.learners.append(new_info)

        # 4) Hitung epsilon_k^t utk semua learner di D_t (Step 4)
        epsilons: List[float] = []
        for info in self.learners:
            y_hat = info.model.predict(X)
            err = float(np.average((y_hat != y).astype(float), weights=F))

            # 5) Clamp ε_k^t ke (0, 0.5] (Step 4–5)
            if err > 0.5:
                err = 0.5
            err = max(err, 1e-6)

            epsilons.append(err)
            # β_k^t = ε_k^t / (1 − ε_k^t)
            info.beta_history[t] = err / (1.0 - err)

        # 6) Hitung betā_k^t: weighted average dgn sigmoid recency (Step 5)
        beta_bars: List[float] = []
        for info in self.learners:
            beta_bar = self._compute_beta_bar(info, current_slot=t)
            beta_bars.append(beta_bar)

        # 7) W_k^t = log(1 / betā_k^t) (Step 6)
        weights: List[float] = []
        for beta_bar in beta_bars:
            beta_bar = min(max(beta_bar, 1e-6), 1.0 - 1e-6)
            w_k = math.log(1.0 / beta_bar)
            weights.append(w_k)

        # self._weights = weights

        prune_info = {"pruned_id": None, "pruned_created_at": None, "pruned_idx": None}
        # print("BEFORE PRUNE:", len(self.learners), len(self._weights), len(weights))
        # 8) Pruning jika > max_learners (Xu: mereka pakai batas 20)
        if len(self.learners) > self.max_learners:
            prune_info = self._prune_oldest()

            # sinkronkan list lokal agar tetap aligned dengan self.learners
            idx = prune_info["pruned_idx"]
            del epsilons[idx]
            del beta_bars[idx]
            del weights[idx]

        # SET ulang weights internal setelah semua sinkron
        self._weights = list(weights)
        # pastikan benar-benar konsisten (ini wajib untuk riset)
        assert len(self.learners) == len(self._weights)
        assert len(epsilons) == len(beta_bars) == len(weights) == len(self.learners)

        num_learners_after = self.num_learners()

        # simpan diagnostik terakhir (opsional)
        self.last_E_t = E_t
        self.last_epsilons = epsilons
        self.last_beta_bars = beta_bars
        self.last_weights = weights

        if return_info:
            if self.learners:
                ensemble_pred_after = self._predict_batch(X)
                acc_after = float((ensemble_pred_after == y).mean())
            else:
                acc_after = None
            return {
                "slot_index": t,
                "num_samples": n_samples,
                "num_label_1": num_label_1,
                "num_label_0": num_label_0,
                "num_learners_before": num_learners_before,
                "num_learners_after": num_learners_after,
                "E_t": E_t,
                "acc_before": acc_before,
                "acc_after": acc_after,
                "epsilons": epsilons,
                "beta_bars": beta_bars,
                "weights": weights,
                "new_learner_id": new_id,
                "prune": prune_info,
            }
        return None

    # ----------------- fungsi internal -----------------

    def _prepare_xy(self, dataset: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        X_list: List[List[float]] = []
        y_list: List[int] = []
        for item in dataset:
            x = item["x"]
            y = int(item["y"])
            if len(x) != self.n_features:
                raise ValueError(f"n_features mismatch: expected {self.n_features}, got {len(x)}")
            X_list.append(x)
            y_list.append(y)

        X = np.asarray(X_list, dtype=float)
        y_arr = np.asarray(y_list, dtype=int)
        return X, y_arr

    def _predict_batch(self, X: np.ndarray) -> np.ndarray:
        if not self.learners:
            # default semua 1 (agresif) jika belum ada learner
            return np.ones(X.shape[0], dtype=int)

        n_samples = X.shape[0]
        scores_0 = np.zeros(n_samples, dtype=float)
        scores_1 = np.zeros(n_samples, dtype=float)

        for info, w in zip(self.learners, self._weights):
            y_hat = info.model.predict(X)
            mask1 = (y_hat == 1)
            mask0 = ~mask1
            scores_1[mask1] += w
            scores_0[mask0] += w

        return np.where(scores_1 >= scores_0, 1, 0)

    def _compute_beta_bar(self, info: BaseLearnerInfo, current_slot: int) -> float:
        """
        Menghitung beta_bar_k^t untuk satu learner berdasarkan semua beta_k^tau
        dari slot saat learner ini dibuat sampai slot t, dengan recency weighting
        sigmoid seperti (7)–(8) di Elwell & Polikar.
        """
        betas: List[float] = []
        weights: List[float] = []

        for tau, beta_tau in info.beta_history.items():
            # delta = t - tau (analog t - k di paper)
            delta = current_slot - tau
            # ω(delta) = 1 / (1 + exp(a * (delta - b)))   # delta besar → weight kecil
            s = 1.0 / (1.0 + math.exp(self.a * (delta - self.b)))
            betas.append(beta_tau)
            weights.append(s)

        if not betas:
            return 0.5  # fallback netral

        betas_arr = np.asarray(betas, dtype=float)
        w_arr = np.asarray(weights, dtype=float)
        w_sum = w_arr.sum()
        if w_sum <= 0:
            return float(betas_arr.mean())

        return float((betas_arr * w_arr).sum() / w_sum)
    
    def _prune_oldest(self) -> Dict[str, Any]:
        if not self.learners:
            return {"pruned_id": None, "pruned_created_at": None, "pruned_idx": None}
        
        # cari learner dengan created_at_slot terkecil
        oldest_idx = 0
        oldest_slot = self.learners[0].created_at_slot
        for i, info in enumerate(self.learners[1:], start=1):
            if info.created_at_slot < oldest_slot:
                oldest_slot = info.created_at_slot
                oldest_idx = i

        pruned = self.learners[oldest_idx]
        pruned_id = pruned.learner_id
        pruned_created = pruned.created_at_slot

        # hapus learner tersebut dan weight-nya
        del self.learners[oldest_idx]
        # del self._weights[oldest_idx]

        return {"pruned_id": pruned_id, "pruned_created_at": pruned_created, "pruned_idx": oldest_idx}


# Bismillah uji coba CacheAwareLearn++.NSE rev. Elwell, Xu, Chen
class CacheAwareLearnNSE(LearnNSE):
    """
    Cache-Aware Learn++.NSE v1 (untuk optimasi replikasi Xu):

    [A] Admission gating berbasis hist_len (hist_len<2 -> y_hat=0)
        -> BUKAN di kelas ini. Itu ada di layer simulasi cache (run_single_capacity).

    [B] Transform fitur gap: clip ke missing_gap_value lalu log1p
        -> menutup celah "missing gaps sebagai angka raksasa" yang merusak Gaussian NB.

    [C] Class-mass balancing pada instance-weight w
        -> menutup celah "kelas minoritas kehilangan mass" yang membuat model bias berat
           dan bisa memicu under-admission / collapse.
    """

    def __init__(
        self,
        n_features: int,
        a: float = 0.5,
        b: float = 10.0,
        max_learners: int = 20,
        base_learner_factory=None,
        lambda_freq: float = 1.0,
        alpha_prune: float = 0.5,
        protect_newest: int = 1,
        class_mass_balance: bool = True,
        missing_gap_value: float = 1e6,
        freq_cap: float = 3.0,
        balance_ema_alpha: float = 0.2,
        balance_ratio_cap: float = 3.0,
    ) -> None:
        super().__init__(n_features=n_features, a=a, b=b, max_learners=max_learners, base_learner_factory=base_learner_factory)
        self.lambda_freq = float(lambda_freq)
        self.alpha_prune = float(alpha_prune)
        self.protect_newest = int(protect_newest)
        # batas maksimal faktor frekuensi (1 berarti tidak aktif)
        self.freq_cap = float(freq_cap)
        self.balance_ema_alpha = float(balance_ema_alpha)
        self.balance_ratio_cap = float(balance_ratio_cap)

        self._ema_w_pos: Optional[float] = None
        self._ema_w_neg: Optional[float] = None
        self._ema_balance_ratio: Optional[float] = None

        # === [C] enable/disable class-mass balancing ===
        self.class_mass_balance = bool(class_mass_balance)

        # === [B] missing gap cap (harus sama dengan config) ===
        self.missing_gap_value = float(missing_gap_value)   # sentinel asli (mis. 1e6)
        self.G = float(missing_gap_value)                   # tetap untuk clip upper bound


    # ---------------------------------------------------------------------
    # [B] Feature transform (gap -> clip -> log1p)
    # ---------------------------------------------------------------------
    def _transform_X(self, X: np.ndarray) -> np.ndarray:
        """Transform untuk batch/matrix X: shape (n_samples, n_features)."""
        X = np.asarray(X, dtype=float)

        # baseline Xu: jangan bikin mask -> overhead minimal
        is_missing_aware = (self._base_learner_factory is GaussianNaiveBayesMissingAware)
        miss = (X == self.missing_gap_value) if is_missing_aware else None

        X = np.clip(X, 0.0, self.G)
        X = np.log1p(X)

        # hanya optimasi: sentinel dijadikan NaN supaya NB bisa skip dimensi missing
        if is_missing_aware and miss is not None:
            X[miss] = np.nan

        return X

    def _transform_x(self, x: Sequence[float]) -> np.ndarray:
        """Transform untuk single vector x: shape (n_features,)."""
        x_arr = np.asarray(x, dtype=float)

        is_missing_aware = (self._base_learner_factory is GaussianNaiveBayesMissingAware)
        miss = (x_arr == self.missing_gap_value) if is_missing_aware else None

        x_arr = np.clip(x_arr, 0.0, self.G)
        x_arr = np.log1p(x_arr)

        if is_missing_aware and miss is not None:
            x_arr[miss] = np.nan

        return x_arr

    def predict(self, x: Sequence[float]) -> int:
        # === [B] pastikan inference memakai transform yang sama dgn training ===
        x2 = self._transform_x(x)
        return super().predict(x2.tolist())

    def score_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Skor kontinu untuk batch dengan transform konsisten.
        """
        X2 = self._transform_X(X)
        return super().score_batch(X2)

    def score_one(self, x: Sequence[float]) -> float:
        """
        Skor kontinu untuk satu sampel dengan transform konsisten.
        """
        x2 = self._transform_x(x)
        return super().score_one(x2.tolist())

    # ---------------------------------------------------------------------
    # Learn++.NSE slot update
    # ---------------------------------------------------------------------
    def update_slot(self, dataset: List[Dict], return_info: bool = False) -> Optional[Dict[str, Any]]:
        """
        Update per slot (struktur Learn++.NSE tetap):
          1) Siapkan X,y,freq
          2) Hitung E_t dan base_w (Elwell/Xu)
          3) [optional] freq-aware weighting
          4) [C] class-mass balancing pada w
          5) Normalisasi -> F
          6) Train base learner baru
          7) Hitung eps, beta, beta_bar, weights
          8) Pruning ala Xu (buang tertua)
        """
        # slot kosong
        if not dataset:
            self.current_slot += 1
            if return_info:
                return {
                    "slot_index": self.current_slot,
                    "num_samples": 0,
                    "num_label_1": 0,
                    "num_label_0": 0,
                    "num_learners_before": self.num_learners(),
                    "num_learners_after": self.num_learners(),
                    "E_t": None,
                    "acc_before": None,
                    "acc_after": None,
                    "epsilons": [],
                    "beta_bars": [],
                    "weights": [],
                }
            return None

        self.current_slot += 1
        t = self.current_slot

        # --- siapkan X, y, freq ---
        X_list: List[List[float]] = []
        y_list: List[int] = []
        freq_list: List[float] = []

        for item in dataset:
            x = item["x"]
            yv = int(item["y"])
            f = float(item.get("freq", 1.0))

            if len(x) != self.n_features:
                raise ValueError(f"n_features mismatch: expected {self.n_features}, got {len(x)}")

            X_list.append(x)
            y_list.append(yv)
            freq_list.append(f)

        # === [B] training juga wajib pakai transform yang sama ===
        X_raw = np.asarray(X_list, dtype=float)
        X = self._transform_X(X_raw)

        y = np.asarray(y_list, dtype=int)
        freq_arr = np.asarray(freq_list, dtype=float)
        n_samples = X.shape[0]

        num_label_1 = int((y == 1).sum())
        num_label_0 = n_samples - num_label_1
        num_learners_before = self.num_learners()

        # normalisasi freq dalam slot (0,1] dengan log1p untuk meredam heavy-tail
        max_freq = max(float(freq_arr.max()), 1.0)
        log_denom = math.log1p(max_freq)
        if log_denom > 0.0:
            norm_freq = np.log1p(freq_arr) / log_denom
        else:
            norm_freq = np.zeros_like(freq_arr, dtype=float)

        # --- Step 2: E_t dan base_w (Elwell/Xu) ---
        if self.learners:
            ensemble_pred_before = self._predict_batch(X)

            misclassified = (ensemble_pred_before != y).astype(float)
            E_t = float(misclassified.mean())
            E_t = min(max(E_t, 1e-6), 1.0 - 1e-6)
            acc_before = 1.0 - E_t

            # Elwell/Xu: benar -> E_t, salah -> 1
            base_w = np.where(misclassified == 1.0, 1.0, E_t)
        else:
            ensemble_pred_before = np.ones(n_samples, dtype=int)

            E_t = 0.5
            acc_before = None
            base_w = np.ones(n_samples, dtype=float)

        # --- freq-aware weighting (adaptif) ---
        mean_norm = float(norm_freq.mean()) if n_samples > 0 else 0.0
        std_norm = float(norm_freq.std()) if n_samples > 0 else 0.0
        cv_norm = (std_norm / mean_norm) if mean_norm > 1e-12 else 0.0
        lambda_eff = self.lambda_freq / (1.0 + cv_norm)

        freq_factor = 1.0 + lambda_eff * norm_freq
        # cap dinamis agar distribusi bobot tidak terlalu tajam
        if self.freq_cap > 1.0:
            ff_max = float(freq_factor.max())
            if ff_max > self.freq_cap and ff_max > 1.0:
                scale = (self.freq_cap - 1.0) / (ff_max - 1.0)
                freq_factor = 1.0 + (freq_factor - 1.0) * scale
        w = base_w * freq_factor

        # audit mass sebelum balancing
        w_pos = float(w[y == 1].sum())
        w_neg = float(w[y == 0].sum())

        # === [C] CLASS-MASS BALANCING ===
        # target: setelah balancing, mass kelas-1 dan kelas-0 seimbang
        if self.class_mass_balance and (w_pos > 0.0) and (w_neg > 0.0):
            # stabilize mass estimates with EMA
            if self._ema_w_pos is None:
                self._ema_w_pos = w_pos
                self._ema_w_neg = w_neg
            else:
                alpha = self.balance_ema_alpha
                self._ema_w_pos = alpha * w_pos + (1.0 - alpha) * self._ema_w_pos
                self._ema_w_neg = alpha * w_neg + (1.0 - alpha) * self._ema_w_neg

            pos_mass = self._ema_w_pos
            neg_mass = self._ema_w_neg
            ratio_raw = neg_mass / pos_mass

            # smooth ratio across windows
            if self._ema_balance_ratio is None:
                self._ema_balance_ratio = ratio_raw
            else:
                alpha = self.balance_ema_alpha
                self._ema_balance_ratio = alpha * ratio_raw + (1.0 - alpha) * self._ema_balance_ratio

            ratio = self._ema_balance_ratio

            # cap ratio to avoid extreme weights
            cap = max(self.balance_ratio_cap, 1.0)
            if ratio > cap:
                ratio = cap
            elif ratio < 1.0 / cap:
                ratio = 1.0 / cap

            scale_pos = math.sqrt(ratio)
            scale_neg = 1.0 / scale_pos
            w = np.where(y == 1, w * scale_pos, w * scale_neg)

        # normalize -> F
        w_sum = float(w.sum())
        F = (w / w_sum) if (w_sum > 0.0) else (np.ones_like(w) / len(w))

        # --- Step 3: train base learner baru ---
        # new_model = GaussianNaiveBayes().fit(X, y, sample_weight=F)
        new_model = self._base_learner_factory().fit(X, y, sample_weight=F)
        y_hat_new = new_model.predict(X)
        err_new = float(np.average((y_hat_new != y).astype(float), weights=F))
        if err_new >= 0.5:
            # new_model = GaussianNaiveBayes().fit(X, y, sample_weight=F)
            new_model = self._base_learner_factory().fit(X, y, sample_weight=F)

        new_id = self._next_learner_id
        self._next_learner_id += 1
        self.learners.append(
            BaseLearnerInfo(
                learner_id=new_id,
                model=new_model,
                created_at_slot=t,
                beta_history={},
            )
        )

        # --- Step 4–7: epsilons, beta, beta_bar, W ---
        epsilons: List[float] = []
        for info in self.learners:
            y_hat = info.model.predict(X)
            err = float(np.average((y_hat != y).astype(float), weights=F))
            err = 0.5 if err > 0.5 else err
            err = max(err, 1e-6)

            epsilons.append(err)
            info.beta_history[t] = err / (1.0 - err)

        beta_bars: List[float] = [self._compute_beta_bar(info, current_slot=t) for info in self.learners]

        weights: List[float] = []
        for bb in beta_bars:
            bb = min(max(bb, 1e-6), 1.0 - 1e-6)
            weights.append(math.log(1.0 / bb))

        # --- Step 8: pruning ala Xu (buang tertua) ---
        prune_info = {"pruned_id": None, "pruned_created_at": None, "pruned_idx": None}
        if len(self.learners) > self.max_learners:
            prune_info = self._prune_oldest()
            idx = prune_info["pruned_idx"]
            if idx is not None:
                del epsilons[idx]
                del beta_bars[idx]
                del weights[idx]

        self._weights = list(weights)
        assert len(self.learners) == len(self._weights)
        assert len(epsilons) == len(beta_bars) == len(weights) == len(self.learners)

        # simpan diagnostik terakhir
        self.last_E_t = E_t
        self.last_epsilons = epsilons
        self.last_beta_bars = beta_bars
        self.last_weights = list(self._weights)

        if return_info:
            if self.learners:
                ensemble_pred_after = self._predict_batch(X)
                acc_after = float((ensemble_pred_after == y).mean())
            else:
                acc_after = None
            return {
                "slot_index": t,
                "num_samples": n_samples,
                "num_label_1": num_label_1,
                "num_label_0": num_label_0,
                "num_learners_before": num_learners_before,
                "num_learners_after": self.num_learners(),
                "E_t": E_t,
                "acc_before": acc_before,
                "acc_after": acc_after,
                "epsilons": epsilons,
                "beta_bars": beta_bars,
                "weights": list(self._weights),
                "new_learner_id": new_id,
                "prune": prune_info,
            }
        return None
