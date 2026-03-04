# src/ml/gdbt_model_xu.py

"""
GDBT (Gradient Boosted Decision Trees) untuk Edge Caching

Replika baseline GDBT seperti pada Xu et al. dan Berger et al. (LFO):

- Family model: Gradient-Boosted Decision Trees (GBDT).
- Library: LightGBM (seperti dinyatakan eksplisit oleh Berger et al.).
- Fitur input: access features Gap1..GapL (default L=6, diatur oleh ILConfig.num_gaps).
- Label: biner (1 = popular, 0 = not popular), biasanya:
    y = 1 untuk objek yang termasuk top-20% populer per slot (ILConfig.pop_top_percent).
- Training: offline / batch, bukan incremental.
- Jadwal update: model di-rebuild dari awal setiap
    GDBTConfig.update_interval_requests (default 1_000_000) request.
- Iterasi boosting: GDBTConfig.n_estimators (default 30),
  mengikuti setting "30 iterations" yang disebut Xu, merujuk [9] (Berger).

Seluruh hyperparameter LightGBM selain jumlah iterasi
menggunakan NILAI DEFAULT library (tidak di-set eksplisit), karena
baik Xu maupun Berger tidak menjelaskan detail parameter tersebut.

Kelas ini hanya menangani:
- penyimpanan buffer (X, y) dalam satu atau beberapa slot,
- memutuskan kapan saatnya rebuild model,
- melakukan retrain dari scratch,
- prediksi keputusan admit (1) / reject (0) untuk sebuah objek.

Pembangunan fitur (Gap1..GapL) dan label (top-20% popularity per slot)
dilakukan di pipeline luar yang memanggil add_training_sample / add_training_batch.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import lightgbm as lgb
import warnings

# Pastikan proyek bisa di-import jika modul ini dijalankan langsung
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config.experiment_config import ILConfig, GDBTConfig  # type: ignore

warnings.filterwarnings("ignore", category=UserWarning)


@dataclass
class GDBTCachePredictor:
    """
    GDBT-based cache admission predictor (LightGBM).

    Objek ini merepresentasikan baseline GDBT di Xu et al., yang
    merujuk skema GBDT Berger et al. (LFO):

    - Menggunakan fitur akses (Gap1..GapL) yang sama dengan IL (ILConfig.num_gaps).
    - Menggunakan schema label yang sama (top pop_top_percent per slot = 1).
    - Melakukan training ulang (rebuild) secara periodik, setiap
      GDBTConfig.update_interval_requests request.
    - Setiap rebuild menggunakan seluruh buffer data saat itu (offline batch).
    - Model adalah LightGBM GBDT dengan num_boost_round = GDBTConfig.n_estimators.
      Hyperparameter lain dibiarkan default.

    Catatan:
    --------
    - Kelas ini tidak mengelola cache secara langsung. Ia hanya memutuskan
      admit(1)/reject(0) berdasarkan fitur akses.
    - Underlying eviction policy (mis. LRU) ditangani oleh lapisan cache terpisah.
    """

    il_config: ILConfig
    gdbt_config: GDBTConfig = GDBTConfig()

    # Akan diinisialisasi di __post_init__
    n_features: int = 0
    model: Optional[lgb.Booster] = None
    _X_buffer: List[List[float]] = None  # type: ignore
    _y_buffer: List[int] = None          # type: ignore

    # Statistik / monitoring
    num_rebuilds: int = 0
    total_training_samples: int = 0
    last_rebuild_info: Dict[str, Any] = None  # type: ignore

    def __post_init__(self) -> None:
        # Jumlah fitur mengikuti ILConfig.num_gaps (Gap1..GapL)
        self.n_features = int(self.il_config.num_gaps)

        # Inisialisasi buffer training
        self._X_buffer = []
        self._y_buffer = []

        self.num_rebuilds = 0
        self.total_training_samples = 0
        self.last_rebuild_info = {}

    # -------------------------------------------------------------------------
    # Internal: konfigurasi LightGBM
    # -------------------------------------------------------------------------

    def _create_lgb_params(self) -> Dict[str, Any]:
        """
        Membuat dict parameter LightGBM.

        Hanya parameter yang eksplisit di Xu/Berger yang diatur:
        - objective: binary classification
        - boosting_type: gbdt

        Semua parameter lain (learning_rate, num_leaves, max_depth, dll.)
        dibiarkan mengikuti nilai default LightGBM.
        """
        return {
            "objective": "binary",
            "boosting_type": "gbdt",
            "verbose": -1,
        }

    # -------------------------------------------------------------------------
    # API untuk membangun dataset training (dipanggil oleh pipeline slot)
    # -------------------------------------------------------------------------

    def add_training_sample(self, x: Sequence[float], y: int) -> None:
        """
        Menambahkan satu sample ke buffer training.

        Parameters
        ----------
        x : Sequence[float]
            Vektor fitur Gap1..GapL (L = il_config.num_gaps).
        y : int
            Label biner (1 = popular, 0 = not popular), biasanya:
            y = 1 untuk objek yang termasuk top pop_top_percent di slot.
        """
        if len(x) != self.n_features:
            raise ValueError(
                f"n_features mismatch: expected {self.n_features}, got {len(x)}"
            )
        self._X_buffer.append(list(x))
        self._y_buffer.append(int(y))

    def add_training_batch(self, dataset: Sequence[Dict[str, Any]]) -> None:
        """
        Menambahkan batch sample dari dataset slot.

        Dataset per slot biasanya disusun oleh pipeline di luar dengan skema:
            {"x": List[float], "y": int}
        """
        for item in dataset:
            self.add_training_sample(item["x"], item["y"])

    def get_buffer_size(self) -> int:
        """Mengembalikan jumlah sample di buffer saat ini."""
        return len(self._X_buffer)

    # -------------------------------------------------------------------------
    # Jadwal rebuild (sesuai 1M request di artikel)
    # -------------------------------------------------------------------------

    def should_rebuild(self, requests_since_last_rebuild: int) -> bool:
        """
        Cek apakah sudah waktunya rebuild model.
        """
        return (
            requests_since_last_rebuild
            >= int(self.gdbt_config.update_interval_requests)
        )

    # -------------------------------------------------------------------------
    # Rebuild model (offline batch training)
    # -------------------------------------------------------------------------

    def rebuild_model(self) -> Dict[str, Any]:
        """
        Rebuild model dari scratch menggunakan data di buffer.

        Sesuai dengan GDBT di Xu et al. dan Berger:
        - GBDT menjalankan training secara offline per window besar (1M req).
        - Pada saat rebuild, model lama dibuang, model baru dilatih dari awal
          menggunakan seluruh buffer training terkini (X_buffer, y_buffer).
        - Tidak ada incremental update ke model lama.
        """
        rebuild_info: Dict[str, Any] = {
            "rebuild_number": self.num_rebuilds + 1,
            "buffer_size": len(self._X_buffer),
            "success": False,
        }

        if len(self._X_buffer) == 0:
            rebuild_info["error"] = "Empty buffer, cannot rebuild"
            self.last_rebuild_info = rebuild_info
            return rebuild_info

        X = np.asarray(self._X_buffer, dtype=float)
        y = np.asarray(self._y_buffer, dtype=int)

        unique_classes = np.unique(y)
        if len(unique_classes) < 2:
            # Hanya satu kelas, tidak bisa melatih classifier biner
            rebuild_info["error"] = f"Only one class in buffer: {unique_classes}"
            rebuild_info["fallback"] = "Using previous model or default prediction"
            self.last_rebuild_info = rebuild_info
            return rebuild_info

        # LightGBM dataset
        train_data = lgb.Dataset(X, label=y)

        params = self._create_lgb_params()
        num_boost_round = int(self.gdbt_config.n_estimators)

        try:
            # Latih model baru dari awal
            self.model = lgb.train(
                params=params,
                train_set=train_data,
                num_boost_round=num_boost_round,
            )
            self.num_rebuilds += 1
            self.total_training_samples += len(X)

            # Monitoring: akurasi pada training data
            y_prob = self.model.predict(X)
            y_pred = (y_prob >= 0.5).astype(int)
            train_accuracy = float((y_pred == y).mean())

            rebuild_info.update(
                {
                    "success": True,
                    "num_samples": len(X),
                    "num_label_1": int((y == 1).sum()),
                    "num_label_0": int((y == 0).sum()),
                    "train_accuracy": train_accuracy,
                    "n_estimators_actual": int(self.model.num_trees()),
                }
            )

        except Exception as e:
            rebuild_info["error"] = str(e)

        self.last_rebuild_info = rebuild_info
        return rebuild_info

    def clear_buffer(self) -> None:
        """
        Kosongkan buffer training setelah rebuild.
        Dipanggil biasanya setelah rebuild_model() sukses,
        sebelum mengumpulkan data untuk window berikutnya.
        """
        self._X_buffer = []
        self._y_buffer = []

    # -------------------------------------------------------------------------
    # Prediksi admit / reject untuk satu objek atau batch
    # -------------------------------------------------------------------------

    def predict(self, x: Sequence[float]) -> int:
        """
        Prediksi kelas (0/1) untuk satu sample.
        """
        if self.model is None:
            # Belum ada model → default: admit (agresif, mirip IL awal)
            return 1

        x_arr = np.asarray(x, dtype=float).reshape(1, -1)
        if x_arr.shape[1] != self.n_features:
            raise ValueError(
                f"n_features mismatch: expected {self.n_features}, got {x_arr.shape[1]}"
            )

        proba = float(self.model.predict(x_arr)[0])
        return int(proba >= 0.5)

    def predict_proba(self, x: Sequence[float]) -> float:
        """
        Mengembalikan probabilitas kelas positif (popular).

        Jika model belum dilatih, kembalikan 1.0 (default admit).
        """
        if self.model is None:
            return 1.0

        x_arr = np.asarray(x, dtype=float).reshape(1, -1)
        if x_arr.shape[1] != self.n_features:
            raise ValueError(
                f"n_features mismatch: expected {self.n_features}, got {x_arr.shape[1]}"
            )

        proba = float(self.model.predict(x_arr)[0])
        return proba

    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Prediksi batch: mengembalikan 0/1 untuk setiap baris X.
        """
        if self.model is None:
            return np.ones(X.shape[0], dtype=int)

        if X.shape[1] != self.n_features:
            raise ValueError(
                f"n_features mismatch: expected {self.n_features}, got {X.shape[1]}"
            )

        y_prob = self.model.predict(X)
        return (y_prob >= 0.5).astype(int)

    # -------------------------------------------------------------------------
    # Analisis & monitoring
    # -------------------------------------------------------------------------

    def get_feature_importances(self) -> Optional[np.ndarray]:
        """
        Ambil feature importances dari model (jika sudah dilatih).
        """
        if self.model is None:
            return None
        try:
            return self.model.feature_importance()
        except Exception:
            return None

    def get_stats(self) -> Dict[str, Any]:
        """
        Statistik model saat ini, berguna untuk logging / evaluasi.
        """
        return {
            "num_rebuilds": self.num_rebuilds,
            "total_training_samples": self.total_training_samples,
            "current_buffer_size": self.get_buffer_size(),
            "has_model": self.model is not None,
            "last_rebuild_info": self.last_rebuild_info,
            "n_features": self.n_features,
            "pop_top_percent": self.il_config.pop_top_percent,
            "update_interval_requests": self.gdbt_config.update_interval_requests,
            "n_estimators": self.gdbt_config.n_estimators,
            "n_estimators_actual": (
                int(self.model.num_trees()) if self.model is not None else None
            ),
        }


def create_default_gdbt_predictor(
    il_config: Optional[ILConfig] = None,
    gdbt_config: Optional[GDBTConfig] = None,
) -> GDBTCachePredictor:
    """
    Helper untuk membuat GDBTCachePredictor dengan konfigurasi default
    yang konsisten dengan Xu et al. dan Berger et al.

    Hanya parameter yang eksplisit di artikel yang digunakan:
    - n_estimators (jumlah boosting rounds)
    - update_interval_requests (1M request per rebuild)

    Hyperparameter lain mengikuti nilai default LightGBM.
    """
    il_conf = il_config or ILConfig()
    gdbt_conf = gdbt_config or GDBTConfig()
    return GDBTCachePredictor(il_config=il_conf, gdbt_config=gdbt_conf)
