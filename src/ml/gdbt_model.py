# src/ml/gdbt_model.py

"""
GDBT (Gradient Boosted Decision Trees) untuk Edge Caching

Replika baseline GDBT seperti pada Xu et al.:

- Family model: Gradient-Boosted Decision Trees (GDBT).
- Fitur input: access features Gap1..GapL (default L=6, diatur oleh ILConfig.num_gaps).
- Label: biner (1 = popular, 0 = not popular), biasanya:
    y = 1 untuk objek yang termasuk top-20% populer per slot (ILConfig.pop_top_percent).
- Training: offline / batch, bukan incremental.
- Jadwal update: model di-rebuild dari awal setiap
    GDBTConfig.update_interval_requests (default 1_000_000) request.
- Iterasi boosting: GDBTConfig.n_estimators (default 30),
  mengikuti setting "30 iterations" yang disebut di paper.

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
from sklearn.ensemble import GradientBoostingClassifier
import warnings

# Pastikan proyek bisa di-import jika modul ini dijalankan langsung
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config.experiment_config import ILConfig, GDBTConfig  # type: ignore

warnings.filterwarnings("ignore", category=UserWarning)


@dataclass
class GDBTCachePredictor:
    """
    GDBT-based cache admission predictor.

    Objek ini merepresentasikan baseline GDBT di Xu et al.:

    - Menggunakan fitur akses (Gap1..GapL) yang sama dengan IL (ILConfig.num_gaps).
    - Menggunakan schema label yang sama (top pop_top_percent per slot = 1).
    - Melakukan training ulang (rebuild) secara periodik, setiap
      GDBTConfig.update_interval_requests request.
    - Setiap rebuild menggunakan seluruh buffer data saat itu (offline batch).

    Parameter:
    ----------
    il_config : ILConfig
        Konfigurasi IL yang memuat:
        - num_gaps: jumlah fitur gap (L)
        - pop_top_percent: persen teratas objek populer yang dilabeli 1
    gdbt_config : Optional[GDBTConfig]
        Konfigurasi GDBT:
        - n_estimators: jumlah tree (iterations)
        - learning_rate, max_depth, dsb.
        - update_interval_requests: jarak request antar rebuild
        - random_state: seed (optional, untuk reprodusibilitas)

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
    model: Optional[GradientBoostingClassifier] = None
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
    # Internal: pembuatan model GDBT =
    # -------------------------------------------------------------------------

    def _create_model(self) -> GradientBoostingClassifier:
        """
        Membuat instance GradientBoostingClassifier.

        Hyperparameter utama yang disinkronkan dengan artikel:
        - n_estimators = 30 (default di GDBTConfig)
        - update_interval_requests diatur di konfigurasi, bukan di sini.

        Hyperparameter lain (learning_rate, max_depth, dsb.) mengikuti
        GDBTConfig sebagai aproksimasi terhadap [9].
        """
        return GradientBoostingClassifier(
            n_estimators=self.gdbt_config.n_estimators,
            learning_rate=self.gdbt_config.learning_rate,
            max_depth=self.gdbt_config.max_depth,
            min_samples_split=self.gdbt_config.min_samples_split,
            min_samples_leaf=self.gdbt_config.min_samples_leaf,
            random_state=self.gdbt_config.random_state,
        )

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

        Parameters
        ----------
        dataset : Sequence[Dict[str, Any]]
            List sample, masing-masing dengan key 'x' dan 'y'.
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

        Parameters
        ----------
        requests_since_last_rebuild : int
            Jumlah request sejak rebuild terakhir (atau sejak awal training).

        Returns
        -------
        bool
            True jika requests_since_last_rebuild >= update_interval_requests.
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

        Sesuai dengan GDBT di Xu et al.:
        - GDBT menjalankan training secara offline per window besar (1M req).
        - Pada saat rebuild, model lama dibuang, model baru dilatih dari awal
          menggunakan seluruh buffer training terkini (X_buffer, y_buffer).
        - Tidak ada incremental update ke model lama.

        Returns
        -------
        Dict[str, Any]
            Informasi tentang proses rebuild (sukses/tidak, ukuran buffer, dll).
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

        # Buat model baru dan latih dari awal
        self.model = self._create_model()

        try:
            self.model.fit(X, y)
            self.num_rebuilds += 1
            self.total_training_samples += len(X)

            # Monitoring: akurasi pada training data
            y_pred = self.model.predict(X)
            train_accuracy = float((y_pred == y).mean())

            rebuild_info.update(
                {
                    "success": True,
                    "num_samples": len(X),
                    "num_label_1": int((y == 1).sum()),
                    "num_label_0": int((y == 0).sum()),
                    "train_accuracy": train_accuracy,
                    # Atribut ini ada di sklearn >= 0.24
                    "n_estimators_actual": getattr(self.model, "n_estimators_", None),
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

        Parameters
        ----------
        x : Sequence[float]
            Vektor fitur Gap1..GapL.

        Returns
        -------
        int
            1 jika objek diprediksi "popular" (layak di-cache),
            0 jika tidak.
        """
        if self.model is None:
            # Belum ada model → default: admit (agresif, mirip IL awal)
            return 1

        x_arr = np.asarray(x, dtype=float).reshape(1, -1)
        if x_arr.shape[1] != self.n_features:
            raise ValueError(
                f"n_features mismatch: expected {self.n_features}, got {x_arr.shape[1]}"
            )

        return int(self.model.predict(x_arr)[0])

    def predict_proba(self, x: Sequence[float]) -> float:
        """
        Mengembalikan probabilitas kelas positif (popular) jika model mendukung.

        Jika model belum dilatih, kembalikan 1.0 (default admit).
        """
        if self.model is None:
            return 1.0

        x_arr = np.asarray(x, dtype=float).reshape(1, -1)
        if x_arr.shape[1] != self.n_features:
            raise ValueError(
                f"n_features mismatch: expected {self.n_features}, got {x_arr.shape[1]}"
            )

        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(x_arr)[0, 1]
            return float(proba)
        else:
            # fallback: gunakan prediksi keras sebagai pseudo-probabilitas
            return float(self.predict(x))

    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Prediksi batch.

        Parameters
        ----------
        X : np.ndarray
            Array shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Array shape (n_samples,) berisi 0/1.
        """
        if self.model is None:
            return np.ones(X.shape[0], dtype=int)

        if X.shape[1] != self.n_features:
            raise ValueError(
                f"n_features mismatch: expected {self.n_features}, got {X.shape[1]}"
            )

        return self.model.predict(X).astype(int)

    # -------------------------------------------------------------------------
    # Analisis & monitoring
    # -------------------------------------------------------------------------

    def get_feature_importances(self) -> Optional[np.ndarray]:
        """
        Ambil feature importances dari model (jika sudah dilatih).

        Returns
        -------
        Optional[np.ndarray]
            Importance untuk setiap fitur, atau None jika belum ada model.
        """
        if self.model is None:
            return None
        return getattr(self.model, "feature_importances_", None)

    def get_stats(self) -> Dict[str, Any]:
        """
        Statistik model saat ini, berguna untuk logging / evaluasi.

        Returns
        -------
        Dict[str, Any]
            Informasi jumlah rebuild, jumlah sample, dll.
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
        }


def create_default_gdbt_predictor(
    il_config: Optional[ILConfig] = None,
    gdbt_config: Optional[GDBTConfig] = None,
) -> GDBTCachePredictor:
    """
    Helper untuk membuat GDBTCachePredictor dengan konfigurasi default
    yang konsisten dengan Xu et al. jika caller tidak menyuplai config eksplisit.
    """
    il_conf = il_config or ILConfig()
    gdbt_conf = gdbt_config or GDBTConfig()
    return GDBTCachePredictor(il_config=il_conf, gdbt_config=gdbt_conf)
