# src/data/trace_reader.py

import os
import tarfile, io, gzip
from typing import Dict, Generator, Optional, List, Tuple

import pyarrow.parquet as pq
import pyarrow as pa


class TraceReader:
    """
    Trace reader generik untuk eksperimen edge caching.

    - Jika `path` adalah direktori yang berisi file-file .parquet:
        → dibaca sebagai kumpulan chunk Parquet (seperti versi awal).

    - Jika `path` adalah file .gz (raw Wikipedia trace, Urdaneta et al.):
        → dibaca baris per baris dari file gzip.

    Output API diseragamkan:
        iter_requests() menghasilkan dict:

        {
            "unique_id": str,
            "timestamp": float,   # biasanya Unix time (detik, boleh ada pecahan)
            "object_id": str,     # ID objek untuk caching (requested_url)
            "is_update": str,     # flag apakah request menghasilkan save-operation
        }

    Sesuai paper Urdaneta: setiap request ditandai oleh:
        unique ID, timestamp, requested URL, save-flag.
    """

    def __init__(
        self,
        path: str,
        max_rows: Optional[int] = None,
    ) -> None:
        """
        :param path:
            - Direktori berisi .parquet  → mode="parquet"
            - File .gz                    → mode="raw_gz"
        :param max_rows: jika tidak None, batasi total request yang di-stream.
        """
        self.path = path
        self.max_rows = max_rows

        if os.path.isdir(path):
            # Mode Parquet
            self.mode = "parquet"
            self.files: List[str] = sorted(
                os.path.join(path, f)
                for f in os.listdir(path)
                if f.endswith(".parquet")
            )
            if not self.files:
                raise RuntimeError(f"Tidak ada file .parquet di direktori {path}")
        else:
            # Bukan direktori → cek ekstensi
            if path.endswith(".tar.gz"):
                self.mode = "tar_gz_wiki2018"
                if not os.path.isfile(path):
                    raise RuntimeError(f"File .tar.gz tidak ditemukan: {path}")
                self.files = [path]
            elif path.endswith("wiki2018.gz"):
                # sampel 10M baris, format sama dengan .tr tapi tanpa tar
                self.mode = "raw_gz_wiki2018"
                if not os.path.isfile(path):
                    raise RuntimeError(f"File .gz tidak ditemukan: {path}")
                self.files = [path]
            elif path.endswith(".gz"):
                self.mode = "raw_gz"
                if not os.path.isfile(path):
                    raise RuntimeError(f"File .gz tidak ditemukan: {path}")
                self.files = [path]
            else:
                raise RuntimeError(
                    f"path '{path}' bukan direktori .parquet dan bukan file .gz"
                )

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def iter_requests(self) -> Generator[Dict, None, None]:
        """
        Stream seluruh request dalam bentuk dict standar:

            {
                "unique_id": str,
                "timestamp": float,
                "object_id": str,
                "is_update": str,
            }
        """
        if self.mode == "tar_gz_wiki2018":
            yield from self._iter_requests_wiki2018_tar()
        elif self.mode == "raw_gz_wiki2018":
            yield from self._iter_requests_wiki2018_gz()
        elif self.mode == "parquet":
            yield from self._iter_requests_parquet()
        elif self.mode == "raw_gz":
            yield from self._iter_requests_raw_gz()
        else:
            raise RuntimeError(f"Mode tidak dikenal: {self.mode}")


    def _iter_requests_wiki2018_gz(self) -> Generator[Dict, None, None]:
        """
        Membaca wiki2018.gz: isi baris sama seperti file .tr di dalam tar,
        tapi langsung gzip tunggal: time id size extra
        """
        assert len(self.files) == 1
        fpath = self.files[0]

        count = 0
        with gzip.open(fpath, "rt", encoding="utf-8", errors="ignore") as text_f:
            for line in text_f:
                if self.max_rows is not None and count >= self.max_rows:
                    break
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 3:
                    continue

                t_str, id_str, size_str, *extra = parts
                try:
                    t_val = int(t_str)
                    obj_id = id_str
                    size = int(size_str)
                except ValueError:
                    continue

                yield {
                    "object_id": obj_id,
                    "timestamp": float(t_val),
                    "size": size,
                    # "extra": extra
                }
                count += 1

    def _iter_requests_wiki2018_tar(self):
        count = 0
        with gzip.open(self.path, "rb") as gz_f:
            with tarfile.open(fileobj=gz_f, mode="r:*") as tf:
                # cari file .tr
                member = next(m for m in tf.getmembers() if m.name.endswith(".tr"))
                f_raw = tf.extractfile(member)
                if f_raw is None:
                    raise RuntimeError("Tidak bisa extract *.tr dari tar")
                text_f = io.TextIOWrapper(f_raw, encoding="utf-8", errors="ignore")

                for line in text_f:
                    if self.max_rows is not None and count >= self.max_rows:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) < 3:
                        continue   # atau raise jika mau strict

                    t_str, id_str, size_str, *extra = parts
                    try:
                        t_val = int(t_str)
                        obj_id = id_str          # string sudah cukup
                        size = int(size_str)
                    except ValueError:
                        continue

                    yield {
                        "object_id": obj_id,
                        "timestamp": float(t_val),
                        "size": size,
                        # "extra": extra  # kalau mau disimpan
                    }
                    count += 1

    # ------------------------------------------------------------------
    # PARQUET MODE
    # ------------------------------------------------------------------

    def _iter_requests_parquet(self) -> Generator[Dict, None, None]:
        count = 0
        for fpath in self.files:
            parquet_file = pq.ParquetFile(fpath)

            for batch in parquet_file.iter_batches():
                table = pa.Table.from_batches([batch])
                df = table.to_pandas()

                expected = {"unique_id", "timestamp", "requested_url", "is_update"}
                if set(df.columns) != expected:
                    raise RuntimeError(
                        f"Schema tidak sesuai di file {fpath}: {df.columns.tolist()}"
                    )

                for _, row in df.iterrows():
                    if self.max_rows is not None and count >= self.max_rows:
                        return

                    ts_raw = row["timestamp"]
                    try:
                        ts_val = float(ts_raw)
                    except Exception:
                        raise ValueError(f"Timestamp tidak bisa dikonversi: {ts_raw}")

                    yield {
                        "unique_id": str(row["unique_id"]),
                        "timestamp": ts_val,
                        "object_id": str(row["requested_url"]),
                        "is_update": str(row["is_update"]),
                    }

                    count += 1
                    if self.max_rows is not None and count >= self.max_rows:
                        return

    # ------------------------------------------------------------------
    # RAW GZ MODE (untuk wikipedia_september_2007 & oktober_2007)
    # ------------------------------------------------------------------

    def _iter_requests_raw_gz(self) -> Generator[Dict, None, None]:
        """
        Membaca file Wikipedia raw .gz baris per baris.

        Sesuai deskripsi paper, tiap baris merepresentasikan satu request
        dengan 4 field utama: unique_id, timestamp, requested_url, is_update.

        PENTING (ASUMSI FORMAT TEKS):
        ------------------------------
        Karena saya tidak melihat isi mentah file di sini, saya gunakan
        asumsi **konservatif** yang sejalan dengan kalimat di paper:

            unique_id, timestamp, requested_url, is_update

        dan memperlakukannya sebagai:

            parts[0]  -> unique_id        (angka seperti 929840853)
            parts[1]  -> timestamp        (UNIX time 2007.xxx, misal 1190146243.326)
            parts[2:-1] digabung -> URL   (requested_url, biasanya 1 token)
            parts[-1] -> is_update        (flag, misal '-' atau '0'/'1')

        Kalau file Anda punya format berbeda, Anda HANYA perlu
        menyesuaikan fungsi `_parse_raw_wikibench_line` di bawah.
        """

        assert len(self.files) == 1, (
            "Mode raw_gz saat ini diasumsikan satu file per TraceReader. "
            f"files={self.files}"
        )
        fpath = self.files[0]

        count = 0
        with gzip.open(fpath, "rt", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if self.max_rows is not None and count >= self.max_rows:
                    return

                line = line.strip()
                if not line:
                    continue

                uid, ts_val, url, is_update = self._parse_raw_wikibench_line(line)

                yield {
                    "unique_id": uid,
                    "timestamp": ts_val,
                    "object_id": url,
                    "is_update": is_update,
                }

                count += 1
                if self.max_rows is not None and count >= self.max_rows:
                    return

    # ------------------------------------------------------------------
    # PARSER SATU BARIS RAW WIKIPEDIA
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_raw_wikibench_line(line: str) -> Tuple[str, float, str, str]:
        """
        Parser untuk satu baris trace raw Wikipedia (.gz).

        ASUMSI (berdasarkan deskripsi paper):
        -------------------------------------
        - Baris dipisah dengan whitespace (spasi / tab).
        - Urutan field: unique_id, timestamp, requested_url, is_update.
        - URL tidak mengandung spasi; jika ternyata ada token ekstra di tengah,
          token [2:-1] akan digabung dengan spasi.

        Jika layout file Anda beda (misalnya ada kolom tambahan di depan),
        ubah indeks-indeks di sini supaya cocok dengan data nyata.

        Fungsi ini SENGAJA ketat:
        - Kalau jumlah kolom < 4, ia raise error.
        - Kalau timestamp tidak bisa dikonversi ke float, ia raise error.
        """

        parts = line.split()
        if len(parts) < 4:
            raise ValueError(f"Baris terlalu pendek untuk diparse: {line!r}")

        uid_raw = parts[0]
        ts_raw = parts[1]
        # Gabungkan semua token kecuali pertama, kedua, dan terakhir
        # untuk mengantisipasi URL dengan spasi (meski jarang terjadi).
        url_tokens = parts[2:-1]
        url = " ".join(url_tokens)
        is_update = parts[-1]

        try:
            ts_val = float(ts_raw)
        except Exception as exc:
            raise ValueError(
                f"Timestamp tidak bisa dikonversi dari baris: {line!r}"
            ) from exc

        return str(uid_raw), ts_val, url, str(is_update)
