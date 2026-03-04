import gzip
import tarfile
import io

in_path = "data/raw/wiki2018/wiki2018.tr.tar.gz"
out_path = "data/raw/wiki2018/wiki2018.gz"

max_rows = 10_000_000
count = 0

with gzip.open(in_path, "rb") as gz_f:
    # buka tar di dalam .gz
    with tarfile.open(fileobj=gz_f, mode="r:*") as tf:
        # cari file .tr di dalam tar
        member = next(m for m in tf.getmembers() if m.name.endswith(".tr"))
        f_raw = tf.extractfile(member)
        if f_raw is None:
            raise RuntimeError("Tidak bisa extract *.tr dari tar")

        # bungkus jadi text file
        text_f = io.TextIOWrapper(f_raw, encoding="utf-8", errors="ignore")

        # tulis prefix 10M baris ke file gzip baru (.gz)
        with gzip.open(out_path, "wt", encoding="utf-8") as out_f:
            for line in text_f:
                out_f.write(line)
                count += 1
                if count >= max_rows:
                    break

print("Total rows written:", count)
print("Output file        :", out_path)
