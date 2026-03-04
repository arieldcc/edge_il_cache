import gzip

path = "data/raw/wiki2018/wiki2018.tr.tar.gz"

count = 0
with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as f:
    for _ in f:
        count += 1

print("Total rows:", count)
