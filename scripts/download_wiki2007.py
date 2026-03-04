import os
import requests
from bs4 import BeautifulSoup
import gzip
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

BASE_URL = "http://www.globule.org/wiki/2007-10/"
PARQUET_DIR = "data/parquet/wikipedia_2007/"
TOTAL_ROWS = 10_000_000
CHUNK_SIZE = 100_000

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def list_gz_files():
    print("[INFO] Fetching directory listing...")
    r = requests.get(BASE_URL)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    files = []
    for link in soup.find_all("a"):
        href = link.get("href")
        if href and href.endswith(".gz"):
            files.append(href)

    print(f"[INFO] Found {len(files)} .gz files on server")
    return files

def stream_gz_from_web(fname):
    """Streaming .gz directly without saving to disk."""
    url = BASE_URL + fname
    r = requests.get(url, stream=True)
    r.raise_for_status()

    decompressor = gzip.GzipFile(fileobj=r.raw, mode='rb')
    for line in decompressor:
        yield line.decode('utf-8', errors='ignore')

def write_parquet_chunk(rows, chunk_id):
    table = pa.Table.from_pylist(rows)
    fname = f"wiki2007_{chunk_id:03d}.parquet"
    fpath = os.path.join(PARQUET_DIR, fname)
    pq.write_table(table, fpath)
    print(f"[WRITE] Saved {fpath} ({len(rows)} rows)")
    return

def main():
    ensure_dir(PARQUET_DIR)

    files = list_gz_files()

    total = 0
    chunk_id = 0
    rows = []

    for fname in files:
        print(f"[PROCESS] Streaming {fname} ...")

        for line in tqdm(stream_gz_from_web(fname), desc=f"Reading {fname}"):
            parts = line.strip().split(" ", 3)
            if len(parts) != 4:
                continue

            unique_id, timestamp, requested_url, is_update = parts

            rows.append({
                "unique_id": unique_id,
                "timestamp": timestamp,
                "requested_url": requested_url,
                "is_update": is_update
            })

            total += 1

            if len(rows) >= CHUNK_SIZE:
                write_parquet_chunk(rows, chunk_id)
                rows = []
                chunk_id += 1

            if total >= TOTAL_ROWS:
                if rows:
                    write_parquet_chunk(rows, chunk_id)
                print(f"[DONE] Collected exactly {TOTAL_ROWS} rows.")
                return

    print(f"[FINISHED] Only {total} rows available, reached end of all files.")

if __name__ == "__main__":
    main()
