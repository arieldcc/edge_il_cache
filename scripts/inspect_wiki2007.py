import gzip
import os

# PATH Lokasi dataset mentah
RAW_FILE = "data/raw/wikipedia_2007/wiki.1191201596.gz"

def inspect_raw_gz(path, max_preview_lines=20):
    if not os.path.exists(path):
        print(f"[ERROR] File not found: {path}")
        return

    print(f"[INFO] Inspecting file: {path}\n")

    preview_lines = []
    total_lines = 0
    col_count_freq = {}  # jumlah kolom -> berapa baris
    first_line = None

    # Baca semua baris (streaming) untuk statistik, tapi simpan hanya beberapa awal
    with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.rstrip("\n")
            if total_lines == 0:
                first_line = line

            if len(preview_lines) < max_preview_lines:
                preview_lines.append(line)

            # hitung jumlah kolom
            # gunakan split biasa, jangan pakai limit dulu, untuk lihat realita
            parts = line.split()
            ncols = len(parts)
            col_count_freq[ncols] = col_count_freq.get(ncols, 0) + 1

            total_lines += 1

    print("========== RAW PREVIEW (first "
          f"{len(preview_lines)} lines) ==========\n")
    for i, l in enumerate(preview_lines):
        print(f"LINE {i:02d}: {l}")
    print("\n============================================\n")

    # Deteksi header sederhana:
    # Jika baris pertama tidak tampak seperti 4 kolom: [id, ts, url, flag],
    # atau kolom pertama bukan angka → kandidat header.
    header_suspect = False
    header_reason = []

    if first_line is not None:
        parts = first_line.split()
        if len(parts) != 4:
            header_suspect = True
            header_reason.append(f"first line has {len(parts)} columns (expected 4 for WikiBench-style trace)")

        # coba cek apakah kolom pertama numeric
        try:
            _ = float(parts[0])
        except Exception:
            header_suspect = True
            header_reason.append("first column is not numeric (cannot be unique_id)")

        # cek kolom kedua numeric (timestamp)
        if len(parts) >= 2:
            try:
                _ = float(parts[1])
            except Exception:
                header_suspect = True
                header_reason.append("second column is not numeric (cannot be timestamp)")

    print("========== HEADER / FIRST LINE ANALYSIS ==========\n")
    print("First line:")
    print(first_line if first_line is not None else "[EMPTY FILE]")
    print()

    if header_suspect:
        print("[SUSPECT] First line looks like HEADER or non-data.")
        print("Reason(s):")
        for r in header_reason:
            print(" -", r)
    else:
        print("[INFO] First line looks like a DATA row, not a separate header.")
    print("\n============================================\n")

    print("========== COLUMN COUNT STATISTICS ==========\n")
    print(f"Total lines (excluding any interpretation of header): {total_lines}")
    print("Distribution of number of columns per line:")
    for ncols, freq in sorted(col_count_freq.items()):
        print(f"  {ncols} columns: {freq} lines")
    print("\n============================================\n")

    print("========== PARSED SAMPLE ROWS ==========\n")
    # Tampilkan parse kolom untuk beberapa baris awal
    for i, l in enumerate(preview_lines):
        parts = l.split()
        print(f"LINE {i:02d}:")
        print("  raw :", l)
        print("  cols:", len(parts))
        for j, p in enumerate(parts):
            print(f"    col[{j}] = {repr(p)}")
        print()
    print("============================================\n")


if __name__ == "__main__":
    inspect_raw_gz(RAW_FILE)