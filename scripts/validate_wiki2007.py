import os
import pyarrow.parquet as pq
import pyarrow as pa

PARQUET_DIR = "data/parquet/wikipedia_2007/"

def validate_all_parquet():
    files = sorted([
        os.path.join(PARQUET_DIR, f)
        for f in os.listdir(PARQUET_DIR)
        if f.endswith(".parquet")
    ])

    print(f"[INFO] Found {len(files)} parquet files")

    total_rows = 0
    example_rows = []
    unique_urls = set()
    fields_ok = True

    required_fields = {"unique_id", "timestamp", "requested_url", "is_update"}

    for fpath in files:
        print(f"[READ] {fpath}")

        # Streaming read: batch-by-batch
        parquet_file = pq.ParquetFile(fpath)

        for batch in parquet_file.iter_batches():
            table = pa.Table.from_batches([batch])
            df = table.to_pandas()

            # Field validation
            if set(df.columns) != required_fields:
                print(f"[ERROR] Field mismatch in file {fpath}")
                print(f"Columns present: {df.columns.tolist()}")
                fields_ok = False
                return

            # Count rows
            total_rows += len(df)

            # Capture first 10 rows
            if len(example_rows) < 10:
                needed = 10 - len(example_rows)
                example_rows.extend(df.head(needed).to_dict("records"))

            # Count unique objects
            unique_urls.update(df["requested_url"].tolist())

    print("\n================ VALIDATION REPORT ================\n")

    print(f"Total parquet files: {len(files)}")
    print(f"Total rows (should be 10,000,000): {total_rows}")
    print(f"Unique requested_url (object_id): {len(unique_urls)}")

    print("\nSample 10 rows (raw data):")
    for row in example_rows:
        print(row)

    print("\nField check:", "OK" if fields_ok else "FAILED")
    print("Fields expected:", required_fields)

    print("\n================ END OF REPORT ====================\n")


if __name__ == "__main__":
    validate_all_parquet()
