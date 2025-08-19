import polars as pl

# Input and output file paths
input_file = "./mt5/audusd_15min_klines.csv"   # Source file (TSV format)
output_file = "./mt5/audusd_15min_klines.csv"  # CSV output
is_tsv = False  # Set to True if input is TSV, False if CSV


if is_tsv:
    # Read TSV with UTF-8 encoding
    df = pl.read_csv(input_file, separator="\t", encoding="utf-8")

    # Rename columns to lowercase, more convenient names
    df = df.rename({
        "<DATE>": "date",
        "<TIME>": "time",
        "<OPEN>": "open",
        "<HIGH>": "high",
        "<LOW>": "low",
        "<CLOSE>": "close",
        "<TICKVOL>": "tick_volume",
        "<VOL>": "volume",
        "<SPREAD>": "spread"
    })
else:
    # Read CSV with UTF-8 encoding
    df = pl.read_csv(input_file, encoding="utf-8")

# Save as compressed CSV (gzip)
df.write_csv(output_file, separator=",")
df.write_parquet(f"{output_file}.parquet", compression="zstd")

print(f"✅ CSV file written to {output_file}")
