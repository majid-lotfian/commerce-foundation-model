
import argparse
from pathlib import Path

import duckdb


def sql_path(path):
    return str(Path(path).resolve()).replace("'", "''")


def clean_cpt(args):
    source = Path(args.input)
    output_dir = Path(args.output)
    temp_dir = Path(args.temp_dir)

    if not source.exists():
        raise FileNotFoundError(source)

    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)

    cleaned = output_dir / "cpt_clean.parquet"
    sample = output_dir / "cpt_sample.parquet"

    con = duckdb.connect()

    try:
        con.execute(f"SET memory_limit = '{args.memory}'")
        con.execute(f"SET threads = {args.threads}")
        con.execute("SET preserve_insertion_order = false")
        con.execute("SET enable_progress_bar = true")
        con.execute("SET progress_bar_time = 1000")
        con.execute(
            f"SET temp_directory = '{sql_path(temp_dir)}'"
        )

        src = sql_path(source)
        dst = sql_path(cleaned)
        sample_dst = sql_path(sample)

        print("\n[1/4] Reading and deduplicating CPT corpus...",
              flush=True)

        # Hash partitioning helps DuckDB distribute the work.
        # Exact text equality, rather than hash equality, is
        # used to determine duplicates.
        con.execute(
            f"""
            COPY (
                SELECT DISTINCT text
                FROM read_json_auto(
                    '{src}',
                    format = 'newline_delimited',
                    columns = {{'text': 'VARCHAR'}}
                )
                WHERE text IS NOT NULL
                  AND LENGTH(TRIM(text)) >= {args.min_length}
            )
            TO '{dst}'
            (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )

        print("\n[2/4] Counting cleaned documents...",
              flush=True)

        clean_count = con.execute(
            "SELECT COUNT(*) FROM read_parquet(?)",
            [str(cleaned)],
        ).fetchone()[0]

        print(f"Cleaned documents: {clean_count:,}", flush=True)

        print("\n[3/4] Measuring review-text duplicates...",
              flush=True)

        # Review text begins after the "Review: " marker.
        # This counts repeated review bodies, including
        # cases where the product information differs.
        duplicate_stats = con.execute(
            f"""
            WITH reviews AS (
                SELECT
                    split_part(text, 'Review: ', 2) AS review
                FROM read_parquet('{dst}')
            ),
            counts AS (
                SELECT review, COUNT(*) AS n
                FROM reviews
                WHERE review <> ''
                GROUP BY review
            )
            SELECT
                COALESCE(SUM(n), 0),
                COUNT(*),
                COALESCE(SUM(n - 1), 0)
            FROM counts
            """
        ).fetchone()

        print(
            f"Review bodies examined: {duplicate_stats[0]:,}"
        )
        print(
            f"Unique review bodies: {duplicate_stats[1]:,}"
        )
        print(
            f"Repeated review bodies: {duplicate_stats[2]:,}"
        )

        print("\n[4/4] Creating reproducible CPT sample...",
              flush=True)

        # Stable hash ranking gives the same sample on
        # repeated runs, regardless of Parquet row order.
        # This is a global sample, not category-stratified.
        con.execute(
            f"""
            COPY (
                SELECT text
                FROM read_parquet('{dst}')
                ORDER BY hash(text)
                LIMIT {args.sample_size}
            )
            TO '{sample_dst}'
            (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )

        sample_count = con.execute(
            "SELECT COUNT(*) FROM read_parquet(?)",
            [str(sample)],
        ).fetchone()[0]

        print("\nCPT cleaning completed.", flush=True)
        print(f"Cleaned corpus: {cleaned}")
        print(f"Cleaned documents: {clean_count:,}")
        print(f"Sample: {sample}")
        print(f"Sample documents: {sample_count:,}")

    finally:
        con.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default="data/cpt/cpt_train.jsonl.gz",
    )
    parser.add_argument(
        "--output",
        default="data/cpt/cleaned",
    )
    parser.add_argument(
        "--temp-dir",
        default="data/duckdb_tmp",
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=1_000_000,
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--memory",
        default="48GB",
    )

    clean_cpt(parser.parse_args())
