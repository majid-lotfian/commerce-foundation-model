
import argparse
import os
import time
from pathlib import Path

import duckdb


def main(args):
    start = time.perf_counter()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    temp_dir = Path(
        os.environ.get("TMPDIR", "data/tmp/duckdb")
    ) / "rating_targets"
    temp_dir.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()

    con.execute(f"SET threads = {args.threads}")
    con.execute(f"SET memory_limit = '{args.memory_limit}'")
    con.execute(
        "SET temp_directory = ?",
        [str(temp_dir.resolve())],
    )
    con.execute("SET preserve_insertion_order = false")

    print(f"CPU threads: {args.threads}", flush=True)
    print(f"Memory limit: {args.memory_limit}", flush=True)
    print(f"Temporary directory: {temp_dir}", flush=True)

    files = [
        str(Path(args.train).resolve()),
        str(Path(args.validation).resolve()),
        str(Path(args.test).resolve()),
    ]

    for path in files:
        if not Path(path).exists():
            raise FileNotFoundError(path)

    print("\nLoading interaction metadata...", flush=True)

    con.execute(
        """
        CREATE TEMP TABLE interactions AS
        SELECT
            user_id,
            parent_asin,
            timestamp,
            rating,
            CASE
                WHEN filename = ? THEN 'train'
                WHEN filename = ? THEN 'validation'
                WHEN filename = ? THEN 'test'
            END AS split
        FROM read_parquet(?, filename=true)
        WHERE user_id IS NOT NULL
          AND parent_asin IS NOT NULL
          AND timestamp IS NOT NULL
          AND rating BETWEEN 1 AND 5
        """,
        [files[0], files[1], files[2], files],
    )

    print(
        "Interaction metadata loaded in "
        f"{time.perf_counter() - start:.1f}s",
        flush=True,
    )

    print("\nComputing chronological eligibility...", flush=True)

    con.execute(
        """
        CREATE TEMP TABLE eligible AS
        WITH ordered AS (
            SELECT
                *,
                COUNT(*) OVER (
                    PARTITION BY user_id
                    ORDER BY timestamp
                    RANGE BETWEEN UNBOUNDED PRECEDING
                              AND CURRENT ROW
                ) AS interactions_through_timestamp,
                COUNT(*) OVER (
                    PARTITION BY user_id, timestamp
                ) AS same_timestamp_count,
                ROW_NUMBER() OVER (
                    PARTITION BY user_id
                    ORDER BY timestamp, parent_asin
                ) AS interaction_number
            FROM interactions
        )
        SELECT
            user_id,
            parent_asin,
            timestamp,
            rating,
            split,
            interaction_number,
            interactions_through_timestamp
                - same_timestamp_count AS prior_interactions
        FROM ordered
        WHERE interactions_through_timestamp
                - same_timestamp_count >= ?
        """,
        [args.min_history],
    )

    print(
        "Eligibility computed in "
        f"{time.perf_counter() - start:.1f}s",
        flush=True,
    )

    limits = {
        "train": args.train_examples,
        "validation": args.validation_examples,
        "test": args.test_examples,
    }

    
    for split, limit in limits.items():
        print(
            f"\nSelecting {limit:,} {split} targets...",
            flush=True,
        )

        output_path = output_dir / f"{split}_targets.parquet"

        # Safely construct the output path as a SQL string literal.
        sql_path = str(output_path.resolve()).replace("'", "''")

        con.execute(
            f"""
            COPY (
                SELECT
                    user_id,
                    parent_asin,
                    timestamp,
                    rating,
                    split,
                    interaction_number,
                    prior_interactions
                FROM eligible
                WHERE split = ?
                ORDER BY hash(
                    user_id,
                    parent_asin,
                    timestamp,
                    rating
                )
                LIMIT ?
            )
            TO '{sql_path}'
            (FORMAT PARQUET, COMPRESSION ZSTD)
            """,
            [split, limit],
        )

        count = con.execute(
            "SELECT COUNT(*) FROM read_parquet(?)",
            [str(output_path.resolve())],
        ).fetchone()[0]

        print(f"Saved {count:,} targets: {output_path}", flush=True)


    con.close()

    print(
        f"\nStage 1 completed in "
        f"{time.perf_counter() - start:.1f} seconds",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--train", default="data/splits/train.parquet")
    parser.add_argument(
        "--validation",
        default="data/splits/validation.parquet",
    )
    parser.add_argument("--test", default="data/splits/test.parquet")

    parser.add_argument(
        "--output-dir",
        default="data/sft/rating/targets",
    )

    parser.add_argument("--min-history", type=int, default=5)

    parser.add_argument("--train-examples", type=int, default=100000)
    parser.add_argument("--validation-examples", type=int, default=10000)
    parser.add_argument("--test-examples", type=int, default=10000)

    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--memory-limit", default="48GB")

    main(parser.parse_args())
