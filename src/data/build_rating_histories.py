
import argparse
import os
import time
from pathlib import Path

import duckdb


def main(args):
    start = time.perf_counter()

    target_dir = Path(args.target_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = [
        str(Path(args.train).resolve()),
        str(Path(args.validation).resolve()),
        str(Path(args.test).resolve()),
    ]

    targets = [
        str((target_dir / f"{split}_targets.parquet").resolve())
        for split in ("train", "validation", "test")
    ]

    for path in files + targets:
        if not Path(path).exists():
            raise FileNotFoundError(path)

    temp_dir = (
        Path(os.environ.get("TMPDIR", "data/tmp/duckdb"))
        / "rating_histories"
    )
    temp_dir.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()

    con.execute(f"SET threads = {args.threads}")
    con.execute(f"SET memory_limit = '{args.memory_limit}'")
    con.execute("SET preserve_insertion_order = false")

    sql_temp = str(temp_dir.resolve()).replace("'", "''")
    con.execute(f"SET temp_directory = '{sql_temp}'")

    print(f"Threads: {args.threads}", flush=True)
    print(f"Memory limit: {args.memory_limit}", flush=True)

    print("\nLoading selected targets...", flush=True)

    con.execute(
        """
        CREATE TEMP TABLE targets AS
        SELECT
            *,
            ROW_NUMBER() OVER (
                ORDER BY split, user_id, timestamp, parent_asin
            ) AS target_id
        FROM read_parquet(?)
        """,
        [targets],
    )

    print(
        f"Targets loaded: "
        f"{con.execute('SELECT COUNT(*) FROM targets').fetchone()[0]:,}",
        flush=True,
    )

    print("\nFinding interactions for selected users...", flush=True)

    con.execute(
        """
        CREATE TEMP TABLE relevant_interactions AS
        SELECT
            i.user_id,
            i.category,
            i.parent_asin,
            i.product_title,
            i.product_description,
            i.price,
            i.rating,
            i.review_text,
            i.timestamp
        FROM read_parquet(?) AS i
        SEMI JOIN (
            SELECT DISTINCT user_id FROM targets
        ) AS u
        ON i.user_id = u.user_id
        WHERE i.user_id IS NOT NULL
          AND i.parent_asin IS NOT NULL
          AND i.timestamp IS NOT NULL
          AND i.rating BETWEEN 1 AND 5
        """,
        [files],
    )

    print(
        "Relevant interactions loaded in "
        f"{time.perf_counter() - start:.1f}s",
        flush=True,
    )

    print("\nBuilding chronological histories...", flush=True)

    con.execute(
        """
        CREATE TEMP TABLE history_rows AS
        SELECT
            t.target_id,
            t.split,
            t.user_id,
            t.parent_asin AS target_asin,
            t.timestamp AS target_timestamp,
            t.rating AS target_rating,
            i.category AS history_category,
            i.parent_asin AS history_asin,
            i.product_title AS history_title,
            i.rating AS history_rating,
            i.review_text AS history_review,
            i.timestamp AS history_timestamp,
            ROW_NUMBER() OVER (
                PARTITION BY t.target_id
                ORDER BY i.timestamp DESC, i.parent_asin DESC
            ) AS history_rank
        FROM targets t
        JOIN relevant_interactions i
          ON t.user_id = i.user_id
         AND i.timestamp < t.timestamp
         AND i.parent_asin != t.parent_asin
        QUALIFY history_rank <= ?
        """,
        [args.max_history],
    )

    print(
        "History rows built in "
        f"{time.perf_counter() - start:.1f}s",
        flush=True,
    )

    print("\nAttaching target product metadata...", flush=True)

    con.execute(
        """
        CREATE TEMP TABLE target_metadata AS
        SELECT
            t.target_id,
            i.category,
            i.product_title,
            i.product_description,
            i.price
        FROM targets t
        JOIN relevant_interactions i
          ON t.user_id = i.user_id
         AND t.parent_asin = i.parent_asin
         AND t.timestamp = i.timestamp
         AND t.rating = i.rating
        QUALIFY ROW_NUMBER() OVER (
            PARTITION BY t.target_id
            ORDER BY i.category
        ) = 1
        """
    )

    for split in ("train", "validation", "test"):
        output_path = output_dir / f"{split}_histories.parquet"
        sql_path = str(output_path.resolve()).replace("'", "''")

        print(f"\nExporting {split}...", flush=True)

        con.execute(
            f"""
            COPY (
                SELECT
                    t.target_id,
                    t.split,
                    t.user_id,
                    t.parent_asin AS target_asin,
                    t.timestamp AS target_timestamp,
                    t.rating AS target_rating,
                    m.category AS target_category,
                    m.product_title AS target_title,
                    m.product_description AS target_description,
                    m.price AS target_price,
                    h.history_category,
                    h.history_asin,
                    h.history_title,
                    h.history_rating,
                    h.history_review,
                    h.history_timestamp,
                    h.history_rank
                FROM targets t
                LEFT JOIN target_metadata m
                  ON t.target_id = m.target_id
                LEFT JOIN history_rows h
                  ON t.target_id = h.target_id
                WHERE t.split = ?
                ORDER BY t.target_id, h.history_rank DESC
            )
            TO '{sql_path}'
            (FORMAT PARQUET, COMPRESSION ZSTD)
            """,
            [split],
        )

        count = con.execute(
            "SELECT COUNT(DISTINCT target_id) FROM read_parquet(?)",
            [str(output_path.resolve())],
        ).fetchone()[0]

        print(f"Exported {count:,} targets to {output_path}", flush=True)

    con.close()

    print(
        f"\nStage 2 completed in "
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
        "--target-dir",
        default="data/sft/rating/targets",
    )
    parser.add_argument(
        "--output-dir",
        default="data/sft/rating/histories",
    )

    parser.add_argument("--max-history", type=int, default=20)
    parser.add_argument("--threads", type=int, default=64)
    parser.add_argument("--memory-limit", default="150GB")

    main(parser.parse_args())
