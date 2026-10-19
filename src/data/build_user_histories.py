
import argparse
from pathlib import Path

import duckdb


def sql_path(path):
    return str(path.resolve()).replace("'", "''")


def build_user_histories(categories, input_dir, output_dir):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if "all" in categories:
        files = sorted(input_dir.glob("*.parquet"))
    else:
        files = [
            input_dir / f"{category}.parquet"
            for category in categories
        ]

    if not files:
        raise RuntimeError("No interaction files found.")

    
    print(f"\nProcessing {len(files)} categories:", flush=True)

    for i, path in enumerate(files, start=1):
        print(f"  [{i}/{len(files)}] {path.stem}", flush=True)
    
    
    for path in files:
        if not path.exists():
            raise FileNotFoundError(path)

    destination = output_dir / "user_histories.parquet"

    temp_dir = Path("data/duckdb_tmp")
    temp_dir.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()

    con.execute("SET enable_progress_bar = true")
    con.execute("SET progress_bar_time = 1000")

    try:
        con.execute("SET memory_limit = '8GB'")
        con.execute("SET threads = 2")
        con.execute("SET preserve_insertion_order = false")
        con.execute(
            f"SET temp_directory = '{sql_path(temp_dir)}'"
        )

        file_list = ", ".join(
            f"'{sql_path(path)}'" for path in files
        )

        output_sql = sql_path(destination)

        print(
            "\nBuilding unified user histories across all categories...",
            flush=True,
        )

        con.execute(
            f"""
            COPY (
                SELECT
                    user_id,
                    category,
                    parent_asin,
                    product_title,
                    product_description,
                    price,
                    rating,
                    verified_purchase,
                    review_title,
                    review_text,
                    timestamp,

                    ROW_NUMBER() OVER (
                        PARTITION BY user_id
                        ORDER BY
                            timestamp,
                            category,
                            parent_asin
                    ) AS interaction_number,

                    COUNT(*) OVER (
                        PARTITION BY user_id
                    ) AS history_length

                FROM read_parquet([{file_list}])

                WHERE user_id IS NOT NULL
                  AND parent_asin IS NOT NULL
                  AND timestamp IS NOT NULL
            )
            TO '{output_sql}'
            (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )

        stats = con.execute(
            """
            SELECT
                COUNT(*) AS interactions,
                COUNT(DISTINCT user_id) AS users,
                COUNT(DISTINCT category) AS categories,
                COUNT(*) FILTER (WHERE history_length >= 5)
                    AS interactions_from_users_with_5_plus
            FROM read_parquet(?)
            """,
            [str(destination)],
        ).fetchone()

        print("\nUnified user histories created")
        print(f"Categories: {stats[2]}")
        print(f"Interactions: {stats[0]:,}")
        print(f"Unique users: {stats[1]:,}")
        print(
            "Interactions from users with 5+ reviews: "
            f"{stats[3]:,}"
        )
        print(f"Saved: {destination}")

    finally:
        con.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--categories",
        nargs="+",
        required=True,
    )
    parser.add_argument(
        "--input",
        default="data/processed/interactions",
    )
    parser.add_argument(
        "--output",
        default="data/processed/user_histories",
    )

    args = parser.parse_args()

    build_user_histories(
        args.categories,
        args.input,
        args.output,
    )
