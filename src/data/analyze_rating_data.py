import argparse
import time

import duckdb


def main(args):
    start = time.perf_counter()
    con = duckdb.connect()
    con.execute(f"SET threads = {args.threads}")

    print("Analyzing customer rating histories...", flush=True)

    query = """
    WITH user_stats AS (
        SELECT
            user_id,
            COUNT(*) AS interactions,
            COUNT(DISTINCT parent_asin) AS unique_products,
            COUNT(DISTINCT category) AS categories,
            COUNT(*) FILTER (WHERE rating >= 4) AS positive,
            COUNT(*) FILTER (WHERE rating <= 2) AS negative,
            COUNT(*) FILTER (WHERE rating = 3) AS neutral
        FROM read_parquet(?)
        WHERE user_id IS NOT NULL
          AND parent_asin IS NOT NULL
          AND rating BETWEEN 1 AND 5
        GROUP BY user_id
    )
    SELECT
        COUNT(*) AS total_users,
        COUNT(*) FILTER (WHERE interactions >= 5) AS users_5,
        COUNT(*) FILTER (WHERE interactions >= 10) AS users_10,
        COUNT(*) FILTER (WHERE interactions >= 20) AS users_20,
        COUNT(*) FILTER (
            WHERE interactions >= 5
              AND positive > 0
              AND negative > 0
        ) AS mixed_rating_users_5,
        COUNT(*) FILTER (
            WHERE interactions >= 10
              AND positive > 0
              AND negative > 0
        ) AS mixed_rating_users_10,
        COUNT(*) FILTER (
            WHERE interactions >= 5
              AND categories >= 2
        ) AS multi_category_users_5,
        MAX(interactions) AS max_interactions,
        AVG(interactions) AS avg_interactions
    FROM user_stats
    """

    result = con.execute(query, [args.input])
    columns = [column[0] for column in result.description]
    values = result.fetchone()

    print("\n========== CUSTOMER STATISTICS ==========")

    for name, value in zip(columns, values):
        if isinstance(value, float):
            print(f"{name}: {value:,.2f}")
        else:
            print(f"{name}: {value:,}")

    print(
        f"\nCompleted in {time.perf_counter() - start:.1f} seconds",
        flush=True,
    )

    con.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="data/splits/train.parquet",
    )
    parser.add_argument("--threads", type=int, default=8)
    main(parser.parse_args())
