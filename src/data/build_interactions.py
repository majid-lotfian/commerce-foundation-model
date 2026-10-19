
import argparse
from pathlib import Path

import duckdb


def discover_categories(input_dir):
    """Find categories with both reviews and product metadata."""
    reviews_dir = input_dir / "reviews"
    products_dir = input_dir / "products"

    if not reviews_dir.exists() or not products_dir.exists():
        raise FileNotFoundError(
            "Reviews or products directory does not exist."
        )

    review_categories = {
        path.stem for path in reviews_dir.glob("*.parquet")
    }

    product_categories = {
        path.stem for path in products_dir.glob("*.parquet")
    }

    return sorted(review_categories & product_categories)


def sql_path(path):
    """Escape a filesystem path for use in DuckDB SQL."""
    return str(path.resolve()).replace("'", "''")


def build_interactions(
    category,
    input_dir,
    output_dir,
    temp_dir,
    overwrite=False,
):
    reviews = input_dir / "reviews" / f"{category}.parquet"
    products = input_dir / "products" / f"{category}.parquet"
    destination = output_dir / f"{category}.parquet"
    temporary_output = output_dir / f"{category}.partial.parquet"

    for path in (reviews, products):
        if not path.exists():
            raise FileNotFoundError(path)

    if destination.exists() and not overwrite:
        print(f"Skipping {category}: output already exists.")
        return

    # Remove an incomplete output from a previous failed run.
    if temporary_output.exists():
        temporary_output.unlink()

    print(f"\nProcessing category: {category}", flush=True)

    con = duckdb.connect()

    try:

        con.execute("SET enable_progress_bar = true")
        con.execute("SET progress_bar_time = 1000")
        con.execute("SET memory_limit = '8GB'")
        con.execute("SET threads = 2")
        con.execute("SET preserve_insertion_order = false")

        temp_sql = sql_path(temp_dir)
        con.execute(f"SET temp_directory = '{temp_sql}'")

        reviews_sql = sql_path(reviews)
        products_sql = sql_path(products)
        output_sql = sql_path(temporary_output)
        
        print(
            f"\nJoining reviews and products for {category}...",
            flush=True,
        )

        con.execute(
            f"""
            COPY (
                SELECT
                    r.category,
                    r.user_id,
                    r.parent_asin,
                    p.title AS product_title,
                    p.description AS product_description,
                    p.price,
                    r.rating,
                    r.verified_purchase,
                    r.title AS review_title,
                    r.text AS review_text,
                    r.timestamp,
                    ROW_NUMBER() OVER (
                        PARTITION BY r.user_id
                        ORDER BY
                            r.timestamp,
                            r.parent_asin,
                            r.title,
                            r.text
                    ) AS interaction_number
                FROM read_parquet('{reviews_sql}') AS r
                LEFT JOIN read_parquet('{products_sql}') AS p
                    ON r.category = p.category
                   AND r.parent_asin = p.parent_asin
                WHERE r.user_id IS NOT NULL
                  AND r.parent_asin IS NOT NULL
                  AND r.timestamp IS NOT NULL
            )
            TO '{output_sql}'
            (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )

        count = con.execute(
            "SELECT COUNT(*) FROM read_parquet(?)",
            [str(temporary_output)],
        ).fetchone()[0]

        # Publish only after DuckDB has finished successfully.
        temporary_output.replace(destination)

        print(f"Interactions: {count:,}")
        print(f"Saved: {destination}")

    finally:
        con.close()


def main():
    parser = argparse.ArgumentParser(
        description="Build Amazon review-product interactions."
    )

    parser.add_argument(
        "--categories",
        nargs="+",
        required=True,
        help="Category names or 'all'.",
    )

    parser.add_argument(
        "--input",
        default="data/processed",
    )

    parser.add_argument(
        "--output",
        default="data/processed/interactions",
    )

    parser.add_argument(
        "--temp-dir",
        default="data/duckdb_tmp",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild existing output files.",
    )

    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    temp_dir = Path(args.temp_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)

    if "all" in args.categories:
        categories = discover_categories(input_dir)
    else:
        categories = list(dict.fromkeys(args.categories))

    if not categories:
        raise RuntimeError("No categories found to process.")

    print(f"Selected categories: {', '.join(categories)}")

    for category in categories:
        build_interactions(
            category=category,
            input_dir=input_dir,
            output_dir=output_dir,
            temp_dir=temp_dir,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
