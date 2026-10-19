
import argparse
from pathlib import Path

import duckdb


def sql_path(path):
    return str(Path(path).resolve()).replace("'", "''")


def prepare_cpt(input_file, output_dir, min_length):
    input_file = Path(input_file)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    temp_dir = Path("data/duckdb_tmp")
    temp_dir.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()

    try:
        con.execute("SET memory_limit = '96GB'")
        con.execute("SET threads = 16")
        con.execute("SET preserve_insertion_order = false")
        con.execute("SET enable_progress_bar = true")
        con.execute("SET progress_bar_time = 1000")
        con.execute(
            f"SET temp_directory = '{sql_path(temp_dir)}'"
        )

        print("Inspecting training data columns...", flush=True)

        columns = {
            row[0]
            for row in con.execute(
                "DESCRIBE SELECT * FROM read_parquet(?)",
                [str(input_file)],
            ).fetchall()
        }

        print(f"Available columns: {sorted(columns)}")

        if "text" in columns:
            review_column = "text"
        elif "review_text" in columns:
            review_column = "review_text"
        else:
            raise ValueError(
                "No review text column found in training data."
            )

        title_column = (
            "product_title"
            if "product_title" in columns
            else "NULL"
        )

        output_file = output_dir / "cpt_train.jsonl.gz"

        print("\nGenerating CPT documents...", flush=True)

        query = f"""
            COPY (
                SELECT
                    CONCAT(
                        'Category: ', COALESCE(category, 'Unknown'), chr(10),
                        'Product: ', COALESCE({title_column}, 'Unknown'), chr(10),
                        'Rating: ', CAST(rating AS VARCHAR), '/5', chr(10),
                        'Review: ',
                        TRIM(
                            regexp_replace(
                                regexp_replace(
                                    {review_column},
                                    '<br\\s*/?>',
                                    ' ',
                                    'gi'
                                ),
                                '<[^>]+>',
                                ' ',
                                'g'
                            )
                        )
                    ) AS text
                FROM read_parquet('{sql_path(input_file)}')
                WHERE {review_column} IS NOT NULL
                  AND LENGTH(TRIM({review_column})) >= {int(min_length)}
            )
            TO '{sql_path(output_file)}'
            (FORMAT JSON, COMPRESSION GZIP)
        """

        con.execute(query)

        print(f"\nCPT dataset saved: {output_file}")

    finally:
        con.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default="data/splits/train.parquet",
    )
    parser.add_argument(
        "--output",
        default="data/cpt",
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=30,
    )

    args = parser.parse_args()

    prepare_cpt(
        args.input,
        args.output,
        args.min_length,
    )
