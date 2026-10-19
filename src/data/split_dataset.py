
import argparse
from pathlib import Path

import duckdb


def sql_path(path):
    return str(path.resolve()).replace("'", "''")


def split_dataset(input_file, output_dir, train_ratio, val_ratio):
    input_file = Path(input_file)
    output_dir = Path(output_dir)

    if not input_file.exists():
        raise FileNotFoundError(input_file)

    if not (0 < train_ratio < 1 and 0 < val_ratio < 1):
        raise ValueError("Split ratios must be between 0 and 1.")

    if train_ratio + val_ratio >= 1:
        raise ValueError("Train + validation ratios must be below 1.")

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

        source = sql_path(input_file)

        print("Calculating chronological split boundaries...", flush=True)

        # Approximate quantiles avoid sorting the full dataset
        # solely to identify timestamp boundaries.
        train_end, val_end = con.execute(
            f"""
            SELECT
                approx_quantile(timestamp, {train_ratio}),
                approx_quantile(timestamp, {train_ratio + val_ratio})
            FROM read_parquet('{source}')
            """
        ).fetchone()

        if train_end is None or val_end is None:
            raise RuntimeError("No valid timestamps found.")

        if train_end >= val_end:
            raise RuntimeError(
                "Split boundaries overlap. Inspect timestamp distribution."
            )

        print(f"Train ends at: {train_end}")
        print(f"Validation ends at: {val_end}")

        splits = {
            "train": f"timestamp <= {train_end}",
            "validation": (
                f"timestamp > {train_end} AND timestamp <= {val_end}"
            ),
            "test": f"timestamp > {val_end}",
        }

        for split_name, condition in splits.items():
            destination = output_dir / f"{split_name}.parquet"

            print(f"\nWriting {split_name} split...", flush=True)

            con.execute(
                f"""
                COPY (
                    SELECT *
                    FROM read_parquet('{source}')
                    WHERE {condition}
                )
                TO '{sql_path(destination)}'
                (FORMAT PARQUET, COMPRESSION ZSTD)
                """
            )

            count = con.execute(
                "SELECT COUNT(*) FROM read_parquet(?)",
                [str(destination)],
            ).fetchone()[0]

            print(f"{split_name}: {count:,} interactions", flush=True)

        print("\nDataset splitting completed.")

    finally:
        con.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default="data/processed/user_histories/user_histories.parquet",
    )
    parser.add_argument(
        "--output",
        default="data/splits",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
    )

    args = parser.parse_args()

    split_dataset(
        args.input,
        args.output,
        args.train_ratio,
        args.val_ratio,
    )
