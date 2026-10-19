
import argparse
import gzip
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

def read_jsonl_gz(path):
    """Stream compressed JSONL records with a progress bar."""
    path = Path(path)

    with open(path, "rb") as raw:
        with tqdm(
            total=path.stat().st_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=path.name,
        ) as progress:
            with gzip.GzipFile(fileobj=raw, mode="rb") as f:
                previous_position = 0

                for line in f:
                    if line.strip():
                        yield json.loads(line)

                    current_position = raw.tell()
                    progress.update(
                        current_position - previous_position
                    )
                    previous_position = current_position


REVIEW_SCHEMA = pa.schema([
    ("category", pa.string()),
    ("user_id", pa.string()),
    ("parent_asin", pa.string()),
    ("rating", pa.float32()),
    ("timestamp", pa.int64()),
    ("verified_purchase", pa.bool_()),
    ("title", pa.string()),
    ("text", pa.string()),
])

PRODUCT_SCHEMA = pa.schema([
    ("category", pa.string()),
    ("parent_asin", pa.string()),
    ("title", pa.string()),
    ("description", pa.string()),
    ("features", pa.string()),
    ("price", pa.float64()),
    ("store", pa.string()),
])


def normalize_review(record, category):
    return {
        "category": category,
        "user_id": record.get("user_id"),
        "parent_asin": record.get("parent_asin"),
        "rating": record.get("rating"),
        "timestamp": record.get("timestamp"),
        "verified_purchase": record.get("verified_purchase"),
        "title": record.get("title"),
        "text": record.get("text"),
    }


def normalize_product(record, category):
    description = record.get("description") or []
    features = record.get("features") or []

    price = record.get("price")
    try:
        price = float(price) if price is not None else None
    except (ValueError, TypeError):
        price = None

    return {
        "category": category,
        "parent_asin": record.get("parent_asin"),
        "title": record.get("title"),
        "description": " ".join(map(str, description)),
        "features": " ".join(map(str, features)),
        "price": price,
        "store": record.get("store"),
    }


def process_file(source, destination, schema, normalize, category, batch_size):
    destination.parent.mkdir(parents=True, exist_ok=True)

    writer = None
    batch = []
    total = 0

    try:
        for record in read_jsonl_gz(source):
            batch.append(normalize(record, category))

            if len(batch) >= batch_size:
                table = pa.Table.from_pylist(batch, schema=schema)

                if writer is None:
                    writer = pq.ParquetWriter(
                        destination,
                        schema,
                        compression="zstd",
                    )

                writer.write_table(table)
                total += len(batch)
                batch.clear()

        if batch:
            table = pa.Table.from_pylist(batch, schema=schema)

            if writer is None:
                writer = pq.ParquetWriter(
                    destination,
                    schema,
                    compression="zstd",
                )

            writer.write_table(table)
            total += len(batch)

    finally:
        if writer is not None:
            writer.close()

    print(f"Processed {total:,} records -> {destination}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--categories", nargs="+", required=True)
    parser.add_argument("--input", default="src/data/raw")
    parser.add_argument("--output", default="data/processed")
    parser.add_argument("--batch-size", type=int, default=10000)

    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if "all" in args.categories:
        categories = sorted(
            p.name.removesuffix(".jsonl.gz")
            for p in (input_dir / "reviews").glob("*.jsonl.gz")
        )
    else:
        categories = args.categories

    for category in categories:
        review_source = (
            input_dir / "reviews" / f"{category}.jsonl.gz"
        )
        product_source = (
            input_dir / "metadata" / f"meta_{category}.jsonl.gz"
        )

        for source in [review_source, product_source]:
            if not source.exists():
                raise FileNotFoundError(source)

        print(f"\nProcessing category: {category}")

        process_file(
            review_source,
            output_dir / "reviews" / f"{category}.parquet",
            REVIEW_SCHEMA,
            normalize_review,
            category,
            args.batch_size,
        )

        process_file(
            product_source,
            output_dir / "products" / f"{category}.parquet",
            PRODUCT_SCHEMA,
            normalize_product,
            category,
            args.batch_size,
        )


if __name__ == "__main__":
    main()
