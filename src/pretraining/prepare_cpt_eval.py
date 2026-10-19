
import argparse
import math
from collections import deque
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
from transformers import AutoTokenizer


def main(args):
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    print("Loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        use_fast=True,
    )

    print("Sampling validation reviews...", flush=True)
    con = duckdb.connect()
    con.execute("SET threads TO 8")
    con.execute("SET memory_limit TO '32GB'")

    # Use the same text construction as CPT training.
    # Exclude missing reviews and very short texts.
    query = """
    SELECT
        'Category: ' || COALESCE(category, '') || chr(10) ||
        'Product: ' || COALESCE(product_title, '') || chr(10) ||
        'Rating: ' || COALESCE(CAST(rating AS VARCHAR), '') || chr(10) ||
        'Review: ' ||
        regexp_replace(
            COALESCE(review_text, ''),
            '<[^>]*>',
            ' ',
            'g'
        ) AS document
    FROM read_parquet(?)
    WHERE review_text IS NOT NULL
    AND length(trim(review_text)) >= 50
    ORDER BY hash(
        user_id,
        parent_asin,
        timestamp,
        review_text
    )
    LIMIT ?
    """

    rows = con.execute(
        query,
        [args.input, args.num_documents],
    ).fetchall()
    con.close()

    print(f"Selected {len(rows):,} validation documents.", flush=True)

    if not rows:
        raise RuntimeError("No validation documents found.")

    buffer = deque()
    packed = []
    total_tokens = 0

    print("Tokenizing and packing...", flush=True)

    for (document,) in tqdm(rows, desc="Validation documents"):
        ids = tokenizer(
            document,
            add_special_tokens=False,
        )["input_ids"]

        ids.append(tokenizer.eos_token_id)
        total_tokens += len(ids)
        buffer.extend(ids)

        while len(buffer) >= args.sequence_length:
            packed.append([
                buffer.popleft()
                for _ in range(args.sequence_length)
            ])

    schema = pa.schema([
        ("input_ids", pa.list_(pa.int32())),
    ])

    table = pa.Table.from_arrays(
        [
            pa.array(packed, type=pa.list_(pa.int32()))
        ],
        schema=schema,
    )

    pq.write_table(
        table,
        output,
        compression="zstd",
    )

    print("\nEvaluation dataset ready.")
    print(f"Documents: {len(rows):,}")
    print(f"Tokens: {total_tokens:,}")
    print(f"Packed sequences: {len(packed):,}")
    print(f"Unused tokens: {len(buffer):,}")
    print(f"Saved: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default="data/splits/validation.parquet",
    )
    parser.add_argument(
        "--output",
        default="data/cpt/eval/cpt_validation_2048.parquet",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-8B",
    )
    parser.add_argument(
        "--num-documents",
        type=int,
        default=10000,
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=2048,
    )

    main(parser.parse_args())
