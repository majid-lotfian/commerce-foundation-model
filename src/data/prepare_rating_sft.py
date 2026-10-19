
import argparse
import hashlib
import heapq
import json
import time
from collections import deque
from pathlib import Path

import duckdb
from tqdm import tqdm


SYSTEM_PROMPT = (
    "You are a personalized product rating prediction model. "
    "Predict the integer rating (1 to 5 stars) that this customer "
    "would give the target product based on their previous interactions. "
    "Return only one integer from 1 to 5."
)


def stable_hash(value):
    return int.from_bytes(
        hashlib.blake2b(
            value.encode("utf-8"), digest_size=8
        ).digest(),
        "big",
    )


def clean(value, max_chars=300):
    if value is None:
        return "Unknown"

    return " ".join(str(value).split())[:max_chars]


def format_example(history, target, max_history):
    recent = list(history)[-max_history:]

    history_lines = []

    for i, row in enumerate(recent, 1):
        history_lines.append(
            f"{i}. [{clean(row['category'], 60)}] "
            f"{clean(row['product_title'], 150)} "
            f"| Rating: {row['rating']} stars "
            f"| Review: {clean(row['review_text'], 200)}"
        )

    user_prompt = (
        "Customer's previous interactions "
        "(oldest to newest):\n"
        + "\n".join(history_lines)
        + "\n\nTarget product:\n"
        + f"Category: {clean(target['category'], 60)}\n"
        + f"Title: {clean(target['product_title'], 200)}\n"
        + f"Description: {clean(target['product_description'], 400)}\n"
        + f"Price: {target['price'] if target['price'] is not None else 'Unknown'}\n\n"
        + "Predict the customer's rating for this product."
    )

    rating = int(target["rating"])

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": str(rating)},
        ],
        "target_rating": rating,
        "target_asin": target["parent_asin"],
        "target_timestamp": target["timestamp"],
        "history_length": len(recent),
    }


def get_rows(con, files):
    query = """
        SELECT
            user_id,
            category,
            parent_asin,
            product_title,
            product_description,
            price,
            rating,
            review_text,
            timestamp,
            split
        FROM (
            SELECT *,
                CASE
                    WHEN filename LIKE '%train.parquet' THEN 'train'
                    WHEN filename LIKE '%validation.parquet' THEN 'validation'
                    WHEN filename LIKE '%test.parquet' THEN 'test'
                END AS split
            FROM read_parquet(?, filename=true)
        )
        WHERE user_id IS NOT NULL
          AND parent_asin IS NOT NULL
          AND rating BETWEEN 1 AND 5
          AND timestamp IS NOT NULL
        ORDER BY user_id, timestamp, parent_asin
    """

    cursor = con.execute(query, [files])

    columns = [d[0] for d in cursor.description]

    while True:
        batch = cursor.fetchmany(10000)

        if not batch:
            break

        for values in batch:
            yield dict(zip(columns, values))


def main(args):
    start = time.perf_counter()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = [
        str(Path(args.train).resolve()),
        str(Path(args.validation).resolve()),
        str(Path(args.test).resolve()),
    ]

    limits = {
        "train": args.train_examples,
        "validation": args.validation_examples,
        "test": args.test_examples,
    }

    # Keep a bounded deterministic sample for each split.
    samples = {split: [] for split in limits}
    eligible = {split: 0 for split in limits}

    con = duckdb.connect()
    con.execute(f"SET threads = {args.threads}")

    print("Reading chronologically ordered interactions...", flush=True)

    current_user = None
    history = deque(maxlen=args.max_history)

    # Tracks ASINs in the current user's history.
    seen_asins = set()

    with tqdm(desc="Processing interactions", unit="row") as progress:
        for row in get_rows(con, files):
            progress.update(1)

            user_id = row["user_id"]

            if user_id != current_user:
                current_user = user_id
                history.clear()
                seen_asins.clear()

            split = row["split"]

            if split not in limits:
                raise ValueError(f"Unknown split: {split}")

            asin = row["parent_asin"]

            if (
                len(history) >= args.min_history
                and asin not in seen_asins
            ):
                eligible[split] += 1

                key = (
                    f"{split}|{user_id}|{asin}|"
                    f"{row['timestamp']}|{row['rating']}"
                )
                score = stable_hash(key)

                example = format_example(
                    history, row, args.max_history
                )

                heap = samples[split]
                item = (-score, eligible[split], example)

                if len(heap) < limits[split]:
                    heapq.heappush(heap, item)
                elif score < -heap[0][0]:
                    heapq.heapreplace(heap, item)

            history.append(row)
            seen_asins.add(asin)

    print("\nWriting datasets...", flush=True)

    for split, heap in samples.items():
        path = output_dir / f"{split}.jsonl"
        counts = {rating: 0 for rating in range(1, 6)}

        ordered = sorted(heap, key=lambda x: -x[0])

        with open(path, "w", encoding="utf-8") as f:
            for _, _, example in ordered:
                f.write(json.dumps(example, ensure_ascii=False) + "\n")
                counts[example["target_rating"]] += 1

        print(f"\n{split.upper()}")
        print(f"  Eligible examples: {eligible[split]:,}")
        print(f"  Saved examples: {len(heap):,}")
        print(f"  Rating distribution: {counts}")
        print(f"  Output: {path}")

    print(
        f"\nCompleted in {time.perf_counter() - start:.1f} seconds",
        flush=True,
    )

    con.close()


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
        default="data/sft/rating",
    )

    parser.add_argument("--min-history", type=int, default=5)
    parser.add_argument("--max-history", type=int, default=20)

    parser.add_argument("--train-examples", type=int, default=100000)
    parser.add_argument("--validation-examples", type=int, default=10000)
    parser.add_argument("--test-examples", type=int, default=10000)

    parser.add_argument("--threads", type=int, default=8)

    main(parser.parse_args())
