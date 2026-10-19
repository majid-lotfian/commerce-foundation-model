
import argparse
import json
import time
from collections import Counter
from pathlib import Path

import duckdb
from tqdm import tqdm


SYSTEM_PROMPT = (
    "You are a personalized e-commerce rating prediction model. "
    "Predict the rating this customer would give the target product "
    "based on their previous product reviews and ratings. "
    "Return only one integer from 1 to 5."
)


def clean(value, max_chars=300):
    if value is None:
        return "Unknown"

    text = " ".join(str(value).split())
    return text[:max_chars] if text else "Unknown"


def build_example(rows):
    target = rows[0]

    history_lines = []

    for index, row in enumerate(rows, 1):
        history_lines.append(
            f"{index}. "
            f"Category: {clean(row['history_category'], 60)} | "
            f"Product: {clean(row['history_title'], 150)} | "
            f"Rating: {int(row['history_rating'])} stars | "
            f"Review: {clean(row['history_review'], 200)}"
        )

    price = target["target_price"]
    price_text = (
        f"{price:.2f}" if price is not None else "Unknown"
    )

    user_prompt = (
        "Customer's previous interactions "
        "(oldest to newest):\n\n"
        + "\n".join(history_lines)
        + "\n\nTarget product:\n"
        + f"Category: {clean(target['target_category'], 60)}\n"
        + f"Title: {clean(target['target_title'], 200)}\n"
        + f"Description: {clean(target['target_description'], 400)}\n"
        + f"Price: {price_text}\n\n"
        + "Predict the customer's rating for this product."
    )

    rating = int(target["target_rating"])

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": str(rating)},
        ],
        "target_rating": rating,
        "target_asin": target["target_asin"],
        "target_timestamp": target["target_timestamp"],
        "history_length": len(rows),
    }


def process_split(con, split, args):
    input_path = (
        Path(args.input_dir) / f"{split}_histories.parquet"
    )
    output_path = (
        Path(args.output_dir) / f"{split}.jsonl"
    )

    if not input_path.exists():
        raise FileNotFoundError(input_path)

    query = """
        SELECT *
        FROM read_parquet(?)
        ORDER BY target_id, history_rank DESC
    """

    cursor = con.execute(query, [str(input_path)])

    columns = [column[0] for column in cursor.description]

    current_target = None
    rows = []

    saved = 0
    skipped = 0
    ratings = Counter()

    def write_target(file):
        nonlocal saved, skipped

        if len(rows) < args.min_history:
            skipped += 1
            return

        example = build_example(rows)

        file.write(
            json.dumps(example, ensure_ascii=False) + "\n"
        )

        saved += 1
        ratings[example["target_rating"]] += 1

    with open(output_path, "w", encoding="utf-8") as file:
        with tqdm(
            desc=f"Formatting {split}",
            unit="history row",
        ) as progress:
            while True:
                batch = cursor.fetchmany(10000)

                if not batch:
                    break

                for values in batch:
                    progress.update(1)

                    row = dict(zip(columns, values))
                    target_id = row["target_id"]

                    if current_target is None:
                        current_target = target_id

                    if target_id != current_target:
                        write_target(file)
                        rows = []
                        current_target = target_id

                    if row["history_asin"] is not None:
                        rows.append(row)

            if rows:
                write_target(file)

    print(f"\n{split.upper()}")
    print(f"  Saved: {saved:,}")
    print(f"  Skipped: {skipped:,}")
    print(f"  Rating distribution: {dict(sorted(ratings.items()))}")
    print(f"  Output: {output_path}", flush=True)


def main(args):
    start = time.perf_counter()

    Path(args.output_dir).mkdir(
        parents=True, exist_ok=True
    )

    con = duckdb.connect()
    con.execute(f"SET threads = {args.threads}")

    for split in ("train", "validation", "test"):
        process_split(con, split, args)

    con.close()

    print(
        f"\nStage 3 completed in "
        f"{time.perf_counter() - start:.1f} seconds"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-dir",
        default="data/sft/rating/histories",
    )
    parser.add_argument(
        "--output-dir",
        default="data/sft/rating",
    )
    parser.add_argument("--min-history", type=int, default=5)
    parser.add_argument("--threads", type=int, default=16)

    main(parser.parse_args())
