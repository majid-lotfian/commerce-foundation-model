
import argparse
import json
from pathlib import Path

import duckdb
from tqdm import tqdm


SYSTEM_PROMPT = (
    "You are an e-commerce recommendation model. "
    "Predict the next product the user will interact with. "
    "Return only the product identifier (parent_asin)."
)


def build_examples(rows, min_history, max_history):
    examples = []
    history = []

    for parent_asin, product_title, timestamp in rows:
        if parent_asin is None:
            continue

        if len(history) >= min_history:
            recent_history = history[-max_history:]

            history_text = "\n".join(
                f"{i + 1}. {title} [ASIN: {asin}]"
                for i, (asin, title) in enumerate(recent_history)
            )

            examples.append({
                "messages": [
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": (
                            "Predict the next product interaction.\n\n"
                            f"Previous interactions:\n{history_text}"
                        ),
                    },
                    {
                        "role": "assistant",
                        "content": parent_asin,
                    },
                ],
                "target_asin": parent_asin,
                "history_length": len(recent_history),
                "target_timestamp": timestamp,
            })

        history.append((
            parent_asin,
            product_title or "Unknown product",
        ))

    return examples


def main(args):
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()

    print("Reading chronological user interactions...", flush=True)

    query = """
        SELECT
            user_id,
            parent_asin,
            product_title,
            timestamp
        FROM read_parquet(?)
        WHERE user_id IS NOT NULL
          AND parent_asin IS NOT NULL
          AND timestamp IS NOT NULL
        ORDER BY user_id, timestamp, parent_asin
    """

    cursor = con.execute(query, [args.input])

    total_examples = 0
    total_users = 0
    current_user = None
    user_rows = []

    with open(output_path, "w", encoding="utf-8") as output:
        with tqdm(
            total=args.max_examples,
            desc="Building SFT examples",
            unit="example",
        ) as progress:

            while True:
                batch = cursor.fetchmany(10000)

                if not batch:
                    break

                for user_id, asin, title, timestamp in batch:
                    if current_user is None:
                        current_user = user_id

                    if user_id != current_user:
                        examples = build_examples(
                            user_rows,
                            args.min_history,
                            args.max_history,
                        )

                        for example in examples:
                            if total_examples >= args.max_examples:
                                break

                            output.write(
                                json.dumps(example, ensure_ascii=False) + "\n"
                            )

                            total_examples += 1
                            progress.update(1)

                        total_users += 1

                        current_user = user_id
                        user_rows = []

                        if total_examples >= args.max_examples:
                            break

                    user_rows.append((asin, title, timestamp))

                if total_examples >= args.max_examples:
                    break

            if user_rows and total_examples < args.max_examples:
                examples = build_examples(
                    user_rows,
                    args.min_history,
                    args.max_history,
                )

                for example in examples:
                    if total_examples >= args.max_examples:
                        break

                    output.write(
                        json.dumps(example, ensure_ascii=False) + "\n"
                    )
                    total_examples += 1
                    progress.update(1)

    con.close()

    print("\nDataset preparation completed.")
    print(f"Examples created: {total_examples:,}")
    print(f"Users processed: {total_users:,}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default="data/splits/train.parquet",
    )
    parser.add_argument(
        "--output",
        default="data/sft/recommendation/train_sample.jsonl",
    )
    parser.add_argument(
        "--min-history",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--max-history",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=10000,
    )

    main(parser.parse_args())
