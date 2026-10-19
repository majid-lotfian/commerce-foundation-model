
import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer


def load_examples(path, max_examples=None):
    examples = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            examples.append(json.loads(line))

            if max_examples and len(examples) >= max_examples:
                break

    return examples


def analyze_split(examples, tokenizer, split, batch_size):
    lengths = []
    ratings = Counter()
    target_keys = set()
    duplicates = 0

    for start in tqdm(
        range(0, len(examples), batch_size),
        desc=f"Analyzing {split}",
    ):
        batch = examples[start:start + batch_size]

        texts = [
            tokenizer.apply_chat_template(
                example["messages"],
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=False,
            )
            for example in batch
        ]

        encoded = tokenizer(
            texts,
            add_special_tokens=False,
            padding=False,
            truncation=False,
            return_attention_mask=False,
        )

        lengths.extend(len(ids) for ids in encoded["input_ids"])

        for example in batch:
            ratings[int(example["target_rating"])] += 1

            key = (
                example["target_asin"],
                example["target_timestamp"],
                example["messages"][1]["content"],
            )

            if key in target_keys:
                duplicates += 1

            target_keys.add(key)

    arr = np.array(lengths)

    stats = {
        "examples": len(examples),
        "mean_tokens": round(float(arr.mean()), 1),
        "median_tokens": int(np.median(arr)),
        "p90_tokens": int(np.percentile(arr, 90)),
        "p95_tokens": int(np.percentile(arr, 95)),
        "p99_tokens": int(np.percentile(arr, 99)),
        "max_tokens": int(arr.max()),
        "over_2048": int((arr > 2048).sum()),
        "over_3072": int((arr > 3072).sum()),
        "over_4096": int((arr > 4096).sum()),
        "rating_distribution": dict(sorted(ratings.items())),
        "duplicate_targets_within_split": duplicates,
    }

    return stats, target_keys


def main(args):
    start = time.perf_counter()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
    )

    results = {}
    keys_by_split = {}

    for split in ["train", "validation", "test"]:
        path = Path(args.data_dir) / f"{split}.jsonl"

        examples = load_examples(path, args.max_examples)

        stats, keys = analyze_split(
            examples,
            tokenizer,
            split,
            args.batch_size,
        )

        results[split] = stats
        keys_by_split[split] = keys

        print(f"\n{split.upper()}")
        for key, value in stats.items():
            print(f"  {key}: {value}")

    print("\nCROSS-SPLIT DUPLICATES")

    for a, b in [
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ]:
        overlap = len(keys_by_split[a] & keys_by_split[b])
        print(f"  {a} vs {b}: {overlap}")

    results["cross_split_duplicates"] = {
        f"{a}_vs_{b}": len(keys_by_split[a] & keys_by_split[b])
        for a, b in [
            ("train", "validation"),
            ("train", "test"),
            ("validation", "test"),
        ]
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    with open(output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved report: {output}")
    print(f"Completed in {time.perf_counter() - start:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-8B",
    )
    parser.add_argument(
        "--data-dir",
        default="data/sft/rating",
    )
    parser.add_argument(
        "--output",
        default="data/sft/rating/token_analysis.json",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
    )

    main(parser.parse_args())
