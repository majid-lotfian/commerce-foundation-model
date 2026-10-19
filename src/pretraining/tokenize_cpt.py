
import argparse
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
from transformers import AutoTokenizer


def tokenize_cpt(args):
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(input_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading tokenizer: {args.model}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        use_fast=True,
    )

    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer has no EOS token.")

    print(f"Vocabulary size: {len(tokenizer):,}")
    print(f"EOS token ID: {tokenizer.eos_token_id}")

    source = pq.ParquetFile(input_path)

    total_documents = source.metadata.num_rows
    print(f"Input documents: {total_documents:,}", flush=True)

    schema = pa.schema([
        ("input_ids", pa.list_(pa.int32())),
        ("length", pa.int32()),
    ])

    writer = None

    document_count = 0
    token_count = 0
    min_length = float("inf")
    max_length = 0

    # Keep a bounded histogram for approximate percentiles.
    # Values above the cap share the last bucket.
    histogram_cap = 8192
    histogram = [0] * (histogram_cap + 1)

    try:
        writer = pq.ParquetWriter(
            output_path,
            schema,
            compression="zstd",
        )

        with tqdm(
            total=total_documents,
            desc="Tokenizing CPT",
            unit="docs",
        ) as progress:

            for batch in source.iter_batches(
                batch_size=args.batch_size,
                columns=["text"],
            ):
                texts = batch.column(0).to_pylist()

                encoded = tokenizer(
                    texts,
                    add_special_tokens=False,
                    padding=False,
                    truncation=False,
                    return_attention_mask=False,
                )

                rows = []

                for ids in encoded["input_ids"]:
                    # Separate documents with an EOS token.
                    ids = ids + [tokenizer.eos_token_id]
                    length = len(ids)

                    rows.append({
                        "input_ids": ids,
                        "length": length,
                    })

                    document_count += 1
                    token_count += length

                    min_length = min(min_length, length)
                    max_length = max(max_length, length)

                    histogram[
                        min(length, histogram_cap)
                    ] += 1

                table = pa.Table.from_pylist(
                    rows,
                    schema=schema,
                )

                writer.write_table(table)
                progress.update(len(rows))

    finally:
        if writer is not None:
            writer.close()

    def percentile(p):
        if document_count == 0:
            return 0

        target = document_count * p / 100
        cumulative = 0

        for length, count in enumerate(histogram):
            cumulative += count
            if cumulative >= target:
                return length

        return histogram_cap

    if document_count == 0:
        raise RuntimeError("No documents were tokenized.")

    print("\nTokenization completed.")
    print(f"Documents: {document_count:,}")
    print(f"Total tokens: {token_count:,}")
    print(f"Average tokens/document: {token_count / document_count:.1f}")
    print(f"Minimum length: {min_length}")
    print(f"Maximum length: {max_length}")
    print(f"P50 length: {percentile(50)}")
    print(f"P90 length: {percentile(90)}")
    print(f"P95 length: {percentile(95)}")
    print(f"P99 length: {percentile(99)}")
    print(f"Saved: {output_path}")

    estimated_sequences = token_count // args.sequence_length

    print(
        f"Estimated {args.sequence_length}-token sequences: "
        f"{estimated_sequences:,}"
    )
    print(
        "Note: Actual packing count depends on how "
        "remaining tokens are handled."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default="data/cpt/cleaned/cpt_sample.parquet",
    )
    parser.add_argument(
        "--output",
        default="data/cpt/tokenized/cpt_sample.parquet",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-8B",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=2048,
    )

    args = parser.parse_args()
    tokenize_cpt(args)
