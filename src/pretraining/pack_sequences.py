
import argparse
from collections import deque
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


def pack_sequences(args):
    source = Path(args.input)
    destination = Path(args.output)

    if not source.exists():
        raise FileNotFoundError(source)

    if args.sequence_length < 2:
        raise ValueError("Sequence length must be at least 2.")

    destination.parent.mkdir(parents=True, exist_ok=True)

    reader = pq.ParquetFile(source)
    total_documents = reader.metadata.num_rows

    schema = pa.schema([
        ("input_ids", pa.list_(pa.int32())),
    ])

    buffer = deque()
    packed_batch = []

    documents = 0
    total_tokens = 0
    sequences = 0

    writer = None

    def write_batch():
        nonlocal packed_batch, writer

        if not packed_batch:
            return

        table = pa.Table.from_pylist(
            [{"input_ids": seq} for seq in packed_batch],
            schema=schema,
        )

        writer.write_table(table)
        packed_batch.clear()

    print(f"Input: {source}")
    print(f"Output: {destination}")
    print(f"Sequence length: {args.sequence_length}")
    print(f"Documents: {total_documents:,}", flush=True)

    try:
        writer = pq.ParquetWriter(
            destination,
            schema,
            compression="zstd",
        )

        with tqdm(
            total=total_documents,
            desc="Packing sequences",
            unit="docs",
        ) as progress:

            for batch in reader.iter_batches(
                batch_size=args.batch_size,
                columns=["input_ids"],
            ):
                for ids in batch.column(0).to_pylist():
                    buffer.extend(ids)

                    documents += 1
                    total_tokens += len(ids)

                    while len(buffer) >= args.sequence_length:
                        sequence = [
                            buffer.popleft()
                            for _ in range(args.sequence_length)
                        ]

                        packed_batch.append(sequence)
                        sequences += 1

                        if len(packed_batch) >= args.write_batch_size:
                            write_batch()

                progress.update(batch.num_rows)

        remaining_tokens = len(buffer)

        if remaining_tokens and not args.drop_remainder:
            # Pad only the final sequence.
            sequence = list(buffer)
            sequence.extend(
                [args.pad_token_id] *
                (args.sequence_length - len(sequence))
            )
            packed_batch.append(sequence)
            sequences += 1

        write_batch()

    finally:
        if writer is not None:
            writer.close()

    print("\nPacking completed.")
    print(f"Documents processed: {documents:,}")
    print(f"Input tokens: {total_tokens:,}")
    print(f"Packed sequences: {sequences:,}")
    print(f"Remaining tokens: {remaining_tokens:,}")
    print(f"Saved: {destination}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default="data/cpt/tokenized/cpt_sample.parquet",
    )
    parser.add_argument(
        "--output",
        default="data/cpt/packed/cpt_sample_2048.parquet",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=2048,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--write-batch-size",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--pad-token-id",
        type=int,
        default=151643,
    )
    parser.add_argument(
        "--drop-remainder",
        action="store_true",
    )

    args = parser.parse_args()
    pack_sequences(args)
