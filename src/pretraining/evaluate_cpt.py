
import argparse
import math
import time
from pathlib import Path

import pyarrow.parquet as pq
import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, BitsAndBytesConfig


@torch.inference_mode()
def evaluate(model, sequences, device, label):
    model.eval()

    total_loss = 0.0
    total_tokens = 0
    start = time.perf_counter()

    for ids in tqdm(sequences, desc=label, unit="seq"):
        input_ids = torch.tensor(
            ids, dtype=torch.long, device=device
        ).unsqueeze(0)

        # Causal language modeling predicts token t+1 from token t.
        # We use the same token positions for both models.
        outputs = model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            labels=input_ids,
            use_cache=False,
        )

        predicted_tokens = input_ids.numel() - 1
        total_loss += outputs.loss.item() * predicted_tokens
        total_tokens += predicted_tokens

    mean_loss = total_loss / total_tokens
    perplexity = math.exp(mean_loss)
    elapsed = time.perf_counter() - start

    print(f"\n{label} results:")
    print(f"  Loss: {mean_loss:.6f}")
    print(f"  Perplexity: {perplexity:.4f}")
    print(f"  Tokens evaluated: {total_tokens:,}")
    print(f"  Time: {elapsed:.1f} seconds", flush=True)

    return mean_loss, perplexity


def main(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not available.")

    if not Path(args.adapter).exists():
        raise FileNotFoundError(args.adapter)

    print("GPU:", torch.cuda.get_device_name(0), flush=True)

    print("Loading validation sequences...", flush=True)
    table = pq.read_table(args.input, columns=["input_ids"])
    sequences = table.column("input_ids").to_pylist()

    if args.max_sequences:
        sequences = sequences[:args.max_sequences]

    print(f"Evaluation sequences: {len(sequences):,}", flush=True)

    if not sequences:
        raise RuntimeError("No evaluation sequences found.")

    print("Loading base Qwen3-8B...", flush=True)

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=quantization_config,
        dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="sdpa",
    )

    device = torch.device("cuda:0")

    base_loss, base_ppl = evaluate(
        model, sequences, device, "Base Qwen3-8B"
    )

    print("\nAttaching CPT LoRA adapter...", flush=True)

    model = PeftModel.from_pretrained(
        model,
        args.adapter,
        is_trainable=False,
    )

    cpt_loss, cpt_ppl = evaluate(
        model, sequences, device, "CPT Qwen3-8B"
    )

    print("\n========== COMPARISON ==========")
    print(f"Base loss:        {base_loss:.6f}")
    print(f"CPT loss:         {cpt_loss:.6f}")
    print(f"Base perplexity:  {base_ppl:.4f}")
    print(f"CPT perplexity:   {cpt_ppl:.4f}")

    improvement = (base_loss - cpt_loss) / base_loss * 100
    print(f"Relative loss reduction: {improvement:.2f}%")

    print(
        "Peak allocated GPU memory:",
        f"{torch.cuda.max_memory_allocated() / 1e9:.2f} GB",
    )

    print("Evaluation completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default="data/cpt/eval/cpt_validation_2048.parquet",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-8B",
    )
    parser.add_argument(
        "--adapter",
        default="outputs/cpt_qlora",
    )
    parser.add_argument(
        "--max-sequences",
        type=int,
        default=100,
        help="Number of sequences to evaluate; 0 means all.",
    )

    main(parser.parse_args())
