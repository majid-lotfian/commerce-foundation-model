
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
)
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    rating_token_ids = [
        tokenizer.encode(str(r), add_special_tokens=False)[0]
        for r in range(1, 6)
    ]

    with open(args.test_file, encoding="utf-8") as f:
        examples = [json.loads(line) for line in f]

    if args.limit:
        examples = examples[:args.limit]

    print(f"Evaluating {len(examples):,} examples", flush=True)

    model_kwargs = {
        "dtype": torch.bfloat16,
        "device_map": {"": 0},
        "attn_implementation": "sdpa",
    }

    if args.precision == "qlora":
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        **model_kwargs,
    )

    if args.adapter:
        model = PeftModel.from_pretrained(
            model,
            args.adapter,
        )

    model.eval()

    predictions = []
    targets = []
    records = []

    with torch.inference_mode():
        for example in tqdm(examples, desc="Evaluating"):
            prompt_ids = tokenizer.apply_chat_template(
                example["messages"][:-1],
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
            )["input_ids"]

            input_ids = torch.tensor(
                [prompt_ids],
                dtype=torch.long,
                device="cuda",
            )

            outputs = model(input_ids=input_ids)

            rating_logits = outputs.logits[0, -1, rating_token_ids]
            probabilities = torch.softmax(
                rating_logits.float(),
                dim=-1,
            ).cpu().numpy()

            prediction = int(np.argmax(probabilities)) + 1
            target = int(example["target_rating"])

            predictions.append(prediction)
            targets.append(target)

            records.append({
                "target_asin": example.get("target_asin"),
                "target_rating": target,
                "predicted_rating": prediction,
                "probabilities": probabilities.tolist(),
            })

            del outputs

    mae = mean_absolute_error(targets, predictions)
    rmse = np.sqrt(mean_squared_error(targets, predictions))
    accuracy = accuracy_score(targets, predictions)
    macro_f1 = f1_score(
        targets,
        predictions,
        labels=[1, 2, 3, 4, 5],
        average="macro",
        zero_division=0,
    )

    precision, recall, f1, support = (
        precision_recall_fscore_support(
            targets,
            predictions,
            labels=[1, 2, 3, 4, 5],
            zero_division=0,
        )
    )

    results = {
        "experiment": args.name,
        "precision": args.precision,
        "adapter": args.adapter,
        "num_examples": len(examples),
        "mae": float(mae),
        "rmse": float(rmse),
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "per_class": {
            str(rating): {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, rating in enumerate(range(1, 6))
        },
        "confusion_matrix": confusion_matrix(
            targets,
            predictions,
            labels=[1, 2, 3, 4, 5],
        ).tolist(),
        "prediction_distribution": {
            str(r): int(predictions.count(r))
            for r in range(1, 6)
        },
    }

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    with open(output_dir / "predictions.jsonl", "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument(
        "--test-file",
        default="data/sft/rating/test.jsonl",
    )
    parser.add_argument(
        "--precision",
        choices=["qlora", "bf16"],
        default="qlora",
    )
    parser.add_argument("--limit", type=int, default=None)

    main(parser.parse_args())
