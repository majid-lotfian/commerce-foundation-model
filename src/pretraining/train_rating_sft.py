import argparse
import hashlib

import json
import random
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset, load_from_disk
from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)

from tqdm import tqdm


class ProgressCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and state.is_world_process_zero:
            print(
                f"[step {state.global_step}] "
                + " | ".join(
                    f"{k}={v:.4f}" if isinstance(v, float)
                    else f"{k}={v}"
                    for k, v in logs.items()
                ),
                flush=True,
            )


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def make_dataset(path, tokenizer, max_length):
    source = Path(path)

    # Cache depends on source data, tokenizer, sequence length,
    # and tokenization implementation version.
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()[:12]

    cache_key = (
        f"{source.stem}_"
        f"{tokenizer.name_or_path.replace('/', '_')}_"
        f"{max_length}_v1_{source_hash}"
    )

    cache_path = Path("data/sft/cache") / cache_key

    if cache_path.exists():
        print(f"Loading cached dataset: {cache_path}", flush=True)
        return load_from_disk(str(cache_path))

    examples = load_jsonl(path)
    records = []

    for example in tqdm(examples, desc=f"Tokenizing {source.name}"):
        messages = example["messages"]
        rating = str(example["target_rating"])

        prompt_ids = tokenizer.apply_chat_template(
            messages[:-1],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )["input_ids"]

        answer_ids = tokenizer.encode(
            rating,
            add_special_tokens=False,
        )

        end_ids = tokenizer.encode(
            "<|im_end|>",
            add_special_tokens=False,
        )

        input_ids = prompt_ids + answer_ids + end_ids

        if len(input_ids) > max_length:
            raise ValueError(
                f"Example has {len(input_ids)} tokens, "
                f"exceeding max_length={max_length}"
            )

        labels = (
            [-100] * len(prompt_ids)
            + answer_ids
            + end_ids
        )

        records.append({
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": labels,
        })

    dataset = Dataset.from_list(records)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(cache_path))

    print(
        f"Prepared and cached {len(dataset):,} examples: {cache_path}",
        flush=True,
    )

    return dataset

class RatingCollator:
    def __init__(self, tokenizer):
        self.pad_id = tokenizer.pad_token_id

    def __call__(self, features):
        max_len = max(len(x["input_ids"]) for x in features)

        input_ids = []
        attention_mask = []
        labels = []

        for x in features:
            padding = max_len - len(x["input_ids"])

            input_ids.append(
                x["input_ids"] + [self.pad_id] * padding
            )
            attention_mask.append(
                x["attention_mask"] + [0] * padding
            )
            labels.append(
                x["labels"] + [-100] * padding
            )

        return {
            "input_ids": torch.tensor(
                input_ids, dtype=torch.long
            ),
            "attention_mask": torch.tensor(
                attention_mask, dtype=torch.long
            ),
            "labels": torch.tensor(
                labels, dtype=torch.long
            ),
        }


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not available.")

    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    print("Experiment:", args.experiment, flush=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Preparing training dataset...", flush=True)

    train_dataset = make_dataset(
        args.train_file,
        tokenizer,
        args.max_length,
    )

    print("Preparing validation dataset...", flush=True)

    eval_dataset = make_dataset(
        args.validation_file,
        tokenizer,
        args.max_length,
    )

    # Use a fixed, representative subset for intermediate evaluation.
    # Keep the full validation dataset available for later evaluation.
    full_eval_dataset = eval_dataset

    rng = np.random.default_rng(args.seed)
    eval_indices = rng.choice(
        len(full_eval_dataset),
        size=min(500, len(full_eval_dataset)),
        replace=False,
    )

    eval_dataset = full_eval_dataset.select(
        sorted(eval_indices.tolist())
    )

    print(
        f"Intermediate validation: {len(eval_dataset):,} examples "
        f"(full set: {len(full_eval_dataset):,})",
        flush=True,
    )

    print("Loading Qwen3-8B...", flush=True)
    print(f"Precision: {args.precision}", flush=True)

    if args.precision == "qlora":
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
            trust_remote_code=True,
            attn_implementation="sdpa",
        )

        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True,
        )

    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            dtype=torch.bfloat16,
            device_map={"": 0},
            trust_remote_code=True,
            attn_implementation="sdpa",
        )

        # Freeze the original model parameters.
        for param in model.parameters():
            param.requires_grad = False

    model.config.use_cache = False

    if args.experiment == "base":
        print("Creating new LoRA adapter...", flush=True)

        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        )

        model = get_peft_model(model, lora_config)

    elif args.experiment == "cpt":
        print(
            f"Loading CPT adapter: {args.cpt_adapter}",
            flush=True,
        )

        model = PeftModel.from_pretrained(
            model,
            args.cpt_adapter,
            is_trainable=True,
        )

    else:
        raise ValueError(args.experiment)

    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=20,
        optim="paged_adamw_8bit",
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={
            "use_reentrant": False,
        },
        max_grad_norm=1.0,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        logging_steps=10,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=4,
        seed=args.seed,
        data_seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=RatingCollator(tokenizer),
        processing_class=tokenizer,
        callbacks=[ProgressCallback()],
    )

    print("\nStarting SFT...", flush=True)

    from transformers.trainer_utils import get_last_checkpoint

    last_checkpoint = get_last_checkpoint(str(output_dir))

    if last_checkpoint:
        print(f"Resuming from {last_checkpoint}", flush=True)
    else:
        print("Starting training from scratch", flush=True)

    trainer.train(resume_from_checkpoint=last_checkpoint)

    final_dir = output_dir / "final_adapter"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    with open(output_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"\nTraining complete: {final_dir}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--experiment",
        choices=["base", "cpt"],
        required=True,
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-8B",
    )
    parser.add_argument(
        "--cpt-adapter",
        default="outputs/cpt_qlora_1000/final_adapter",
    )
    parser.add_argument(
        "--train-file",
        default="data/sft/rating/train.jsonl",
    )
    parser.add_argument(
        "--validation-file",
        default="data/sft/rating/validation.jsonl",
    )
    parser.add_argument("--output-dir", required=True)

    parser.add_argument("--max-length", type=int, default=3072)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--precision", choices=["qlora", "bf16"], default="qlora")

    main(parser.parse_args())
