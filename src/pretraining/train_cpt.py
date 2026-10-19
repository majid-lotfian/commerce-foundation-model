
import argparse
import csv
import time
from pathlib import Path

import torch
from datasets import load_dataset
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)


CHECKPOINT_STEPS = {100, 250, 500, 1000}


class ExperimentCallback(TrainerCallback):
    def __init__(self, output_dir):
        self.metrics_file = Path(output_dir) / "metrics.csv"
        self.metrics_file.parent.mkdir(parents=True, exist_ok=True)
        self.start_time = time.perf_counter()

        if not self.metrics_file.exists():
            with open(self.metrics_file, "w", newline="") as f:
                csv.writer(f).writerow([
                    "step",
                    "epoch",
                    "train_loss",
                    "eval_loss",
                    "learning_rate",
                    "elapsed_seconds",
                ])

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step in CHECKPOINT_STEPS:
            control.should_save = True
            control.should_evaluate = True
        return control

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return

        train_loss = logs.get("loss")
        eval_loss = logs.get("eval_loss")

        if train_loss is None and eval_loss is None:
            return

        elapsed = time.perf_counter() - self.start_time

        with open(self.metrics_file, "a", newline="") as f:
            csv.writer(f).writerow([
                state.global_step,
                state.epoch,
                train_loss,
                eval_loss,
                logs.get("learning_rate"),
                round(elapsed, 2),
            ])

        if eval_loss is not None:
            print(
                f"\nStep {state.global_step} | "
                f"Validation loss: {eval_loss:.4f}",
                flush=True,
            )


def collate_fn(features):
    input_ids = torch.tensor(
        [f["input_ids"] for f in features],
        dtype=torch.long,
    )

    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "labels": input_ids.clone(),
    }


def main(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not available.")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    print("GPU:", torch.cuda.get_device_name(0), flush=True)

    print("Loading training dataset...", flush=True)
    train_dataset = load_dataset(
        "parquet",
        data_files=str(Path(args.input).resolve()),
        split="train",
    )

    print("Loading validation dataset...", flush=True)
    eval_dataset = load_dataset(
        "parquet",
        data_files=str(Path(args.eval_input).resolve()),
        split="train",
    )

    eval_dataset = eval_dataset.select(
        range(min(args.eval_sequences, len(eval_dataset)))
    )

    print(f"Training sequences: {len(train_dataset):,}")
    print(f"Validation sequences: {len(eval_dataset):,}", flush=True)

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print("Loading Qwen3-8B...", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=quantization_config,
        dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="sdpa",
    )

    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
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
    model.gradient_checkpointing_enable()
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=str(output),
        max_steps=args.max_steps,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        bf16=True,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=1000000,
        save_strategy="steps",
        save_steps=1000000,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=2,
        gradient_checkpointing=True,
        optim="adamw_torch",
        warmup_steps=20,
        lr_scheduler_type="cosine",
        disable_tqdm=False,
        save_total_limit=None,
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_fn,
        callbacks=[ExperimentCallback(output)],
    )

    print("\nStarting CPT experiment...", flush=True)

    trainer.train(
        resume_from_checkpoint=args.resume
        if args.resume else None
    )

    print("\nSaving final adapter...", flush=True)
    trainer.save_model(str(output / "final_adapter"))
    trainer.save_state()

    print(
        "Peak allocated GPU memory:",
        f"{torch.cuda.max_memory_allocated() / 1e9:.2f} GB",
    )
    print("Training completed.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default="data/cpt/packed/cpt_sample_2048.parquet",
    )
    parser.add_argument(
        "--eval-input",
        default="data/cpt/eval/cpt_validation_2048.parquet",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-8B",
    )
    parser.add_argument(
        "--output",
        default="outputs/cpt_qlora_1000",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--eval-sequences",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--resume",
        default=None,
    )

    main(parser.parse_args())
