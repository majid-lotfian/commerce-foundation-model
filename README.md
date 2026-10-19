# Commerce Foundation Model: Amazon Reviews 2023

**Continued pretraining and parameter-efficient fine-tuning of Qwen3-8B for e-commerce review understanding and rating prediction.**

This repository documents an end-to-end experimentation pipeline, from Amazon Reviews 2023 data preparation through continued pretraining (CPT), supervised fine-tuning (SFT), and evaluation. The focus is on reproducible training scripts and transparent reporting of both successful and limited outcomes.

![Amazon Review LLM Adaptation Pipeline](Amazon%20Review%20LLM%20Adaptation%20Pipeline.png)

## Project overview

The project explores two related questions:

1. Can continued pretraining on domain-specific product reviews improve the language-modeling fit of a general-purpose LLM to e-commerce text?
2. Do continued pretraining and different parameter-efficient fine-tuning configurations improve downstream **1–5 star rating prediction**?

The base model is **Qwen3-8B**. Experiments used NVIDIA H100 GPUs, PyTorch, Hugging Face Transformers, PEFT/LoRA, and QLoRA with 4-bit NF4 quantization. Batch execution scripts are provided for Slurm environments.

## Data

The pipeline processes public **Amazon Reviews 2023** data from four categories:

- All Beauty
- Electronics
- Home & Kitchen
- Sports & Outdoors

Approximately **131.6 million interactions** were processed during data preparation. The continued-pretraining corpus comprised approximately **1 million documents / 108 million tokens**. The downstream evaluation set contained **9,885 examples**.

The dataset is not redistributed in this repository. Scripts under `src/data/` cover downloading, inspection, cleaning, interaction and history construction, splitting, and preparation of CPT and rating-SFT inputs.

## Pipeline

### 1. Data preparation

Use the scripts in `src/data/` to prepare raw reviews, interactions, user histories, and training/evaluation datasets. Relevant entry points include:

- `download_dataset.py`, `inspect_dataset.py`, `prepare_data.py`
- `build_interactions.py`, `build_user_histories.py`, `build_rating_histories.py`
- `clean_cpt.py`, `prepare_cpt.py`
- `select_rating_targets.py`, `prepare_rating_sft.py`, `format_rating_sft.py`
- `split_dataset.py`

Additional exploratory and recommendation-data scripts are retained in the same directory. Their presence does not imply that a recommendation model was trained or evaluated in the results below.

### 2. Continued pretraining (CPT)

The domain-adaptation workflow includes:

- `src/pretraining/tokenize_cpt.py`
- `src/pretraining/pack_sequences.py`
- `src/pretraining/train_cpt.py`
- `src/pretraining/prepare_cpt_eval.py`
- `src/pretraining/evaluate_cpt.py`

CPT was performed using a parameter-efficient QLoRA setup. Evaluation compares held-out review next-token prediction before and after domain adaptation.

### 3. Rating supervised fine-tuning (SFT)

Rating-prediction experiments use `src/pretraining/train_rating_sft.py`. The evaluated variants include:

- QLoRA with LoRA rank **8**, **16**, and **32**
- SFT starting from the original base model versus the CPT-adapted model
- **BF16 LoRA** at rank 16 as an alternative to 4-bit QLoRA

Slurm job definitions for the evaluated rating variants are in `slurm/`.

### 4. Evaluation

- CPT: `src/pretraining/evaluate_cpt.py`
- Rating prediction: `src/evaluation/evaluate_rating.py`
- Rating metrics: accuracy, MAE, RMSE, macro F1, and per-class diagnostics

## Results

### Continued pretraining: held-out next-token prediction

| Metric | Original Qwen3-8B | After CPT | Change |
|---|---:|---:|---:|
| Validation loss ↓ | 2.9701 | **2.3914** | ~19.5% lower |
| Perplexity ↓ | 19.49 | **10.93** | ~43.9% lower |

These results indicate substantially better fit to held-out review text following continued pretraining. **They do not, by themselves, establish better downstream task performance.**

### Downstream task: 1–5 star rating prediction

All configurations were evaluated on **9,885 test examples**.

| Configuration | Accuracy ↑ | MAE ↓ | RMSE ↓ | Macro F1 ↑ |
|---|---:|---:|---:|---:|
| Original Qwen3-8B | 0.5927 | 0.7226 | **1.3154** | 0.2461 |
| Base + QLoRA, r16 | 0.6816 | 0.7171 | 1.4432 | 0.2433 |
| CPT + QLoRA, r16 | **0.6836** | **0.7116** | 1.4366 | 0.2460 |
| QLoRA, r8 | 0.6829 | 0.7156 | 1.4427 | **0.2465** |
| QLoRA, r32 | 0.6828 | 0.7137 | 1.4393 | 0.2454 |
| BF16 LoRA, r16 | 0.6824 | 0.7140 | 1.4394 | 0.2450 |

**Interpretation:** Fine-tuned configurations achieved higher accuracy than the original-model evaluation, and CPT + QLoRA r16 had the best observed accuracy and MAE. However, the original model had the lowest RMSE, and macro F1 remained low across all configurations. Thus, these results **do not support a claim of consistent improvement across metrics**. Differences between fine-tuned configurations are small; no statistical-significance claim is made.

The rating distribution is highly imbalanced: **6,588 / 9,885 (66.65%)** of test examples are five-star reviews. Accuracy should therefore be considered alongside macro F1, class-wise results, and error metrics. The original-model evaluation's prediction distribution and confusion matrix are recorded in its evaluation output, rather than reproduced here.

> **Comparability note:** The table reports the recorded evaluation outputs. Before treating the original-model row as a strictly controlled baseline, verify that prompt formatting, decoding, and rating extraction were identical across configurations.

## Repository structure

```text
commerce-foundation-model/
├── README.md
├── Amazon Review LLM Adaptation Pipeline.png
├── .gitignore
├── src/
│   ├── data/                  # Downloading, preprocessing, splits, CPT/SFT preparation
│   ├── pretraining/           # CPT and rating-SFT training; CPT evaluation
│   ├── evaluation/            # Rating prediction evaluation
│   ├── alignment/             # Placeholder
│   ├── finetuning/            # Placeholder
│   └── serving/               # Placeholder
├── slurm/                     # Rating training and evaluation batch jobs
├── configs/                   # Placeholder
└── tests/                     # Placeholder
```

Directories such as `data/`, `checkpoints/`, `output/`, and `results/` may exist locally or on the training system but are **not published as datasets, model checkpoints, or large generated artifacts** in this repository.

## Reproducing the experiments

1. Obtain access to the public **Amazon Reviews 2023** dataset and the **Qwen3-8B** base model, accepting any applicable licenses and terms.
2. Set up a Python environment with the versions of PyTorch, Transformers, Datasets, PEFT, Accelerate, and bitsandbytes compatible with your GPU/CUDA environment.
3. Inspect the relevant scripts and adapt their input/output paths and configuration arguments to your environment. Prepare the four categories using `src/data/`.
4. Build, tokenize, and pack the CPT corpus; run `src/pretraining/train_cpt.py`; then evaluate held-out next-token prediction.
5. Prepare the rating-SFT dataset, run the selected configurations with `src/pretraining/train_rating_sft.py`, and evaluate using `src/evaluation/evaluate_rating.py`.
6. On a Slurm cluster, adapt the `.sbatch` scripts in `slurm/` for your account, partitions, storage paths, and GPU allocation before submission.

**Important:** This repository documents the experimental scripts, but does not currently provide a pinned environment/lockfile or a single-command end-to-end reproduction workflow. Exact CLI flags and input file paths should be taken from the scripts rather than assumed from this overview.

## Scope and limitations

- The clearest positive finding is improved **domain language modeling** after CPT.
- Rating prediction remained challenging, particularly on minority rating classes.
- This repository does **not** claim that CPT reliably improves downstream rating prediction or that any configuration is statistically superior.
- Raw datasets, large generated files, checkpoints, and model weights are intentionally excluded from Git.

## Acknowledgments

This work uses the Amazon Reviews 2023 dataset and the Qwen3 model family. Please consult the original dataset and model documentation for their citations and licenses.
