# Finetuning Report — Second Iteration
**Model:** CodeLlama-7B-Instruct-HF · **Method:** QLoRA (4-bit) · **Platform:** Kaggle T4 GPU  
**Date:** July 29, 2026 · **Status:** Completed Successfully (Early Stopping triggered at Epoch 4)

---

## 1. Overview

This report documents the second finetuning experiment for the **ITS AI** project. The base model `codellama/CodeLlama-7b-Instruct-hf` was again fine-tuned using **QLoRA** (Quantized Low-Rank Adaptation) on the same Indonesian student code assessment dataset used in the first iteration. Key changes from the first iteration include:

- **Increased LoRA rank** from `r=16` to `r=32`, and **LoRA alpha** from `32` to `64` — doubling the adapter capacity for stronger task adaptation.
- **Added Early Stopping** (patience=2) to automatically halt training when validation loss stopped improving — directly addressing the overfitting observed in iteration 1.
- **Extended max epochs** from 3 to 6, allowing early stopping to determine the optimal stopping point instead of a fixed epoch count.
- **Warmup Ratio adjusted** from 0.03 to a linear ramp over the first steps (learning rate climbs from `1.33e-4` to peak `~2e-4` in ~20 steps).

The goal remains: teach the model to evaluate student Python code and produce structured scoring + feedback in Bahasa Indonesia.

---

## 2. Training Environment

| Property | Value |
|---|---|
| **Platform** | Kaggle Notebook |
| **GPU** | 2× NVIDIA Tesla T4 (15,360 MiB each) |
| **CUDA Version** | 13.0 |
| **Driver Version** | 580.159.04 |
| **Python Version** | 3.12.13 |
| **Framework** | PyTorch + HuggingFace Transformers |
| **PEFT Version** | 0.13.2 |
| **TRL** | SFTTrainer / SFTConfig |
| **Quantization** | BitsAndBytes 4-bit (NF4) |

---

## 3. Dataset

### Sources
Three datasets were used from Kaggle (`hanzfr` account) — identical to the first iteration:
- `/kaggle/input/datasets/hanzfr/soal-ujian/soal_ujian.json` — exam questions + expected outputs
- `/kaggle/input/datasets/hanzfr/nilai-ujian/nilai_ujian.json` — student scores & feedback
- `/kaggle/input/datasets/hanzfr/train-data/train_data.jsonl` — pre-built training data

### Dataset Statistics

| Metric | Value |
|---|---|
| **Total Usable Examples** | 1,512 |
| **Min Token Length** | 274 tokens |
| **Max Token Length** | 1,151 tokens |
| **Mean Token Length** | ~511 tokens |
| **95th Percentile (p95)** | 842 tokens |
| **Max Sequence Length (training)** | 1,024 tokens |

### Train / Validation Split
Split was performed at the **question level** (`id_soal`) to prevent data leakage — unchanged from iteration 1.

| Split | Count | Proportion |
|---|---|---|
| **Train** | 1,296 examples | ~85.7% |
| **Validation** | 216 examples | ~14.3% |

### Prompt Format
Identical to iteration 1 — CodeLlama instruction template:
```
<s>[INST] Soal: {exam_question}
Output yang diharapkan: {expected_output}

Kode siswa:
```python
{student_code}
```

Nilai kode siswa ini dan berikan feedback. [/INST] Penilaian:
- {criterion}: {score}
...

Rata-rata: {avg_score}

Feedback: {feedback_text} </s>
```

---

## 4. Model Configuration

### Base Model
- **Model:** `codellama/CodeLlama-7b-Instruct-hf`
- **Commit:** `22cb240e0292b0b5ab4c17ccd97aa3a2f799cbed`
- **Task Type:** Causal Language Modeling (CAUSAL_LM)

### Quantization (QLoRA)

| Setting | Value |
|---|---|
| **Quantization** | 4-bit (load_in_4bit) |
| **Quantization Type** | NF4 (Normal Float 4) |
| **Compute Dtype** | float16 |
| **Double Quantization** | Enabled |

### LoRA Adapter Configuration

| Parameter | Value | Change vs. Iteration 1 |
|---|---|---|
| **Rank (r)** | 32 | ↑ from 16 |
| **LoRA Alpha** | 64 | ↑ from 32 |
| **LoRA Dropout** | 0.1 | Unchanged |
| **Bias** | none | Unchanged |
| **Use DoRA** | false | Unchanged |
| **Use RSLoRA** | false | Unchanged |
| **Target Modules** | `q_proj`, `k_proj`, `v_proj`, `o_proj` | Unchanged |

> Doubling rank and alpha increases the adapter's representational capacity. This gives the model more parameters to learn nuanced scoring patterns, at the cost of a larger adapter (~128 MB vs. ~64 MB in iteration 1).

### Training Hyperparameters

| Hyperparameter | Value | Change vs. Iteration 1 |
|---|---|---|
| **Max Epochs** | 6 | ↑ from 3 |
| **Early Stopping** | patience=2 | ✅ New |
| **Per-device Batch Size** | 2 | Unchanged |
| **Gradient Accumulation Steps** | 8 | Unchanged |
| **Effective Batch Size** | 16 | Unchanged |
| **Learning Rate (peak)** | ~2e-4 | Unchanged |
| **LR Scheduler** | Cosine | Unchanged |
| **Optimizer** | paged_adamw_8bit | Unchanged |
| **Precision** | fp16 | Unchanged |
| **Max Sequence Length** | 1,024 tokens | Unchanged |
| **Gradient Checkpointing** | Enabled | Unchanged |
| **Max Training Steps** | 486 (6 epochs) | ↑ from 243 |
| **Actual Stopping Step** | 324 (Epoch 4) | Stopped by Early Stopping |
| **Eval & Save Strategy** | Per epoch | Unchanged |
| **Logging Steps** | Every 10 steps | Unchanged |

---

## 5. Training Results

### Loss Curve (Training Loss per Step)

| Step | Epoch | Training Loss | Grad Norm | Learning Rate |
|---:|---:|---:|---:|---:|
| 10 | 0.12 | 1.4407 | 0.3077 | 1.333e-4 |
| 20 | 0.25 | 1.0094 | 0.2575 | 1.999e-4 |
| 30 | 0.37 | 0.7606 | 0.1893 | 1.995e-4 |
| 40 | 0.49 | 0.6636 | 0.2374 | 1.986e-4 |
| 50 | 0.62 | 0.6156 | 0.2659 | 1.973e-4 |
| 60 | 0.74 | 0.5461 | 0.2538 | 1.955e-4 |
| 70 | 0.86 | 0.5001 | 0.2610 | 1.933e-4 |
| 80 | 0.99 | 0.4893 | 0.2294 | 1.907e-4 |
| **81** | **1.00** | — | — | — |
| 90 | 1.11 | 0.4658 | 0.2221 | 1.877e-4 |
| 100 | 1.23 | 0.4336 | 0.2373 | 1.844e-4 |
| 110 | 1.36 | 0.4338 | 0.2518 | 1.806e-4 |
| 120 | 1.48 | 0.4119 | 0.2822 | 1.765e-4 |
| 130 | 1.60 | 0.4156 | 0.3159 | 1.720e-4 |
| 140 | 1.73 | 0.3807 | 0.2717 | 1.672e-4 |
| 150 | 1.85 | 0.3897 | 0.2389 | 1.621e-4 |
| 160 | 1.98 | 0.4032 | 0.2856 | 1.568e-4 |
| **162** | **2.00** | — | — | — |
| 170 | 2.10 | 0.3540 | 0.2505 | 1.512e-4 |
| 180 | 2.22 | 0.3322 | 0.3030 | 1.453e-4 |
| 190 | 2.35 | 0.3364 | 0.3451 | 1.393e-4 |
| 200 | 2.47 | 0.3441 | 0.3109 | 1.331e-4 |
| 210 | 2.59 | 0.3497 | 0.3119 | 1.267e-4 |
| 220 | 2.72 | 0.3352 | 0.3235 | 1.202e-4 |
| 230 | 2.84 | 0.3341 | 0.2695 | 1.136e-4 |
| 240 | 2.96 | 0.3218 | 0.3217 | 1.070e-4 |
| **243** | **3.00** | — | — | — |
| 250 | 3.09 | 0.2952 | 0.2937 | 1.003e-4 |
| 260 | 3.21 | 0.2826 | 0.2836 | 9.367e-5 |
| 270 | 3.33 | 0.2871 | 0.3133 | 8.703e-5 |
| 280 | 3.46 | 0.2876 | 0.3212 | 8.045e-5 |
| 290 | 3.58 | 0.2876 | 0.3526 | 7.396e-5 |
| 300 | 3.70 | 0.2936 | 0.3228 | 6.758e-5 |
| 310 | 3.83 | 0.2683 | 0.3220 | 6.135e-5 |
| 320 | 3.95 | 0.2844 | 0.2865 | 5.529e-5 |
| **324** | **4.00** | — | — | — |
| *(stopped)* | — | — | — | — |

### Evaluation Loss (Per Epoch)

| Epoch | Eval Loss | Runtime | Samples/sec | Early Stopping Counter |
|---:|---:|---:|---:|---:|
| 1 | 0.7895 | 93.68s | 2.306 | — |
| 2 | **0.7501** ⭐ | 93.32s | 2.315 | 0 (reset — new best) |
| 3 | 0.7671 | 93.26s | 2.316 | 1 |
| 4 | 0.7877 | 93.34s | 2.314 | 2 → **STOPPED** |

### Key Observations

- **Training loss dropped significantly** from 1.4407 → 0.2844 across 4 epochs (~**80% reduction**) — slightly better convergence than iteration 1 (73% over 3 epochs), enabled by the higher-capacity adapter (r=32).
- **Gradient norms remained stable** throughout (~0.19–0.35), indicating a healthy run with no exploding gradients.
- **Early Stopping worked correctly**: validation loss improved at Epoch 2 (0.7501, best), worsened at Epoch 3 (patience counter = 1), worsened again at Epoch 4 (patience counter = 2 → stopped). Training was terminated before Epoch 5 as designed.
- **Best checkpoint is `checkpoint-162` (Epoch 2)** with eval loss of **0.7501**, an improvement of **~0.032** over the best from iteration 1 (0.7820, Epoch 1).
- **Total FLOPs:** ~1.26 × 10¹⁷ (vs. ~9.42 × 10¹⁶ in iteration 1 — ~34% more compute due to 4 epochs of training)

> [!IMPORTANT]
> **Best checkpoint confirmed:** `checkpoint-162` (Epoch 2, eval loss: **0.7501**) is definitively the best model in this run. Early stopping with patience=2 confirmed that neither Epoch 3 nor Epoch 4 could improve on it. Use this checkpoint for inference.

> [!NOTE]
> **Improvement vs. Iteration 1:** The best eval loss improved from **0.7820** (iteration 1, Epoch 1) to **0.7501** (iteration 2, Epoch 2) — a **~4.1% reduction**. This suggests the larger adapter (r=32) better captures the task structure, and that Epoch 2 is a better stopping point than Epoch 1 for this configuration.

---

## 6. Checkpoints & Saved Artifacts

| Checkpoint | Step | Epoch | Adapter Size | Eval Loss | Notes |
|---|---:|---:|---|---:|---|
| `checkpoint-81` | 81 | 1 | ~128 MB | 0.7895 | End of Epoch 1 |
| `checkpoint-162` | 162 | 2 | ~128 MB | **0.7501** | ⭐ **Best checkpoint** |
| `checkpoint-243` | 243 | 3 | ~128 MB | 0.7671 | Early stopping counter: 1 |
| `checkpoint-324` | 324 | 4 | ~128 MB | 0.7877 | Early stopping counter: 2 → **Training stopped here** |
| `final_model/` | 324 | 4 | ~128 MB | 0.7877 | Same as checkpoint-324 |

> [!WARNING]
> **Do NOT use `final_model/` for inference.** It corresponds to `checkpoint-324` (Epoch 4), which has a higher eval loss (0.7877) than the best checkpoint (0.7501). Always load from `checkpoint-162`.

All checkpoints contain: `adapter_config.json`, `adapter_model.safetensors`, tokenizer files, `training_args.bin`, `trainer_state.json`, and optimizer/scheduler state.

---

## 7. Comparison with First Iteration

| Metric | Iteration 1 | Iteration 2 | Δ |
|---|---|---|---|
| **LoRA Rank (r)** | 16 | 32 | +16 |
| **LoRA Alpha** | 32 | 64 | +32 |
| **Adapter Size** | ~64 MB | ~128 MB | +64 MB |
| **Max Epochs Set** | 3 | 6 | +3 |
| **Actual Epochs Run** | 3 | 4 | +1 |
| **Early Stopping** | ❌ No | ✅ Yes (patience=2) | Added |
| **Total Steps** | 243 | 324 | +81 |
| **Best Epoch** | Epoch 1 | Epoch 2 | +1 |
| **Best Eval Loss** | 0.7820 | **0.7501** | **−0.0319 (−4.1%)** |
| **Final Train Loss** | 0.3859 | 0.2844 | −0.1015 (−26.3%) |
| **Total FLOPs** | ~9.42 × 10¹⁶ | ~1.26 × 10¹⁷ | +34% |
| **PEFT Version** | 0.19.1 | 0.13.2 | Downgraded |

> [!NOTE]
> PEFT version was downgraded from 0.19.1 to 0.13.2 between iterations. This may reflect a deliberate pin to a stable version for compatibility, or a different Kaggle environment snapshot.

---

## 8. Output Files Summary

```
finetuning_results/second_iteration/
├── __huggingface_repos__.json       # HF model download manifest
├── train_data.jsonl                 # Full formatted dataset (1,512 examples, ~1.9 MB)
├── train_split.jsonl                # Training split (1,296 examples, ~1.7 MB)
├── val_split.jsonl                  # Validation split (216 examples, ~238 KB)
├── checkpoints/
│   ├── checkpoint-81/               # End of Epoch 1
│   ├── checkpoint-162/              # End of Epoch 2  [Best eval loss ⭐]
│   ├── checkpoint-243/              # End of Epoch 3
│   └── checkpoint-324/              # End of Epoch 4 (= final_model, early stop)
└── final_model/
    ├── adapter_config.json          # LoRA config (r=32, alpha=64)
    ├── adapter_model.safetensors    # Trained LoRA weights (~128 MB)
    ├── tokenizer.json               # Full tokenizer (~3.5 MB)
    ├── tokenizer.model              # SentencePiece model (~488 KB)
    ├── tokenizer_config.json        # Tokenizer settings
    ├── special_tokens_map.json      # Special token mappings
    ├── training_args.bin            # Serialized TrainingArguments
    └── README.md                    # HuggingFace model card (auto-generated)
```

---

## 9. Analysis & Recommendations

### What Improved
- **Best eval loss improved** from 0.7820 → 0.7501 (~4.1% better), demonstrating that the higher-capacity adapter (r=32) provides meaningful gains.
- **Early stopping worked as designed** — eliminated wasted compute in epochs 5 and 6, and provided a principled stopping criterion.
- **Training converged further** — final training loss reached 0.2844 vs. 0.3859 in iteration 1, suggesting the model is learning finer task patterns.

### Remaining Concerns

1. **Eval loss gap persists:** The train/eval divergence gap (~0.42 at Epoch 4: train ~0.28, eval ~0.79) remains significant. While early stopping capped the overfitting, the gap indicates the model still memorizes more than it generalizes.

2. **Eval loss valley is shallow:** The best eval loss (0.7501 at Epoch 2) is only marginally better than Epoch 3 (0.7671). The model may benefit from data augmentation or regularization to deepen this valley.

3. **Adapter size doubled:** The ~128 MB adapter is still deployment-friendly, but represents a cost relative to iteration 1. If inference latency or memory is constrained, consider r=24 as a middle ground.

4. **PEFT version mismatch:** PEFT 0.13.2 (iteration 2) vs. 0.19.1 (iteration 1) — verify that adapter weights are loaded consistently across environments.

### 🔧 Recommendations for Next Iteration

| # | Recommendation | Priority |
|---|---|---|
| 1 | **Use `checkpoint-162`** for inference — confirmed best by early stopping | High |
| 2 | **Qualitative evaluation** — generate predictions on held-out examples and manually review scoring quality | High |
| 3 | **Increase validation set** to 20% split (from 14.3%) to get a more stable eval signal | Medium |
| 4 | **Experiment with data augmentation** or label smoothing to reduce train/eval gap | Medium |
| 5 | **Pin PEFT version** — choose either 0.13.2 or 0.19.1 consistently across training and inference | Medium |
| 6 | **Test intermediate rank** (r=24, alpha=48) to balance capacity vs. adapter size | Low |
| 7 | **Add tokenizer.padding_side = 'right'** fix — if not already applied in this run | Medium |
| 8 | **Benchmark model outputs** side-by-side against iteration 1 `checkpoint-81` on the same held-out set | High |

---

*Report generated: July 29, 2026 · Based on artifacts in `finetuning_results/second_iteration/`*
