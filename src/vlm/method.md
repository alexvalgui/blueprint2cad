## Benchmark: CMOMB (Cross-Modal Object Matching Benchmark)

### Task Formulation
**Input**: Two RGB images from any combination of your three modalities  
**Question**: *"Do these two images show the same object?"*  
**Output**: Binary `yes` / `no`

### Your Data Structure
```
objects/obj_{ID:05d}/
├── render_view_{0-5}.png        (3D rendering)
├── drawing_view_{0-5}.png       (2D line drawing)
└── dimensioned_view_{0-5}.png   (2D drawing + dimensions)
```

---

## Methodology Overview

### Pair Taxonomy (4 Categories)

| Category | Label | Description | Purpose |
|----------|-------|-------------|---------|
| **P-SV** | `yes` | Same object, **same view**, different modality | Tests view-aligned cross-modal matching |
| **P-DV** | `yes` | Same object, **different view**, any modality | Tests view-invariant identity recognition |
| **N-SM** | `no` | Different objects, **same modality** | Tests intra-modality discrimination |
| **N-CM** | `no` | Different objects, **different modalities** | Tests hardest cross-modality discrimination |

### Modality Combinations Covered
- `render ↔ drawing` — photorealistic vs. line art
- `render ↔ dimensioned` — 3D vs. annotated 2D
- `drawing ↔ dimensioned` — clean vs. dimensioned line art
- `render ↔ render`, `drawing ↔ drawing`, `dimensioned ↔ dimensioned` — same-modality negatives

### Sampling Strategy
- **Balanced**: 500 positive / 500 negative
- **Stratified**: 250 pairs per category
- **View coverage**: All 6 views uniformly sampled
- **Object coverage**: ~1000 unique objects drawn from your 30,000 pool

---

## Sample of 1,000 Pairs — Structure

Each row in the dataset contains:

| Field | Description |
|-------|-------------|
| `pair_id` | Unique ID (`pair_0000` to `pair_0999`) |
| `label` | Ground truth: `yes` or `no` |
| `category` | `P-SV`, `P-DV`, `N-SM`, or `N-CM` |
| `obj_a`, `obj_b` | Object IDs (1–30,000) |
| `modality_a`, `modality_b` | `render` / `drawing` / `dimensioned` |
| `view_a`, `view_b` | View index (0–5) |
| `image_a`, `image_b` | Relative file paths |
| `question` | VQA prompt text |

### Example Pairs

**P-SV** (Positive Same-View):
- `objects/obj_01803/render_view_3.png` ↔ `objects/obj_01803/dimensioned_view_3.png` → **yes**

**P-DV** (Positive Diff-View):
- `objects/obj_12776/render_view_4.png` ↔ `objects/obj_12776/drawing_view_5.png` → **yes**

**N-SM** (Negative Same-Modality):
- `objects/obj_26839/render_view_2.png` ↔ `objects/obj_02483/render_view_2.png` → **no**

**N-CM** (Negative Cross-Modality):
- `objects/obj_03984/dimensioned_view_5.png` ↔ `objects/obj_01779/drawing_view_2.png` → **no**

---

## Evaluation Metrics Recommended

1. **Accuracy** — overall correctness
2. **Balanced Accuracy** — average of sensitivity & specificity
3. **F1-Score** — for the `yes` class
4. **Category-wise Accuracy** — breakdown by P-SV / P-DV / N-SM / N-CM
5. **Modality-wise Accuracy** — per combination (Rd, Rm, Dm, etc.)
6. **AUC-ROC** — if your VLM outputs confidence scores

---

## Prompt Template (Basic)

```
Image 1: [image_a_path]
Image 2: [image_b_path]

Question: Do these two images show the same object?
Answer with only "yes" or "no".
```

Alternative variants (with modality context, chain-of-thought, few-shot) are included in the prompt templates file.


Yes. I would add a dedicated **Model Benchmark** section distinguishing **closed/proprietary VLMs** from **open-source/open-weight VLMs**. This makes the benchmark more useful for reproducibility and gives you a clear comparison across model families.

---

## Models to Test

Rather than testing every available model, organize the experiments into four groups:

### A. Closed VLMs

Evaluate 3–5 strong proprietary models:

* GPT-5.6
* Gemini 2.5 Pro
* Gemini 2.5 Flash
* Claude 4 Sonnet
* Claude 4 Opus

This establishes the performance ceiling of current commercial VLMs.

### B. Large Open VLMs

Evaluate:

* Qwen3-VL-30B-A3B-Instruct
* Qwen3-VL-235B-A22B-Instruct
* Llama 4 Maverick
* InternVL3-14B

These test whether open models can approach proprietary models.

### C. Small / Efficient Open VLMs

Evaluate:

* Qwen3-VL-2B-Instruct
* Qwen2.5-VL-3B-Instruct
* Gemma 3 4B
* MiniCPM-V 4.5

This is particularly interesting for your benchmark because **cross-modal CAD matching could potentially be performed locally without a large model**.

---

## Suggested Final Benchmark Structure

I would therefore structure the paper as:

```text
CMOMB
│
├── Closed VLMs
│   ├── GPT-5.6
│   ├── Gemini 2.5 Pro
│   ├── Gemini 2.5 Flash
│   └── Claude 4
│
├── Open-Weight VLMs
    ├── Qwen3-VL
    ├── Qwen2.5-VL
    ├── Llama 4
    ├── InternVL3
    ├── Gemma 3
    └── MiniCPM-V

```