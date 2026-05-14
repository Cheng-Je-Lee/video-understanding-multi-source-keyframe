# Query-Adaptive Multi-Source Frame Selection for Video Understanding

### Comic Book Hypothesis — Phase 2

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper](https://img.shields.io/badge/Paper-coming%20soon-lightgrey)]()
[![Benchmark](https://img.shields.io/badge/Benchmark-OVO--Bench-blue)](https://github.com/JoeLeelyf/OVO-Bench)
[![DOI](https://img.shields.io/badge/DOI-coming%20soon-lightgrey)]()

> This research was developed and tested collaboratively by an independent researcher and Claude Sonnet 4.6 and Gemini 3.1, without institutional affiliation. All development and experiments were conducted on Google Colab L4 GPU.

> **Part 2 of a 3-paper series** on sparse video understanding.
> - Part 1: Keyframe distillation — [`video-understanding-CBH`](https://github.com/cjlee/video-understanding-CBH)
> - Part 2 (this repo): Multi-source superposition architecture — *CBH Phase 2*
> - Part 3: Harness framework — coming soon

---

## Overview

This system answers temporal reasoning questions about videos by dynamically selecting frames from four independent source windows, aggregating them into a structured visual context, and querying a large language model (Claude API) with a Chain-of-Thought prompt.

The key insight is that different question types require fundamentally different temporal attention strategies. Rather than applying a single sampling policy to all questions, the system first classifies the question, then routes it through a specialized parameter configuration optimized for that question type.

The framework is evaluated on [OVO-Bench](https://github.com/JoeLeelyf/OVO-Bench) (CVPR 2025), a streaming video understanding benchmark covering 9 task types.

---

## Core Architecture

### Step 1 — Whole-Video CLIP Encoding + Temporal Decay Clustering

The video is encoded once per clip using CLIP ViT-B/32 at 1 frame/second. A cold-start S2 probe (frames 0–20s and 60–100s) estimates the video's semantic dynamics (μ, σ), which initializes the V4 Temporal Decay Clustering (TDC) algorithm to detect scene-transition keyframes (keyframe anchors) across the full video.

### Step 2 — Question Classification

Each question is parsed by spaCy to extract nouns, verbs, and spatial direction words, then classified into one of three types:

| Type | Examples | T_event Strategy |
|------|----------|-----------------|
| `action` | "What does X do after…" | noun similarity (0.6) + verb × semantic derivative (0.4) |
| `attribute` | "What color is…", "How many…" | noun similarity only |
| `spatial` | "Where is…", "Which direction…" | noun similarity + direction word similarity |

### Step 3 — T_event Localization

The semantic derivative (frame-to-frame CLIP embedding difference) encodes verb semantics analogously to optical flow in semantic space. This allows the system to localize the exact moment described in the question (`T_event`) using the question's noun and verb content as a multi-word weighted query.

### Step 4 — Four-Source Window Expansion

Two temporal anchors are established: `T_event` (the moment described) and `realtime` (the moment the question is asked). Four independent windows are expanded around them:

| Source | Label | Anchor | Sampling Strategy | Resolution |
|--------|-------|--------|-------------------|------------|
| D — Big Window | `[D-Window]` | T_event | Inverse-CDF Gaussian across ±2σ | 160×90 (T_event: full) |
| A — Keyframe Window | `[A-Window*]` | T_event | Sigmoid inverse-CDF per V4 segment | 224×126 (kf_anchor: 640×360) |
| C — Semantic Window | `[T_event]` | T_event | Fibonacci sparse, bilateral | full resolution |
| B — Realtime Window | `[PRE-REALTIME]` | realtime | Fibonacci sparse, unilateral | full resolution |

Each source contributes frames independently. Sources are merged, deduplicated by CLIP cosine similarity (threshold > 0.95), and labeled before being passed to the LLM.

### Step 5 — Auxiliary Analysis

- **YOLO-World**: Open-vocabulary object detection using question nouns as query terms. Applied to key anchor frames (`T_event`, `realtime`, `kf_anchor`).
- **Motion Analysis**: Bounding box displacement across consecutive frames near `T_event` and `realtime`, converted to natural language descriptions.
- **Action Localization**: CLIP-Seg multi-line analysis for `T_event+1` frame prediction (ASI / FPD task types).

### Step 6 — LLM Reasoning (Claude API)

All frames and text materials are organized into four categories and sent to `claude-sonnet-4-20250514` with a Chain-of-Thought system prompt:

1. **Category 1** — Big Window frames: temporal background context
2. **Category 2** — Keyframe Window frames: narrative segment structure
3. **Category 3** — Semantic Window + YOLO: T_event object state
4. **Category 4** — Realtime Window + Motion Analysis: cutoff moment state

The prompt enforces a three-step reasoning chain before answering:
- Step 1: Visual grounding (object state at T_event and realtime)
- Step 2: Motion and inertia analysis (trajectory from PRE-REALTIME to realtime)
- Step 3: Physical deduction (inevitable next state)

The LLM also scores each category's contribution (0–1) after answering, enabling per-source contribution analysis.

---

## Parameter System

### Dynamic Routing — Optuna-Optimized Parameters per Question Type

The 7 tunable parameters are optimized separately for each question type using a Random Forest proxy model trained on 175 labeled examples, with Optuna Bayesian search (2000 trials per type).

| Parameter | Description | action | attribute | spatial |
|-----------|-------------|--------|-----------|---------|
| `REALTIME_NEAR_N` | Frames around realtime anchor | 6 | 4 | 4 |
| `BIG_WINDOW_BASE_SIGMA` | Big window spread (seconds) | 91.72 | 138.01 | 85.76 |
| `DEDUP_THRESH_D` | Dedup threshold, big window | 0.9662 | 0.9824 | 0.9764 |
| `STU_MOVE_THRESH` | Motion detection sensitivity | 0.0312 | 0.0475 | 0.0452 |
| `MAX_N_SMALL` | Max frames per keyframe segment | 11 | 15 | 10 |
| `SEMANTIC_NEAR_N` | Frames dense-sampled near T_event | 6 | 6 | 6 |
| `DEDUP_THRESH_A` | Dedup threshold, keyframe window | 0.9775 | 0.9748 | 0.9817 |

### Fixed Parameters (13)

| Parameter | Value | Role |
|-----------|-------|------|
| `BIG_WINDOW_K` | 1.0 | Big window sigma multiplier |
| `BIG_WINDOW_EPS` | 0.01 | Sigma floor guard |
| `BIG_WINDOW_BASE_N` | 60 | Base frame budget for big window |
| `MIN_INTERVAL_BIG` | 0.5s | Minimum frame spacing, big window |
| `MIN_INTERVAL_SMALL` | 0.25s | Minimum frame spacing, small windows |
| `K_SIGMOID` | 6.0 | Sigmoid sharpness for keyframe window |
| `SEMANTIC_NEAR_INTERVAL` | 0.25s | Dense sampling interval near T_event |
| `REALTIME_NEAR_INTERVAL` | 0.25s | Dense sampling interval near realtime |
| `STU_AREA_THRESH` | 0.10 | Object area change threshold |
| `CLIP_DEDUP_THRESH` | 0.95 | Legacy dedup (compatibility) |
| `K` | 0.3 | TDC threshold coefficient |
| `ALPHA` | 1.0 | TDC r₀ coefficient |
| `BETA` | 1.0 | TDC λ coefficient |

---

## Parameter Optimization Pipeline

The final parameter configuration was derived through a multi-round optimization loop:

1. **Initial data collection** — Three parameter configurations (default, sparse, dense) were evaluated across the same question pool to expose the system to a wide range of parameter-outcome relationships.)

2. **Answer smoothing** — Binary correct/incorrect labels were replaced by per-option smoothed scores generated by Claude API (`score_smoothing.ipynb`). Scores are stored in `question_weights.json`.

3. **Random Forest proxy** — A `RandomForestRegressor` (200 trees, max depth 5) was trained on 175 labeled questions using 7 tunable + 13 fixed parameters + question type one-hot as features, and smoothed weighted score as the regression target (`Data_Augmentation_v3.ipynb`).

4. **Optuna search** — 2000 Bayesian trials per question type (action / attribute / spatial) on the Random Forest proxy, yielding type-specific optimal parameter sets.

5. **Iterative refinement** — Multiple rounds of evaluation → smoothing → retraining. The Optuna-derived parameters were locked into the final configuration embedded in Cell 1 of `ovo_stm_eval.ipynb`.

### Answer Smoothing Rules

Rather than binary scoring, each answer option is assigned a continuous score by Claude API based on its semantic relationship to the correct answer:

| Score | Criteria |
|-------|----------|
| **1.0** | Correct answer |
| **0.5** | Partially correct — describes the same action/object but with wrong details (e.g. wrong direction, wrong quantity) |
| **0.2** | Tangentially related — plausible in context but clearly incorrect |
| **0.0** | Completely unrelated to the correct answer |

This smoothed score becomes the regression target for the Random Forest, providing richer training signal for near-misses than binary 0/1 labels.

### Optuna Search Ranges

| Parameter | Range |
|-----------|-------|
| `REALTIME_NEAR_N` | 1 – 6 (int) |
| `BIG_WINDOW_BASE_SIGMA` | 50.0 – 150.0 |
| `DEDUP_THRESH_D` | 0.85 – 0.99 |
| `STU_MOVE_THRESH` | 0.005 – 0.05 |
| `MAX_N_SMALL` | 3 – 15 (int) |
| `SEMANTIC_NEAR_N` | 1 – 6 (int) |
| `DEDUP_THRESH_A` | 0.85 – 0.99 |

---

## Results

### Pre-Optimization Baseline (175 questions)

| Task | Questions | Accuracy | Weighted Score |
|------|-----------|----------|----------------|
| OCR | 20 | 90.0% | 0.900 |
| ASI | 18 | 77.8% | 0.789 |
| FPD | 18 | 72.2% | 0.778 |
| ACR | 20 | 60.0% | 0.695 |
| ATR | 20 | 50.0% | 0.510 |
| STU | 20 | 45.0% | 0.450 |
| OJR | 19 | 42.1% | 0.442 |
| EPM | 20 | 40.0% | 0.435 |
| HLD | 20 | 40.0% | 0.400 |
| **Overall** | **175** | **57.1%** | **0.597** |

### Final Evaluation (532 questions)

| Task | Questions | Accuracy | Weighted Score | Elapsed / Realtime |
|------|-----------|----------|----------------|-------------------|
| OCR | 60 | **95.0%** | 0.950 | 0.40× |
| ACR | 60 | 81.7% | 0.842 | 0.71× |
| ATR | 59 | 81.4% | 0.834 | 0.69× |
| OJR | 58 | 79.3% | 0.816 | 0.70× |
| ASI | 60 | 75.0% | 0.770 | 1.09× |
| STU | 58 | 72.4% | 0.719 | 0.60× |
| FPD | 57 | 70.2% | 0.711 | 0.71× |
| EPM | 60 | 51.7% | 0.538 | 0.58× |
| HLD | 60 | 46.7% | 0.473 | 0.86× |
| **Overall** | **532** | **72.6%** | **0.739** | — |

Elapsed / Realtime < 1.0 means the system processes faster than the clip's real-time duration. ASI > 1.0 due to shorter average clip length (150s) inflating the ratio.

### Improvement by Question Type

| q_type | Baseline | Final | Δ |
|--------|----------|-------|---|
| action | 69.8% | 74.2% | +4.4pp |
| attribute | 30.4% | 75.0% | **+44.6pp** |
| spatial | 41.3% | 64.6% | +23.3pp |

### Source Contribution (avg weighted, 532 questions)

| Source | Contribution |
|--------|-------------|
| B — Realtime Window | **0.257** |
| C — T_event Semantic | 0.160 |
| A — Keyframe Window | 0.151 |
| D — Big Window | 0.147 |

---

## Installation & Usage

### Running on Google Colab (L4 GPU)

**Step 1 — Mount Drive and install dependencies**

```python
from google.colab import drive
drive.mount('/content/drive')
```

```bash
!pip install -q openai-whisper
!pip install -q git+https://github.com/openai/CLIP.git
!pip install -q ultralytics
!pip install -q anthropic spacy scipy
!python -m spacy download en_core_web_sm
!apt-get install -y ffmpeg -q
```

**Step 2 — Upload notebooks**

Upload `ovo_stm_eval.ipynb`, `score_smoothing.ipynb`, and `Data_Augmentation_v3.ipynb` to your Colab session or Drive.

**Step 3 — Set your API key in Cell 1**

Open `ovo_stm_eval.ipynb` and replace `YOUR_API_KEY_HERE` in Cell 1 with your Anthropic API key. Set the OVO-Bench video path and question IDs in the same cell.

**Step 4 — Run cells sequentially**

Execute Cell 1 through Cell 10 in order. Cell 10 controls the batch run loop — set `run_ids` to the list of question IDs you want to evaluate.

**Step 5 — Parameter optimization (optional)**

To re-run the Random Forest + Optuna optimization on new data:
1. Run `score_smoothing.ipynb` on your result JSONs to generate updated `question_weights.json`
2. Run `Data_Augmentation_v3.ipynb` to train a new proxy model and search for optimal parameters
3. Copy the Optuna output parameters back into Cell 1 of `ovo_stm_eval.ipynb`

> ⚠️ Never hardcode your API key before pushing to a public repository. Use `YOUR_API_KEY_HERE` as a placeholder or load from environment variables.

---

## Repository Structure

```
├── ovo_stm_eval.ipynb          # Main evaluation pipeline (Optuna-optimized routing)
├── score_smoothing.ipynb       # Answer smoothing scorer (Claude API)
├── Data_Augmentation_v3.ipynb  # Random Forest proxy + Optuna optimization
├── question_weights.json       # Per-question smoothed scoring table
├── question_split.json         # Question ID allocation (training / validation)
└── results/
    ├── training_baseline.csv   # 175-question pre-optimization baseline
    └── validation_merged.json  # 532-question final evaluation
```

> **Note on question data**: Original questions sourced from OVO-Bench (CC BY-NC-SA 4.0).  
> `question_split.json` contains item IDs only. Full questions available at [JoeLeelyf/OVO-Bench](https://github.com/JoeLeelyf/OVO-Bench).

---

## Requirements

```
python >= 3.10
torch
clip (openai/CLIP)
openai-whisper
ultralytics          # YOLOv8 + YOLO-World
transformers         # CLIP-Seg
scikit-learn
optuna
anthropic
spacy + en_core_web_sm
opencv-python
scipy
```

---

## Related Work

This is Part 2 of the Comic Book Hypothesis (CBH) project, a training-free inference-time framework for video narrative understanding. Part 1 evaluates long-term memory on Video-MME (84 videos). This part evaluates short-term memory on OVO-Bench using a query-adaptive multi-source frame selection architecture.

---

## Citation

```bibtex
@software{lee2026cbh2,
  title  = {Query-Adaptive Multi-Source Frame Selection for Video Understanding},
  author = {Lee, C.J.},
  year   = {2026},
  note   = {Comic Book Hypothesis — Phase 2, independent research},
  url    = {https://github.com/Cheng-Je-Lee/video-understanding-multi-source-keyframe}
}
```

---

## License

MIT License
