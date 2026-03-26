> This file is just for me to keep all tasks in front of me; so please just ignore it :) T7yati Zohoory ✌️

# Task 1

We will use **HuggingFace SST-2** and keep the **same project skeleton** as the foundation.

The goal of this task is to **verify that the end-to-end pipeline works**, without introducing CATS yet.

---

## Goal of Task 1

```text
SST-2 text
→ pretrained Transformer
→ saved embeddings
→ simple encoder
→ classifier
```

✔ Ensure everything runs end-to-end
✔ Validate data flow and shapes
✔ No routing, no spiking yet

---


```
CATS/
├── src/
│   └── cats/
│       ├── encoder/
│       │   ├── core.py
│       │   ├── routing/
│       │   │   ├── base.py
│       │   │   ├── identity.py
│       │   │   └── linear_router.py
│       │   └── spiking/
│       │       ├── lif.py
│       │       └── population.py
│       │
│       ├── backbones/ 
│       │   └── transformer.py
│       │
│       ├── heads/
│       │   └── classifier.py
│       │
│       ├── data/
│       │   ├── dataset.py
│       │   └── collate.py
│       │
│       └── utils/
│           ├── seed.py
│           └── metrics.py
│
├── data/                         # ❗ datasets (ignored by git)
│   ├── raw/
│   │   └── sst2/
│   │       ├── train.csv
│   │       ├── validation.csv
│   │       └── test.csv
│   │
│   └── processed/
│       └── sst2/
│           ├── train.pt
│           ├── validation.pt
│           └── test.pt
│
├── configs/
│   ├── base.yaml
│   ├── no_routing.yaml
│   └── linear_routing.yaml
│
├── scripts/
│   ├── prepare_sst2.py
│   ├── extract_embeddings.py
│   ├── train_baseline.py
│   ├── train.py
│   └── evaluate.py
│
├── tests/
│   └── dataset_test.py
│
├── .gitignore
├── pyproject.toml
├── README.md
└── LICENSE
```

## What we do first

Before coding CATS itself, we need the **data pipeline** to be correct.


## Milestone 1 — Data & Representation

* download SST-2
* inspect splits
* tokenize
* extract embeddings from a pretrained Transformer
* save them to disk in a format CATS can load later

That means the first files to implement are:

* `scripts/prepare_sst2.py`
* `scripts/extract_embeddings.py`
* `src/cats/data/dataset.py`
* `src/cats/data/collate.py`

Not the spiking files yet.

---

### Recommended first design choice

Use:

* **dataset:** SST-2
* **backbone:** `bert-base-uncased`
* **saved representation:** last hidden states
* **input per sample:** `embedding [T, D] + attention_mask [T]`
* **label:** `0` or `1`

This is the cleanest starting point.

Why this is a good first setup:
- SST-2 is binary classification, so the task is simple
- sentences are short, so sequence lengths stay manageable
- debugging is easier because labels are clear and the pipeline is small
- experiments run relatively fast

---

### Save format

One saved split file:

```python
{
    "split": str,
    "model_name": str,
    "max_length": int,
    "embeddings": tensor[N, T, D],
    "attention_mask": tensor[N, T],
    "labels": tensor[N],          # if exists
    "sentences": List[str]        # optional
}
```

---

### Dataset return format

```python
One sample returned by dataset.py
{
    "embedding": tensor[T, D],
    "attention_mask": tensor[T],
    "label": int
}
```

---

### Folder layout for data

Use something like:

```text
CATS/
├── src/
│   └── cats/
│       ├── data/
│       │   ├── dataset.py
│       │   └── collate.py
│
├── data/
│   ├── raw/
│   │   └── sst2/
│   └── processed/
│       └── sst2/
│           ├── train.pt
│           ├── validation.pt
│           └── test.pt
```

---

### Exact order to begin

#### Step 1

Create the folders:

* `data/raw/sst2/`
* `data/processed/sst2/`

#### Step 2

Write `scripts/prepare_sst2.py`
This script should:

* load SST-2 from HuggingFace
* inspect split sizes
* maybe print a few samples
* optionally save raw text/labels locally

#### Step 3

Write `scripts/extract_embeddings.py`
This script should:

* load SST-2
* load `bert-base-uncased`
* tokenize each sentence
* extract hidden states
* save embeddings and labels into `.pt` files

#### Step 4

Write dataset loader
`src/cats/data/dataset.py`
This should load saved `.pt` files and return:

* embedding
* label

#### Step 5

Write collate function

Since embeddings are already padded to a fixed max_length during extraction,
the collate function should:

* stack embeddings into [B, T, D]
* stack attention masks into [B, T]
* stack labels into [B]

The attention mask will be useful later for routing and spiking.

---

## Milestone 2 — End-to-End Baseline

For **Milestone 2**, our goal is not real CATS routing yet. The objective is to verify that the baseline end-to-end pipeline works correctly using precomputed transformer embeddings.

```text
saved token embeddings
→ minimal encoder / identity router
→ pooling
→ classifier
→ train / evaluate
```

So the files to focus on now are:

```text
src/cats/encoder/core.py
src/cats/encoder/routing/base.py
src/cats/encoder/routing/identity.py
src/cats/heads/classifier.py
scripts/train_baseline.py
scripts/evaluate.py
configs/base.yaml
configs/no_routing.yaml
```

### What “verify full pipeline runs correctly” really means

For this milestone, success is not about best accuracy.
Success means all of these are true:

* train/val/test `.pt` files load correctly
* embedding and attention-mask dimensions are consistent
* labels align with samples where labels are available
* dataloader returns expected batch shapes
* forward pass works
* loss decreases at least somewhat
* validation accuracy is above random
* test split is handled correctly, including inference-only cases with invalid or missing labels
* checkpoint save/load works

That is the real milestone objective.

---

### Suggested implementation order

Do it in this exact order.

#### Step 1

Finish `dataset.py`

* load `.pt`
* return sample dictionaries containing:

  * `embedding`
  * `attention_mask`
  * `label` when available

#### Step 2

Finish `collate.py`

* stack token embeddings into `[B, T, D]`
* stack attention masks into `[B, T]`
* stack labels into `[B]` when available

#### Step 3

Implement `identity.py`

* pure pass-through router

#### Step 4

Implement `core.py`

* wrapper around router
* include pooling to convert token-level representations into sentence-level representations

#### Step 5

Implement `classifier.py`

* one linear layer first

#### Step 6

Implement `train_baseline.py`

* train loop
* validation loop
* checkpoint saving

#### Step 7

Implement `evaluate.py`

* load checkpoint
* evaluate on labeled splits
* handle inference-only test splits gracefully

---

### Strong recommendation about naming

To keep the repo clean and future-proof:

* `IdentityRouter` = baseline no-routing placeholder
* `CATSEncoder` = generic encoder wrapper
* `ClassifierHead` = generic classification head
* `BaselineModel` = Milestone 2 combined model

This will scale nicely when we later add:

* `LinearRouter`
* spiking population encoding
* LIF dynamics
* full `train.py`

---

# Task 2

```text
saved embeddings
→ CATS encoder (routing + spiking)
→ classifier
→ experiments
```

👉 This is where the real model (CATS) is introduced.

---
