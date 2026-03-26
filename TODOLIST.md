> This file is just for me to keep all tasks in front of me; so please just ignore it :) T7yati Zohoory ✌️

# Task 1

We’ll use **HuggingFace SST-2** and keep the **same project skeleton** as the foundation.

So from now on, the working path is:

```text
SST-2 text
→ pretrained Transformer
→ saved embeddings
→ CATS encoder
→ classifier
→ experiments
```

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

So the first milestone is only this:

## Milestone 1

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

# Task 2
