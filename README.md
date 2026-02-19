# CATS: Routing-Guided Adaptive Spiking Encoder for Embedding Sequences

CATS (**C**ontext-**A**ware **T**oken-to-**S**pike) is a research-oriented framework for converting continuous embedding sequences into structured spike-based representations using **routing-guided adaptive spiking encoders**.

The core contribution of CATS is a **modular spiking encoder** that operates on pre-computed embeddings (e.g., from Transformers), rather than on raw inputs.
This design allows CATS to remain agnostic to tokenization, embedding backbones, and input modalities.

---

## Motivation

Contemporary neural systems increasingly depend on rich embedding representations produced by large-scale models such as Transformers.
In contrast, most existing spiking neural network (SNN) approaches remain tightly coupled to raw sensory inputs or rely on fixed, non-adaptive spike encoding schemes, limiting flexibility and reuse.

CATS addresses this gap by introducing a **routing-guided adaptive spiking encoder** that cleanly decouples:

* embedding generation,
* routing and structural organization, and
* spike-based computation.

The proposed encoder:

* operates directly on arbitrary embedding sequences,
* employs learnable routing mechanisms to regulate information flow,
* converts embeddings into spike-based representations via adaptive neuron dynamics.

---

## Key Idea

```
    [Any Transformer / Embedding Source]
                ↓
    Routing-Guided Adaptive Spiking Encoder
                ↓
    Spike-Based Representation / Decision
```

The encoder itself is the main contribution.
Embedding models are treated as interchangeable input sources.

---

## Core Components

### 1. Routing-Guided Encoder

CATS introduces routing mechanisms that guide how embedding activations are assigned to **functional neuron groups** before spiking computation.

Routing operates in the ANN domain and introduces a structural inductive bias prior to spike generation.

---

### 2. CARSON Routing

CATS includes **CARSON**:

> **C**apsule-**A**ware **R**outing for **S**piking-**O**riented **N**etworks

CARSON is a capsule-inspired routing mechanism enabling structured, learnable assignment of embedding dimensions or tokens to spiking neuron groups.

It is implemented independently of the embedding backbone and can be compared against standard routing baselines.

---

### 3. Adaptive Spiking Neuron Dynamics

The spiking layer constitutes the core of the encoder, transforming routed embeddings into temporal spike patterns.

All neurons follow a unified Leaky Integrate-and-Fire (LIF) formulation to maintain architectural consistency and controlled experimentation.

Rather than introducing multiple neuron model families, CATS isolates the effect of learnable intrinsic parameters within a unified LIF formulation to ensure controlled and interpretable experimentation.

The framework supports:

Fixed-parameter LIF (static intrinsic dynamics)

Learnable/adaptive-parameter LIF (e.g., learnable τ, adaptive thresholds, learnable inhibitory strength)

This design enables controlled comparison between static and adaptive spike dynamics within a single neuron model class, avoiding confounding effects from changing neuron formulations.

Note: Alternative neuron models (e.g., AdEx) are maintained only for extensibility and are not part of the primary experimental scope.

---

### 4. Excitatory / Inhibitory Organization

The spiking encoder explicitly models excitatory and inhibitory neuron populations.

Routing determines how embedding dimensions or tokens are assigned to these subpopulations, enabling competitive and balanced spike dynamics.

---

## Project Structure

```
CATS/
│
├── src/
│   └── cats/
│       ├── encoder/
│       │   ├── core.py
│       │   ├── routing/
│       │   │   ├── carson.py
│       │   │   ├── capsule_routing.py
│       │   │   └── inhibitory.py
│       │   └── spiking/
│       │       ├── lif.py
│       │       └── adex.py # Optional / future extension
│       │
│       └── backbones/
│           └── transformer.py
│
├── configs/          # Experiment configurations
├── scripts/          # Training & evaluation entry points
├── experiments/      # Research experiments and analysis
├── tests/            # Unit tests for framework components
│
├── pyproject.toml
├── LICENSE
└── README.md
```

### Directory Responsibilities

* `src/cats/` — Core framework implementation
* `configs/` — Reproducible experiment configurations
* `scripts/` — Training and evaluation entry points
* `experiments/` — Research experiments, ablations, and analysis
* `tests/` — Unit tests ensuring framework stability

This structure cleanly separates reusable components from research-specific experimentation.

---

## Scope and Responsibility

### CATS is responsible for:

* Structured embedding-to-spike transformation
* Routing mechanisms and grouping strategies
* Excitatory/Inhibitory assignment
* Adaptive spiking neuron dynamics
* Supporting controlled ablation studies

### CATS is *not* responsible for:

* Tokenization strategies
* Training of embedding backbones
* Intrinsic quality of embeddings

This separation ensures fair evaluation and avoids confounding architectural factors.

---

## Experimental Design Philosophy

CATS is built to support systematic experimentation, including:

* Fixed vs learnable/adaptive LIF parameters
* CARSON vs routing baselines
* No routing vs structured routing

Configurations are defined under `configs/`, enabling reproducible research workflows.

---

## Research Use Case

CATS is intended for:

* Studying embedding-to-spike encoding mechanisms
* Investigating routing as a structural prior in SNNs
* Exploring adaptive spike dynamics
* Supporting hybrid ANN–SNN systems
* Serving as a modular research platform for neuromorphic-inspired modeling

---

## Status

CATS is under active research development.
The framework prioritizes modularity, clarity, and reproducibility rather than production optimization.

---

## Citation

A formal citation entry will be added upon publication of the associated manuscript.

---

## License

This project is released under the terms of the included LICENSE file.

---