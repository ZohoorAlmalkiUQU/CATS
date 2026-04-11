
# CATS: Context-Aware Token-to-Spike Framework for Spiking Neural Networks

CATS (**C**ontext-**A**ware **T**oken-to-**S**pike) is a modular research framework for converting **precomputed embedding sequences** into structured **spike-based representations** using routing-guided adaptive spiking encoders.

The framework is designed for **modality-agnostic processing**, **controlled experimentation**, and **high extensibility** across routing strategies and neuron dynamics.

---

## Key Idea

```text
[Any Embedding Backbone (Text / Image / Audio)]
                    ↓
        CARSON Routing (ANN domain)
                    ↓
     Learnable & Adaptive Spiking Encoder (LIF)
                    ↓
        Spike-Based Representation
                    ↓
          Classification Head (ANN domain)
```

CATS **decouples representation learning from spike encoding**, allowing researchers to study spiking behavior independently from embedding generation.

---

## Core Contributions

### 1. Routing-Guided Token-to-Spike Encoding

CATS introduces a **routing stage (ANN domain)** before spiking computation:

- Assigns tokens/features to functional neuron groups
- Learns structured information flow
- Acts as a **structural inductive bias**

---

### 2. CARSON Routing (Primary Method)

> **C**apsule-**A**ware **R**outing for **S**piking-**O**riented **N**etworks

CARSON is a **capsule-inspired, context-aware routing mechanism** that operates in the embedding domain prior to spiking computation.

Unlike classical capsule networks, CARSON does not model explicit pose transformations. Instead, it focuses on **structured token grouping and context-driven routing**, making it suitable for modality-agnostic embedding sequences.

#### Core Mechanism

CARSON performs routing in two stages:

##### 1. Iterative Context Routing

- Computes a **global context vector** from token representations
- Refines this context through **iterative routing**
- Uses soft attention-like weighting over tokens
- Produces:
  - context vector
  - token-level routing confidence

This stage acts as a **routing-by-agreement analogue**, where tokens contribute to a shared representation based on relevance.

---

##### 2. Group Assignment (Excitatory / Inhibitory)

Using:

- specialized token features:
  - excitatory features
  - inhibitory features
- shared context vector

CARSON predicts **soft routing weights** over neuron groups:

[B, T, 2] → (excitatory, inhibitory)

This results in:

- interpretable group assignments
- structured competition between neuron populations
- biologically-inspired routing dynamics

---

#### Key Properties

- **Capsule-inspired**
  - iterative refinement
  - soft assignment
  - group-level representation

- **Context-aware**
  - routing depends on global sequence structure

- **Specialized pathways**
  - separate excitatory and inhibitory feature projections

- **Temperature-controlled routing**
  - controls sharpness of assignments

- **Residual integration**
  - preserves original embedding information

- **Masked sequence support**
  - fully compatible with variable-length inputs

---

#### Output Signals

CARSON produces:

- `routing_weights` → soft group assignment
- `routing_logits` → pre-softmax scores
- `context_vector` → global representation
- `routing_confidence` → token importance
- `exc_features`, `inh_features`, `shared_features`
- `routed_x` → context-enhanced embeddings

---

#### Baselines for Comparison

- `identity` → no routing
- `linear` → learned projection
- `rule_based` → deterministic grouping

---

#### Design Philosophy

> CARSON is not a full capsule network, but a **capsule-aware routing mechanism** adapted for embedding-to-spike transformation.

It preserves key capsule principles:

- iterative refinement  
- soft assignment  
- structured grouping  

while remaining:

- lightweight  
- modular  
- compatible with SNN pipelines

### 3. Adaptive Spiking Neuron Dynamics (LIF)

All neurons follow a **unified LIF model**, with configurable behavior:

#### Supported Modes

- Fixed parameters (baseline)
- Learnable parameters:
  - membrane time constant `τ`
  - decay factor
- Adaptive parameters:
  - dynamic threshold (based on firing activity)

#### Design Principle

> Keep neuron model fixed → vary *parameters only*  
→ ensures clean, reviewer-friendly ablations

---

### 4. Excitatory / Inhibitory Architecture

CATS explicitly models:

- Excitatory neurons
- Inhibitory neurons

With:

- configurable ratio (`excitatory_ratio`)
- separate LIF dynamics
- routing-based assignment

This enables:

- competition
- balance control
- biologically-inspired dynamics

---

### 5. Positional Encoding (NEW)

CATS supports **modular positional encoding**:

- Designed for **modality-agnostic use**
- Current support:
  - RoPE (Rotary Positional Encoding)

Key properties:

- Works across text, vision, and audio embeddings
- Fully optional (controlled via config)
- Plug-and-play design

---

## Project Architecture

```text
CATS/
│
├── src/cats/
│   ├── model.py                 # Main model (CATSClassifier)
│   │
│   ├── encoder/
│   │   ├── core.py             # Core encoder logic
│   │   │
│   │   ├── routing/
│   │   │   ├── base.py
│   │   │   ├── identity.py
│   │   │   ├── linear.py
│   │   │   └── carson.py
│   │   │
│   │   ├── spiking/
│   │   │   ├── base.py
│   │   │   ├── lif.py
│   │   │   └── params.py
│   │   │
│   │   └── position/
│   │       ├── base.py
│   │       └── rope.py
│   │
│   ├── heads/
│   │   └── classifier.py
│   │
│   ├── data/
│   │   ├── dataset.py
│   │   └── collate.py
│   │
│   └── utils/
│       ├── config.py
│       ├── logger.py
│       ├── metrics.py
│       └── seed.py
│
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   ├── prepare_embeddings.py
│   └── prepare_raw_dataset.py
│
├── configs/
│   └── *.yaml                  # Experiment configs
│
├── experiments/
├── logs/
├── results/
├── checkpoints/
├── tests/
│
├── pyproject.toml
├── README.md
└── LICENSE
```

---

## Configuration-Driven Design

All experiments are controlled via YAML configs.

Example:

```yaml
experiment:
  dataset_name: sst2
  model_name: cats
  routing_type: carson
  run_name: run_002
  seed: 42

data:
  processed_root: data/processed
  train_split: train
  val_split: validation
  test_split: test
  has_test: true
  num_workers: 0
  pin_memory: true

model:
  class_path: cats.model.CATSClassifier
  embedding_dim: 768
  hidden_dim: 256
  num_classes: 2
  excitatory_ratio: 0.5
  num_groups: 2
  kwargs: {}

routing:
  kwargs:
    embedding_dim: 768
    hidden_dim: 256
    num_groups: 2
    num_iterations: 3
    dropout: 0.1
    temperature: 1.2
    use_residual: true
    use_layernorm: true

lif_exc:
  num_groups: 1
  reset_to_zero: true
  detach_reset: false

  tau:
    learnable: true
    mode: shared
    min: 14.0
    max: 28.0
    init: 20.0

  threshold:
    learnable: true
    mode: shared
    min: 0.35
    max: 1.2
    init: 0.65

  adaptive_threshold:
    enabled: true
    mode: shared
    init: 0.0
    decay: 0.97
    spike_increment: 0.05
    min: 0.0
    max: 2.0
    detach_spikes: true

lif_inh:
  num_groups: 1
  reset_to_zero: true
  detach_reset: false

  tau:
    learnable: true
    mode: shared
    min: 6.0
    max: 14.0
    init: 9.0

  threshold:
    learnable: true
    mode: shared
    min: 0.35
    max: 0.9
    init: 0.55

  adaptive_threshold:
    enabled: true
    mode: shared
    init: 0.0
    decay: 0.96
    spike_increment: 0.08
    min: 0.0
    max: 2.0
    detach_spikes: true

classifier:
  input_dim: 256
  kwargs:
    hidden_dim: null
    dropout: 0.0
    use_layernorm: false

training:
  seed: 42
  deterministic: false
  batch_size: 32
  epochs: 10

  optimizer: adamw
  lr: 1e-4
  weight_decay: 1e-5

  criterion: cross_entropy
  grad_clip_norm: 1.0

  early_stopping:
    enabled: true
    patience: 3
    min_delta: 0.001
    monitor: val_f1
    mode: max

  lambda_balance: 0.2
  lambda_entropy: 0.001

logging:
  base_dir: logs

checkpoints:
  base_dir: checkpoints
  best_name_template: best_{run_name}.pt
  latest_name_template: latest_{run_name}.pt

runtime:
  device: cuda
````

---

## Training Pipeline

CATS supports:

- Full training / validation / test loops
- Metrics:

  - Accuracy
  - Precision / Recall
  - F1-score
- Spiking metrics:

  - firing rate
  - spikes per token
- Routing diagnostics:

  - entropy
  - dominance
  - confidence

---

## Experimental Philosophy

CATS is built for **clean, reviewer-grade experimentation**:

### Supported Studies

- Routing:
  - CARSON vs Linear vs Identity
- Spiking:
  - Fixed vs Learnable vs Adaptive LIF
- Structure:
  - With vs Without positional encoding
- Biology-inspired:
  - Excitatory / Inhibitory balance

---

## Design Principles

### 1. Modularity

Every component is replaceable:

- routing
- neuron dynamics
- positional encoding

---

### 2. Configurability

All behaviors controlled via config => no hardcoding

---

### 3. Extensibility

You can easily add:

- new routing algorithm
- new neuron model
- new encoding strategy

---

### 4. Modality-Agnostic

CATS works with:

- text embeddings (BERT)
- image embeddings (ViT)
- audio embeddings

---

## Out of Scope

CATS does NOT handle:

- raw data preprocessing
- tokenization
- training embedding backbones

---

## Research Use Cases

- Spike encoding from embeddings
- Routing in SNNs
- ANN → SNN hybrid systems
- Neuromorphic-inspired architectures

---

## Status

Active research project.

Focus:

- correctness
- interpretability
- reproducibility

Not optimized for production.

---

## Citation

Will be added upon publication.

---

## License

See LICENSE file.
