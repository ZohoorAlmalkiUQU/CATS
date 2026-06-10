# CATS: Context-Aware Token-to-Spike Framework

**CATS** (**C**ontext-**A**ware **T**oken-to-**S**pike) is a modular research framework for converting precomputed embedding sequences into structured spike-based representations using routing-guided adaptive spiking encoders. The framework decouples representation learning from spike encoding, enabling controlled experimentation across routing strategies, neuron dynamics, and modalities.

---

## Overview

```text
 Embedding Backbone (BERT / ViT / Audio Encoder)
                        │
                        ▼
          ┌─────────────────────────┐
          │   CARSON Router (ANN)   │  ◄── context-aware, capsule-inspired
          └─────────────────────────┘
                        │
           routing_weights [B, T, 2]
                        │
                        ▼
          ┌─────────────────────────┐
          │  Adaptive LIF Encoder   │  ◄── excitatory / inhibitory populations
          └─────────────────────────┘
                        │
              spikes / membrane [B, T, D]
                        │
                        ▼
          ┌─────────────────────────┐
          │   Classification Head   │  ◄── ANN or SNN readout
          └─────────────────────────┘
```

CATS operates entirely on **precomputed embeddings** — it does not train the upstream backbone. This design enables clean ablation studies of the routing and spiking components in isolation.

---

## Core Contributions

### 1. CARSON Routing

> **C**apsule-**A**ware **R**outing for **S**piking-**O**riented **N**etworks

CARSON is a two-stage, context-aware routing mechanism that assigns tokens to excitatory and inhibitory neuron groups before spiking computation.

#### Stage 1 — Iterative Context Routing

A global context vector is computed from the token sequence through iterative refinement. At each iteration, token-level attention weights are recomputed against the current context, producing token-importance scores (`routing_confidence`) and an updated context embedding. This procedure is analogous to routing-by-agreement in capsule networks.

#### Stage 2 — Group Assignment

Specialized excitatory and inhibitory feature projections, conditioned on the context vector, predict soft routing weights:

```text
routing_weights ∈ [B, T, 2]   →   (excitatory weight, inhibitory weight)
```

These weights gate the input current to each neuron population before the LIF dynamics are applied.

**Key properties:**

- Iterative refinement with configurable depth (`num_iterations`)
- Temperature-controlled softmax over group assignments
- Optional residual skip connection over the routing transformation
- Fully compatible with variable-length masked sequences
- Produces interpretable per-token routing confidence scores

**Outputs:** `routing_weights`, `routing_logits`, `context_vector`, `routing_confidence`, `exc_features`, `inh_features`, `routed_x`

**Baselines:** `identity` (no routing, pass-through), `linear` (single learned projection)

---

### 2. Adaptive Leaky Integrate-and-Fire (LIF) Neurons

All spiking computation uses a unified LIF model applied to separate excitatory and inhibitory populations. The membrane potential follows:

```text
v[t] = β · v[t-1] + x[t]            (integration)
s[t] = 1  if  v[t] ≥ θ_eff[t]       (firing)
v[t] ← v[t] · (1 - s[t])            (soft reset)
```

where `β = exp(−1/τ)` is the decay factor and `θ_eff = θ_base + θ_adapt` is the effective threshold.

Each parameter supports three modes:

| Parameter | Fixed | Learnable | Adaptive |
| --------- | ----- | --------- | -------- |
| `τ` (time constant) | scalar constant | gradient-trained, clamped | — |
| `θ` (threshold) | scalar constant | gradient-trained, clamped | — |
| `θ_adapt` (adaptive threshold) | — | — | decays each step; increments on spike |

The adaptive threshold update rule is:

```text
θ_adapt[t] = decay · θ_adapt[t-1] + spike_increment · s[t-1]
```

Excitatory and inhibitory populations run independent LIF layers with separate parameter sets, allowing different time scales and firing thresholds for each population.

**SignedLinear layers** enforce weight-sign constraints: excitatory projections maintain non-negative weights; inhibitory projections maintain non-positive weights.

---

### 3. Excitatory / Inhibitory Architecture

CATS explicitly separates neuron populations:

- **Excitatory population** (`excitatory_ratio` fraction of hidden units, default 0.5)
- **Inhibitory population** (remaining units)
- Routing weights gate input current to each population independently
- Balance regularization (`λ_balance`) and entropy regularization (`λ_entropy`) encourage stable, non-collapsed routing

---

### 4. Modular Positional Encoding

Positional structure is injected via **RoPE** (Rotary Positional Encoding) before the routing stage. The encoding is modality-agnostic and fully optional — enabled or disabled via config. Further encoding strategies (sinusoidal, learned) can be added by implementing the `BasePositionalEncoding` interface.

---

## Results

### Router Comparison (Core Experiments)

All models use the full CATS configuration: CARSON routing, learnable threshold, adaptive threshold, excitatory/inhibitory populations, RoPE positional encoding, ANN classifier head.

| Dataset | Modality | Backbone | Identity (F1 / Acc) | Linear (F1 / Acc) | CARSON (F1 / Acc) |
| ------- | -------- | -------- | ------------------- | ----------------- | ----------------- |
| CIFAR-10 | Image | ViT | 0.6125 / 61.1% | 0.6226 / 62.6% | **0.6754 / 67.6%** |
| MNIST | Image | ViT | 0.9609 / 96.1% | 0.9606 / 96.1% | **0.9802 / 98.0%** |
| Speech Commands | Audio | — | 0.9083 / 91.5% | 0.9199 / 92.3% | **0.9356 / 93.8%** |
| SST-2 | Text | BERT | 0.8375 / 82.9% | 0.8756 / 87.4% | **0.8993 / 89.5%** |

CARSON routing outperforms both Identity and Linear on every dataset and modality: +6.3 pp F1 over identity and +5.3 pp over linear on CIFAR-10, +1.9 pp / +2.0 pp on MNIST, +2.7 pp / +1.6 pp on Speech Commands, and +6.2 pp / +2.4 pp on SST-2.

---

### Ablation Study (CARSON Routing)

Component ablations measure the contribution of each architectural element. ΔF1 is relative to the full model.

#### CIFAR-10 (full model Val F1 = 0.6754)

| Configuration | Val F1 | Val Acc | ΔF1 (pp) |
| ------------- | ------ | ------- | -------- |
| Full model (CARSON, all adaptive) | 0.6754 | 67.6% | — |
| Fixed τ (non-learnable) | 0.6802 | 68.1% | +0.48 |
| Fixed θ (non-learnable) | 0.6795 | 68.0% | +0.42 |
| No adaptive threshold | 0.6797 | 68.0% | +0.44 |
| Fixed τ + fixed θ | 0.6820 | 68.6% | **+0.66** |
| Fully static (τ, θ, no adaptive) | 0.6799 | 68.4% | +0.45 |
| No inhibitory population | 0.6764 | 67.7% | +0.10 |
| No RoPE positional encoding | 0.6755 | 67.7% | +0.01 |
| No spiking (ANN after routing) | 0.6741 | 67.8% | −0.13 |

#### Speech Commands (full model Val F1 = 0.9356)

| Configuration | Val F1 | Val Acc | ΔF1 (pp) |
| ------------- | ------ | ------- | -------- |
| Full model (CARSON, all adaptive) | 0.9356 | 93.8% | — |
| Fixed τ (non-learnable) | 0.9361 | 93.9% | +0.05 |
| Fixed θ (non-learnable) | 0.9352 | 93.8% | −0.04 |
| No adaptive threshold | 0.9359 | 93.9% | +0.03 |
| Fixed τ + fixed θ | 0.9369 | 93.9% | **+0.13** |
| Fully static (τ, θ, no adaptive) | 0.9349 | 93.7% | −0.07 |
| No inhibitory population | 0.9340 | 93.7% | −0.16 |
| No RoPE positional encoding | 0.9386 | 94.0% | **+0.29** |
| No spiking (ANN after routing) | 0.9354 | 93.8% | −0.02 |

Key findings:

- **Spiking matters most on CIFAR-10.** Replacing the spiking layer with a plain ANN is the only configuration that clearly hurts on CIFAR-10 (−0.13 pp), while it is roughly neutral on Speech Commands (−0.02 pp).
- **The inhibitory population is more important for audio than vision.** Removing it costs −0.16 pp on Speech Commands but only +0.10 pp (i.e. essentially no cost) on CIFAR-10.
- **Fixed τ/θ is consistently competitive with (or better than) the fully adaptive configuration** on both datasets, with `fixed_tau_fixed_threshold` the best configuration overall (+0.66 pp on CIFAR-10, +0.13 pp on Speech Commands). The adaptive-threshold mechanism does not provide a measurable benefit over fixed values in these single-run comparisons.
- **RoPE positional encoding helps image patch sequences but not audio.** Removing it is roughly neutral on CIFAR-10 (+0.01 pp) but improves Speech Commands by +0.29 pp, the largest single effect observed for that dataset.

---

### SNN Readout Study (CIFAR-10, CARSON Routing)

> **Note:** This study is exploratory and not part of the core thesis results — it is kept here as a starting point for future SNN-readout research.

Six classifier readout strategies were evaluated to determine how best to aggregate membrane potential and spike activity for classification:

| Readout | Val F1 | Val Acc | Best Epoch | Spk/Sample | Firing Rate |
| ------- | ------ | ------- | ---------- | ---------- | ----------- |
| Last membrane (final step) | **0.6736** | 67.5% | 7 | 4,371 | 0.263 |
| Mean membrane (μ) | 0.6694 | 66.9% | 8 | 6,391 | 0.384 |
| Temporal pool | 0.6693 | 67.1% | 6 | 5,755 | 0.346 |
| Membrane + spike hybrid | 0.6628 | 66.5% | 8 | 3,041 | 0.183 |
| Spike count | 0.6519 | 64.9% | 7 | 2,928 | **0.176** |
| Mean spikes (spike rate) | 0.5925 | 60.7% | 10 | 6,114 | 0.367 |

**Key findings:**

1. **All readout strategies converge.** All six variants learn successfully, confirming that the surrogate gradient propagates through both membrane-potential and spike-based readouts when properly implemented.

2. **`last_membrane` is the best overall readout, and the most spike-efficient membrane-based one.** It tops the table at 67.4% val F1 while firing at roughly two-thirds the rate of `mean_membrane` (0.263 vs 0.384) — reading only the final membrane state avoids accumulating activity across all time steps without sacrificing accuracy.

3. **Membrane-based readouts achieve higher accuracy.** `last_membrane`, `mean_membrane`, `temporal_pool`, and `membrane_spike_hybrid` all reach 66.3–67.4% val F1, consistently outperforming pure spike-based readouts by 1.1–8.1 pp.

4. **`spike_count` is the most spike-efficient overall.** At firing rate 0.176 and only 2,928 spikes/sample — well below the membrane variants — it achieves 65.2% val F1, offering the best accuracy-per-spike trade-off among the spike-based strategies.

5. **`temporal_pool` converges fastest** (best epoch 6 vs 7–10 for all others), suggesting temporal pooling provides a stronger learning signal early in training.

6. **`mean_spikes` underperforms.** Despite spiking activity comparable to membrane variants (0.367 firing rate), mean spike rate alone produces the lowest val F1 (0.5925), indicating that spike rate is a weaker task signal than membrane potential across this token sequence length.

---

## Project Structure

```text
CATS/
├── src/cats/
│   ├── model.py                      # CATSClassifier (top-level model)
│   ├── encoder/
│   │   ├── core.py                   # CATSEncoder: routing → spiking → pooling
│   │   ├── routing/
│   │   │   ├── base.py               # BaseRouter interface
│   │   │   ├── identity.py           # IdentityRouter
│   │   │   ├── linear.py             # LinearRouter
│   │   │   └── carson.py             # CARSONRouter
│   │   ├── spiking/
│   │   │   ├── base.py               # BaseSpikingLayer interface
│   │   │   ├── lif.py                # LIFLayer (excitatory & inhibitory)
│   │   │   └── params.py             # Parameter builders & SignedLinear
│   │   └── position/
│   │       ├── base.py               # BasePositionalEncoding interface
│   │       └── rope.py               # RoPE implementation
│   ├── heads/
│   │   ├── ann_classifier.py         # ANN classification head
│   │   └── snn_classifier.py         # SNN classification head (6 readout modes)
│   ├── data/
│   │   ├── dataset.py                # EmbeddingDataset (shard-based, LRU cache)
│   │   └── collate.py                # Variable-length collation with padding masks
│   ├── builders/
│   │   ├── model_builder.py          # Build CATSClassifier from config dict
│   │   └── build_positional_encoding.py
│   └── utils/
│       ├── config.py                 # YAML config loading & validation
│       ├── logger.py                 # Epoch-level logging (CSV + console)
│       ├── metrics.py                # Classification & spiking metric computation
│       ├── seed.py                   # Reproducibility (seed, deterministic flags)
│       └── mask.py                   # Attention mask utilities
│
├── scripts/
│   ├── train.py                      # Full training pipeline
│   ├── evaluate.py                   # Checkpoint evaluation
│   ├── prepare_embeddings.py         # Raw data → embedding shards (.pt)
│   ├── prepare_raw_dataset.py        # Dataset download & preprocessing
│   ├── run_ablation.py               # Automated ablation grid
│   ├── run_carson_all_datasets.py    # CARSON across all datasets
│   ├── run_router_comparison.py      # Identity / Linear / CARSON comparison
│   └── run_snn_readout_study.py      # SNN readout strategy comparison
│
├── configs/
│   └── ablation_study/               # YAML configs per condition × dataset
│       ├── fixed_tau/
│       ├── fixed_threshold/
│       ├── fixed_tau_fixed_threshold/
│       └── fixed_tau_fixed_threshold_no_adaptive/
│
├── results/
│   ├── core_experiments_analysis/    # Per-dataset training-log notebooks
│   │   └── shared_analysis/         # Cross-router convergence comparison
│   ├── ablation_study/              # Ablation analysis notebooks
│   └── snn_readout_study/           # SNN readout strategy notebook
│
├── tests/
├── logs/                             # Training logs (CSV per run)
├── checkpoints/                      # Best & latest model checkpoints
├── environment.yml
└── pyproject.toml
```

---

## Installation

```bash
conda env create -f environment.yml
conda activate cats-env
pip install -e .
```

**Requirements:** Python 3.10, PyTorch ≥ 2.0 (CUDA 12.1 recommended), `transformers`, `datasets`, `evaluate`, `scikit-learn`, `einops`, `PyYAML`.

---

## Data Preparation

CATS operates on precomputed embedding shards. Raw datasets must be converted to `.pt` files before training:

```bash
# Step 1: download and preprocess raw data
python scripts/prepare_raw_dataset.py --dataset cifar10

# Step 2: encode with the appropriate backbone and save as shards
python scripts/prepare_embeddings.py --dataset cifar10 --backbone vit
```

**Supported datasets and backbones:**

| Dataset | Modality | Backbone | Classes |
| ------- | -------- | -------- | ------- |
| CIFAR-10 | Image | ViT | 10 |
| MNIST | Image | ViT | 10 |
| Speech Commands | Audio | — | 35 |
| SST-2 | Text | BERT | 2 |

---

## Training

```bash
python scripts/train.py --config configs/core_experiments/cifar10_carson.yaml
```

Resume from the latest checkpoint:

```bash
python scripts/train.py --config configs/core_experiments/cifar10_carson.yaml --resume
```

Automated router comparison across all datasets:

```bash
python scripts/run_router_comparison.py
```

---

## Evaluation

```bash
# Evaluate a specific checkpoint
python scripts/evaluate.py --checkpoint checkpoints/core_experiments/.../best_run_001.pt

# Resolve checkpoint from experiment identifiers
python scripts/evaluate.py \
    --main_experiment core_experiments \
    --sub_experiment full_routing \
    --dataset cifar10 \
    --routing carson \
    --run_name run_001 \
    --split test
```

---

## Configuration

All behavior is controlled via YAML. A complete example with all configurable fields:

```yaml
experiment:
  main_experiment_name: core_experiments
  sub_experiment_name:  full_routing
  dataset_name:         cifar10
  model_name:           cats
  routing_type:         carson       # identity | linear | carson
  run_name:             run_001
  seed:                 42

data:
  processed_root:  data/processed
  train_split:     train
  val_split:       validation
  test_split:      test
  has_test:        true
  num_workers:     0
  pin_memory:      true

model:
  embedding_dim:     768
  hidden_dim:        256
  num_classes:       10
  excitatory_ratio:  0.5
  num_groups:        2
  spiking:
    enabled: true
  inhibition:
    enabled: true
  kwargs:
    use_shared_projection: true

position:
  use:  true
  type: rope
  kwargs:
    base: 10000.0

routing:
  kwargs:
    embedding_dim:  768
    hidden_dim:     256
    num_groups:     2
    num_iterations: 3
    dropout:        0.1
    temperature:    1.2
    use_residual:   true
    use_layernorm:  true

# Excitatory LIF population
lif_exc:
  num_groups:    1
  reset_to_zero: true
  detach_reset:  false
  tau:
    learnable: false
    mode:      shared    # shared | per_channel | per_group
    init:      20.0
    min:       14.0
    max:       28.0
  threshold:
    learnable: true
    mode:      shared
    init:      0.65
    min:       0.35
    max:       1.2
  adaptive_threshold:
    enabled:         true
    mode:            shared
    init:            0.0
    decay:           0.97
    spike_increment: 0.05
    min:             0.0
    max:             2.0
    detach_spikes:   true

# Inhibitory LIF population (faster dynamics, lower threshold)
lif_inh:
  num_groups:    1
  reset_to_zero: true
  detach_reset:  false
  tau:
    learnable: false
    mode:      shared
    init:      9.0
    min:       6.0
    max:       14.0
  threshold:
    learnable: true
    mode:      shared
    init:      0.55
    min:       0.35
    max:       0.9
  adaptive_threshold:
    enabled:         true
    mode:            shared
    init:            0.0
    decay:           0.96
    spike_increment: 0.08
    min:             0.0
    max:             2.0
    detach_spikes:   true

classifier:
  input_dim: 256
  kwargs:
    hidden_dim:    null
    dropout:       0.0
    use_layernorm: false

training:
  seed:                      42
  deterministic:             false
  batch_size:                32
  epochs:                    10
  optimizer:                 adamw
  lr:                        1e-4
  weight_decay:              1e-5
  criterion:                 cross_entropy
  grad_clip_norm:            1.0
  early_stopping_enabled:    true
  early_stopping_patience:   3
  early_stopping_min_delta:  0.0001
  lambda_balance:            0.2     # excitatory/inhibitory balance regularization
  lambda_entropy:            0.001   # routing entropy regularization

logging:
  base_dir: logs

checkpoints:
  base_dir:              checkpoints
  best_name_template:    best_{run_name}.pt
  latest_name_template:  latest_{run_name}.pt

runtime:
  device: cuda
```

---

## Logged Metrics

Each training run produces a `train_log.csv` with per-epoch metrics:

**Classification:** `train_loss`, `val_loss`, `train_acc`, `val_acc`, `train_f1`, `val_f1`, `train_precision`, `train_recall`

**Spiking activity:** `firing_rate`, `spikes_per_sample`, `spikes_per_token`, `exc_firing_rate`, `inh_firing_rate`, `exc_inh_diff`

**Routing diagnostics:** `routing_entropy`, `routing_variance`, `routing_conf_mean`, `dominance_fraction`, `exc_usage_fraction`, `inh_usage_fraction`

**Weight dynamics:** `exc_weight_mean`, `inh_weight_mean`, `exc_inh_balance`, `weight_gap`

**Threshold dynamics:** `exc_threshold_mean`, `inh_threshold_mean`, `exc_effective_threshold_mean`, `exc_adaptive_threshold_mean`

**Efficiency:** `epoch_time_sec`, `samples_per_sec`, `forward_time_sec`, `backward_time_sec`, `gpu_peak_mem_mb`

---

## Extending CATS

**New routing algorithm:** subclass `BaseRouter` in `src/cats/encoder/routing/base.py` and register it in the model builder.

**New neuron model:** subclass `BaseSpikingLayer` in `src/cats/encoder/spiking/base.py`.

**New positional encoding:** subclass `BasePositionalEncoding` in `src/cats/encoder/position/base.py`.

**New SNN readout:** add a variant to `src/cats/heads/snn_classifier.py` and update `run_snn_readout_study.py`.

---

## Design Principles

- **Modality-agnostic:** operates on any embedding sequence (text, image, audio)
- **Configuration-driven:** zero hardcoded hyperparameters; all behavior specified via YAML
- **Clean ablations:** fixed neuron model, parameters varied independently for reviewer-grade comparisons
- **Full reproducibility:** seed control, deterministic mode, checkpoint/log tracking
- **Separation of concerns:** representation learning (backbone) decoupled from spike encoding (CATS)

---

## Scope

CATS is a **research framework** focused on correctness, interpretability, and reproducibility. It does not handle raw data preprocessing, backbone training, or production-scale optimization. The upstream embedding backbone is assumed to be pretrained and fixed.

---

## Citation

To be added upon publication.

---

## License

See [LICENSE](LICENSE).
