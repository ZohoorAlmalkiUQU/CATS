# **Experimental Design: CATS and the CARSON Router**

## **1\. Objective**

This document describes the experimental design used to evaluate the proposed **CATS (Context-Aware Token-to-Spike)** framework, with a particular focus on the **CARSON routing mechanism**, under controlled and fair experimental conditions.

The experiments below were executed and their results are reported in the thesis. All raw logs, checkpoints, and analysis notebooks live under `logs/`, `checkpoints/`, and `results/` respectively, organized by experiment name to mirror this document and the configs in `configs/`.

Specifically, we aimed to:

1. Assess whether **CARSON outperforms simpler routing alternatives** (identity and linear routing).
2. Evaluate whether such performance gains **generalize across modalities** (text, image, and audio).
3. Identify which architectural components **contribute most to CARSON's effectiveness**.
4. Analyze whether CARSON exhibits **improved learning dynamics**, such as faster convergence or better early-stage performance.

* * *

## **2\. Research Questions**

We structure our experiments around the following research questions:

- **RQ1 (Router Superiority):**
    Does CARSON outperform identity and linear routing under matched architectural and optimization settings?
- **RQ2 (Cross-Modality Generalization):**
    Do the performance gains of CARSON persist across different modalities (text, image, audio)?
- **RQ3 (Component Attribution):**
    Which components of the CATS framework (e.g., positional encoding, adaptive thresholding, inhibition) are responsible for the observed performance gains?
- **RQ4 (Learning Dynamics):**
    Does CARSON improve optimization behavior, such as faster convergence or better early-stage validation performance?

* * *

## **3\. Experimental Setup**

### **3.1 Modalities and Datasets**

We evaluate the framework across three modalities — image (CIFAR-10, MNIST), text (SST-2), and audio (Speech Commands).

### **Experimental Grid (Core Results)**

| Modality | Dataset         | Router   |
| -------- | --------------- | -------- |
| Image    | CIFAR-10        | identity |
| Image    | CIFAR-10        | linear   |
| Image    | CIFAR-10        | carson   |
| Image    | MNIST           | identity |
| Image    | MNIST           | linear   |
| Image    | MNIST           | carson   |
| Text     | SST-2           | identity |
| Text     | SST-2           | linear   |
| Text     | SST-2           | carson   |
| Audio    | Speech Commands | identity |
| Audio    | Speech Commands | linear   |
| Audio    | Speech Commands | carson   |

All experiments use **precomputed embeddings** to isolate the effect of routing and spiking dynamics. Configs: `configs/core_experiments/`. Results: `results/core_experiments_analysis/`.

* * *

### **3.2 Controlled Experimental Conditions**

To ensure fair comparison, the following factors are **held constant across all models**:

- Embedding inputs (same backbone and preprocessing)
- Hidden dimension and model capacity
- Number of groups
- Classifier head architecture
- Optimizer, learning rate, and scheduler
- Batch size
- Training budget (epochs)
- Early stopping criterion
- Random seed (fixed at **42** for all experiments, for reproducibility)

Only the **routing mechanism or ablated component** is varied per experiment.

* * *

## **4\. Core Experiments (Main Thesis)**

* * *

### **4.1 Router Comparison (RQ1 + RQ2)**

We compare three routing strategies under identical settings:

- **Identity (no routing)**
- **Linear routing**
- **CARSON routing**

All models use:

- Positional Encoding: **RoPE**
- Threshold: **Adaptive**
- Learnable Parameters: **τ (tau) + threshold**
- Inhibition: **Enabled**
- Output: **Spiking**

#### Status: Completed (Router Comparison)

All 12 cells of the experimental grid above (4 datasets × 3 routers) were trained and evaluated. CARSON outperforms both Identity and Linear routing on **every** dataset/modality:

| Dataset | Modality | Identity (F1 / Acc) | Linear (F1 / Acc) | CARSON (F1 / Acc) |
| ------- | -------- | -------------------- | ------------------ | ------------------ |
| CIFAR-10 | Image | 0.6125 / 61.1% | 0.6226 / 62.6% | **0.6754 / 67.6%** |
| MNIST | Image | 0.9609 / 96.1% | 0.9606 / 96.1% | **0.9802 / 98.0%** |
| Speech Commands | Audio | 0.9083 / 91.5% | 0.9199 / 92.3% | **0.9356 / 93.8%** |
| SST-2 | Text | 0.8375 / 82.9% | 0.8756 / 87.4% | **0.8993 / 89.5%** |

See `README.md` § Results and `results/core_experiments_analysis/` for per-dataset training-log notebooks.

#### Evaluation Metrics

- Accuracy
- F1 Score

* * *

### **4.2 Learning Dynamics Analysis (RQ4)**

To evaluate convergence behavior, we analyze:

- Best validation F1 score
- Epoch at which best validation F1 is reached
- Training/validation curves per epoch

#### Status: Completed (Learning Dynamics)

Convergence comparisons across Identity / Linear / CARSON for CIFAR-10 and Speech Commands are in `results/core_experiments_analysis/shared_analysis/convergence_comparison.ipynb`, with figures in `results/core_experiments_analysis/shared_analysis/figures/`.

* * *

### **4.3 CARSON Ablation Study (RQ3)**

We perform a detailed component ablation within the CARSON configuration.

#### Experiments

| Experiment Name                              | Description                                                                |
| --------------------------------------------- | ------------------------------------------------------------------------- |
| full_routing                                 | Full CARSON configuration (baseline)                                       |
| no_positional                                | Remove positional encoding (RoPE disabled)                                 |
| fixed_tau                                    | Fix membrane time constant (τ), keep threshold and adaptive dynamics       |
| fixed_threshold                              | Fix base threshold (θ), keep adaptive threshold enabled                    |
| no_adaptive                                  | Disable adaptive threshold dynamics (θ_adapt), keep τ and θ learnable      |
| fixed_tau_fixed_threshold                    | Fix τ and base θ, keep adaptive threshold enabled                          |
| fixed_tau_fixed_threshold_no_adaptive        | Fully static spiking dynamics (τ fixed, θ fixed, no adaptive threshold)    |
| no_inhibition                                | Disable inhibitory pathway (excitatory-only spiking)                       |
| no_spiking                                   | Remove spiking encoder (ANN-only model after routing)                      |

#### Status: Completed (CIFAR-10 and Speech Commands)

All 9 conditions were run for **CIFAR-10** and **Speech Commands**, with the CARSON router and an ANN classifier head (matching the core-experiment configuration). Identity-router baselines for several conditions were also run on Speech Commands for additional context.

Configs: `configs/ablation_study/`. Results & figures: `results/ablation_study/cifar10/` and `results/ablation_study/speech_commands/`.

#### Goal

To identify which components contribute most to performance and understand the role of:

- Adaptive spiking dynamics
- Temporal encoding (RoPE)
- Inhibitory mechanisms
- Spiking vs. non-spiking representations

#### Key Findings

- **Spiking matters most on CIFAR-10**, but is roughly neutral on Speech Commands.
- **The inhibitory population is more important for audio than vision.**
- **`fixed_tau_fixed_threshold` is the best configuration overall** on both datasets — the adaptive-threshold mechanism does not add measurable benefit over fixed values in these runs.
- **RoPE positional encoding helps image patch sequences but not audio.**

See `README.md` § Ablation Study for the full result tables.

* * *

## **5\. Exploratory / Future Work (Not Included in Thesis)**

* * *

### **5.1 SNN Readout Study (CIFAR-10)**

Six classifier readout strategies (`mean_membrane`, `last_membrane`, `temporal_pool`, `membrane_spike_hybrid`, `spike_count`, `mean_spikes`) were compared for the SNN classifier head, under CARSON routing on CIFAR-10.

**Status: Completed, but excluded from the thesis** — the thesis is already long enough without it, and this line of work (SNN readout design) is left for **future research**, which will focus on the SNN field specifically.

Configs: `configs/snn_readout_study/`. Results: `results/snn_readout_study/`.

* * *

### **5.2 Cross-Modality Ablations (Partial / Future Work)**

`no_positional` ablation runs were additionally executed for **MNIST** and **SST-2** (`configs/ablation_study/no_positional/{mnist,sst2}`), to check whether the RoPE finding from CIFAR-10/Speech Commands generalizes further. These runs are logged under `logs/ablation_study/no_positional/{mnist,sst2}/` but were **not** carried through to a full analysis notebook and are **not** part of the thesis results — left as a starting point for future work.

* * *

### **5.3 Other Appendix Items (Not Pursued)**

The following items from the original proposal were **not pursued** in this iteration and are left for future work:

- Sensitivity analysis over hyperparameters (number of groups, hidden dimension, routing iterations, threshold initialization).
- Per-seed variance analysis — all reported results use a single fixed seed (42) for reproducibility and resource constraints.

* * *

## **6\. Evaluation Metrics**

### **Primary Metrics**

- Accuracy
- F1 Score (preferred for classification robustness)

### **Convergence Metrics**

- Epoch to best performance
- Training/validation learning curves

### **SNN-Specific Metrics**

- Spikes per sample / per token
- Average firing rate (overall, excitatory, inhibitory)
- Routing entropy and variance

* * *

## **7\. Findings Summary**

The completed experiments support the following conclusions:

1. **CARSON outperforms simpler routing mechanisms** (identity and linear) on every dataset evaluated.
2. **The gains generalize across modalities** — image (CIFAR-10, MNIST), audio (Speech Commands), and text (SST-2).
3. **Component contributions are modality-dependent**: spiking and RoPE matter most for vision, while the inhibitory population matters most for audio. Fixing τ and θ (i.e., removing the adaptive-threshold mechanism) is competitive with — or better than — the fully adaptive configuration on both datasets tested.
4. The **SNN readout study** and the **MNIST/SST-2 `no_positional` ablations** produced usable results but are reserved for future work rather than included in the thesis.

* * *

## **8\. Summary**

This document records:

- The full experimental grid that was executed (core experiments + ablations)
- Where to find the corresponding configs, logs, checkpoints, and analysis notebooks for each experiment
- Which exploratory experiments were completed but deliberately excluded from the thesis, and why

Note: results under `archive/` correspond to early, failed pilot runs and are **not** part of any reported result — they predate the current pipeline and should be disregarded.

* * *
