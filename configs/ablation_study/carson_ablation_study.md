# **CARSON Ablation Study (RQ3)**

> **Note:** This ablation study uses the **ANN classifier head** (matching the core experiments configuration). The SNN classifier head with alternative readout strategies is explored separately in `configs/snn_readout_study/` — that study is exploratory and not part of the thesis results.

**RQ3 (Component Attribution):**  
    Which components of the CATS framework (e.g., positional encoding, adaptive thresholding, inhibition) are responsible for the observed performance gains?

## **Vision Backbone Update (Important Implementation Detail)**

For image-based modalities, we adopt a **native-resolution ViT embedding strategy** to improve efficiency and preserve spatial fidelity:

* Replaced fixed **16×16 patch embedding** with **patch size = 4**
* Removed **224×224 upsampling**; inputs are processed at native resolution:

  * MNIST: 28×28
  * CIFAR-10: 32×32
* Transferred pretrained weights via:

  * **Average-pooled patch projection**
  * **Interpolated positional embeddings**

### **Resulting Token Reduction**

* MNIST: **197 → 50 tokens**
* CIFAR-10: **197 → 65 tokens**

### **Impact**

* **Significantly reduced sequence length**, improving:

  * Training speed
  * Memory efficiency
* **Eliminates artificial interpolation artifacts** from upsampling
* Provides a **more faithful evaluation** of routing and spiking dynamics on visual data

We perform a detailed ablation study within the CARSON configuration on **CIFAR-10** and **Speech Commands** — one image and one audio modality, covering the full ablation grid below. The `full_routing` condition is the same run reported as "CARSON" in the core experiments.

## Experiments

| Experiment Name                              | Description                                                                |
| --------------------------------------------- | --------------------------------------------------------------------------- |
| full_routing                                 | Full CARSON configuration (baseline)                                       |
| no_positional                                | Remove positional encoding (RoPE disabled)                                 |
| fixed_tau                                    | Fix membrane time constant (τ), keep threshold and adaptive dynamics       |
| fixed_threshold                              | Fix base threshold (θ), keep adaptive threshold enabled                    |
| no_adaptive                                  | Disable adaptive threshold dynamics (θ_adapt), keep τ and θ learnable      |
| fixed_tau_fixed_threshold                    | Fix τ and base θ, keep adaptive threshold enabled                          |
| fixed_tau_fixed_threshold_no_adaptive        | Fully static spiking dynamics (τ fixed, θ fixed, no adaptive threshold)    |
| no_inhibition                                | Disable inhibitory pathway (excitatory-only spiking)                       |
| no_spiking                                   | Remove spiking encoder (ANN-only model after routing)                      |

Each condition's CARSON config lives under `configs/ablation_study/<condition>/{cifar10,speech_commands}/train_carson_*.yaml`. Several conditions also include `linear`/`identity` variants for additional context, but the primary ablation comparison reported in the thesis uses the **CARSON** runs only.

> `no_positional` was additionally run for **MNIST** and **SST-2** (`configs/ablation_study/no_positional/{mnist,sst2}/`). These runs are logged but were not carried through to a full analysis notebook — left for future work (see `Experimental_Proposal.md`).

### Goal

To identify which components contribute most to performance and understand the role of:

* Adaptive spiking dynamics
* Temporal encoding (RoPE)
* Inhibitory mechanisms
* Spiking vs. non-spiking representations

## Status: Completed (CIFAR-10 and Speech Commands)

Results & figures: `results/ablation_study/cifar10/` and `results/ablation_study/speech_commands/`. Logs: `logs/ablation_study/`.

### Results — CIFAR-10 (full model Val F1 = 0.6754)

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

### Results — Speech Commands (full model Val F1 = 0.9356)

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

### Key Findings

* **Spiking matters most on CIFAR-10** (−0.13 pp without it), but is roughly neutral on Speech Commands (−0.02 pp).
* **The inhibitory population is more important for audio than vision** (−0.16 pp on Speech Commands vs +0.10 pp on CIFAR-10).
* **`fixed_tau_fixed_threshold` is the best configuration overall** on both datasets — the adaptive-threshold mechanism does not provide a measurable benefit over fixed values in these runs.
* **RoPE helps image patch sequences but not audio** — removing it is roughly neutral on CIFAR-10 (+0.01 pp) but improves Speech Commands by +0.29 pp, the largest single effect for that dataset.
