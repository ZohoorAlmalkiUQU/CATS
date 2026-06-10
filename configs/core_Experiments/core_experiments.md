# **Core Experiments (Main Paper) — RQ1 & RQ2 (Updated)**

* **RQ1 (Router Superiority):**
  Does CARSON outperform identity and linear routing under matched architectural and optimization settings?

* **RQ2 (Cross-Modality Generalization):**
  Do the performance gains of CARSON persist across different modalities (text, image, audio)?

---

All experiments use **precomputed embeddings** to isolate the effect of routing and spiking dynamics.

---

## **Controlled Experimental Conditions**

To ensure fair comparison, the following factors are **held constant across all models**:

* Embedding inputs (same backbone and preprocessing)
* Hidden dimension and model capacity
* Number of groups
* Classifier head architecture
* Optimizer, learning rate, and scheduler
* Batch size
* Training budget (epochs)
* Early stopping criterion
* Random seed (fixed at 42 across all experiments)

Only the **routing mechanism or ablated component** is varied per experiment.

---

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

---

## **Router Comparison (RQ1 + RQ2)**

We compare three routing strategies under identical settings:

* **Identity (no routing)**
* **Linear routing**
* **CARSON routing**

All models use:

* Positional Encoding: **RoPE**
* Threshold: **Adaptive**
* Learnable Parameters: **τ (tau) + threshold**
* Inhibition: **Enabled**
* Output: **Spiking**

---

### **Experimental Grid (Core Results)**

| Modality | Dataset         | Router   | Config |
| -------- | --------------- | -------- | ------ |
| Image    | CIFAR-10        | identity | `cifar10/train_identity_cifar10_run_001.yaml` |
| Image    | CIFAR-10        | linear   | `cifar10/train_linear_cifar10_run_001.yaml` |
| Image    | CIFAR-10        | carson   | `cifar10/train_carson_cifar10_run_001.yaml` |
| Image    | MNIST           | identity | `mnist/train_identity_mnist_run_001.yaml` |
| Image    | MNIST           | linear   | `mnist/train_linear_mnist_run_001.yaml` |
| Image    | MNIST           | carson   | `mnist/train_carson_mnist_run_001.yaml` |
| Text     | SST-2           | identity | `sst2/train_identity_sst2_run_001.yaml` |
| Text     | SST-2           | linear   | `sst2/train_linear_sst2_run_001.yaml` |
| Text     | SST-2           | carson   | `sst2/train_carson_sst2_run_001.yaml` |
| Audio    | Speech Commands | identity | `speech_commands/train_identity_speech_commands_run_001.yaml` |
| Audio    | Speech Commands | linear   | `speech_commands/train_linear_speech_commands_run_001.yaml` |
| Audio    | Speech Commands | carson   | `speech_commands/train_carson_speech_commands_run_001.yaml` |

---

## **Status: Completed**

All 12 runs above were trained and evaluated. Logs: `logs/core_experiments/full_routing/`. Checkpoints: `checkpoints/core_experiments/full_routing/`. Analysis notebooks: `results/core_experiments_analysis/`.

### Results

| Dataset | Modality | Identity (F1 / Acc) | Linear (F1 / Acc) | CARSON (F1 / Acc) |
| ------- | -------- | -------------------- | ------------------ | ------------------ |
| CIFAR-10 | Image | 0.6125 / 61.1% | 0.6226 / 62.6% | **0.6754 / 67.6%** |
| MNIST | Image | 0.9609 / 96.1% | 0.9606 / 96.1% | **0.9802 / 98.0%** |
| Speech Commands | Audio | 0.9083 / 91.5% | 0.9199 / 92.3% | **0.9356 / 93.8%** |
| SST-2 | Text | 0.8375 / 82.9% | 0.8756 / 87.4% | **0.8993 / 89.5%** |

CARSON outperforms both Identity and Linear routing on every dataset and modality. See `README.md` § Results for the full discussion and `results/core_experiments_analysis/shared_analysis/` for cross-router convergence comparisons.

---
