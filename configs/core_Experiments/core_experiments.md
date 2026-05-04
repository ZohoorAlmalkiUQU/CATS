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
* Random seeds

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

---
