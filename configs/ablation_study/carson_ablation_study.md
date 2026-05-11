# **CARSON Ablation Study (RQ3)**

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

`We perform a detailed ablation study within the CARSON configuration (on a primary modality, e.g., CIFAR-10).`

## Experiments

| Experiment Name                              | Description                                                                |
|----------------------------------------------|----------------------------------------------------------------------------|
| baseline_cats                                | Full CARSON configuration                                                  |
| no_positional                                | Remove positional encoding (RoPE disabled)                                 |
| fixed_tau                                    | Fix membrane time constant (τ), keep threshold and adaptive dynamics       |
| fixed_threshold                              | Fix base threshold (θ), keep adaptive threshold enabled                    |
| no_adaptive                                  | Disable adaptive threshold dynamics (θ_adapt), keep τ and θ learnable      |
| fixed_tau_fixed_threshold                    | Fix τ and base θ, keep adaptive threshold enabled                          |
| fixed_tau_fixed_threshold_no_adaptive        | Fully static spiking dynamics (τ fixed, θ fixed, no adaptive threshold)    |
| no_inhibition                                | Disable inhibitory pathway (excitatory-only spiking)                       |
| no_spiking                                   | Remove spiking encoder (ANN-only model after routing)                      |

### Goal

To identify which components contribute most to performance and understand the role of:

* Adaptive spiking dynamics
* Temporal encoding
* Inhibitory mechanisms
