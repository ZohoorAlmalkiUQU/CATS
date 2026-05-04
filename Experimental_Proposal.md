# **Experimental Proposal: Evaluating CATS and the CARSON Router**

## **1\. Objective**

The goal of this study is to rigorously evaluate the proposed **CATS (Context-Aware Token-to-Spike)** framework, with a particular focus on the **CARSON routing mechanism**, under controlled and fair experimental conditions.

Specifically, we aim to:

1. Assess whether **CARSON outperforms simpler routing alternatives** (identity and linear routing).
2. Evaluate whether such performance gains **generalize across modalities** (text, image, and audio).
3. Identify which architectural components **contribute most to CARSON’s effectiveness**.
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

We evaluate the framework across three modalities:

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

All experiments use **precomputed embeddings** to isolate the effect of routing and spiking dynamics.

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
- Random seeds

Only the **routing mechanism or ablated component** is varied per experiment.

* * *

## **4\. Core Experiments (Main Paper)**

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

#### Experimental Grid

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

#### Evaluation Metrics

- Accuracy
- F1 Score
- Mean ± standard deviation over multiple seeds

* * *

### **4.2 Learning Dynamics Analysis (RQ4)**

To evaluate convergence behavior, we analyze:

- Best validation F1 score
- Epoch at which best validation F1 is reached
- Wall-clock time to best validation performance
- Epochs required to reach 95% of peak performance
- Area under the validation learning curve (early training phase)

This allows us to assess whether CARSON provides **optimization advantages**, not just final performance gains.

* * *

### **4.3 CARSON Ablation Study (RQ3)**

We perform a detailed ablation study within the CARSON configuration (on a primary modality, e.g., SST-2).

#### Experiments

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

#### Goal

To identify which components contribute most to performance and understand the role of:

- Adaptive spiking dynamics
- Temporal encoding
- Inhibitory mechanisms

* * *

## **5\. Additional Experiments (Appendix)**

* * *

### **5.1 Cross-Modality Ablations**

We replicate a subset of key ablations on image and audio modalities:

- no_positional
- fixed_tau_fixed_threshold
- no_inhibition
- no_spiking

This evaluates whether component importance is **consistent across modalities**.

* * *

### **5.2 Sensitivity Analysis**

We analyze robustness to hyperparameters such as:

- Number of groups
- Hidden dimension
- Routing iterations
- Threshold initialization

* * *

### **5.3 Full Training Curves**

We provide full training dynamics:

- Training/validation loss
- Validation metrics per epoch
- Spike statistics (e.g., firing rate, sparsity)

* * *

### **5.4 Per-Seed Results**

We report:

- Individual seed results
- Aggregate mean ± standard deviation

to ensure statistical reliability.

* * *

## **6\. Evaluation Metrics**

### **Primary Metrics**

- Accuracy
- F1 Score (preferred for classification robustness)

### **Convergence Metrics**

- Epoch to best performance
- Time-to-target performance
- Early training performance (learning curve AUC)

### **SNN-Specific Metrics**

- Spikes per sample
- Average firing rate
- Representation sparsity

* * *

## **7\. Expected Contributions**

This experimental protocol is designed to support the following claims:

1. **CARSON outperforms simpler routing mechanisms** under controlled conditions.
2. The observed gains are **consistent across modalities**, indicating generality.
3. The improvements are driven by **specific architectural components**, particularly:
    - Adaptive thresholding
    - Learnable membrane dynamics
    - Positional encoding
4. CARSON exhibits **favorable learning dynamics**, reaching strong performance earlier in training.

* * *

## **8\. Summary**

This proposal ensures:

- Fair and controlled comparisons
- Clear separation between **performance evaluation** and **component analysis**
- Strong empirical grounding for all claims
- Scalability across modalities
- Transparency and reproducibility

* * *
