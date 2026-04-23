
# **Core Experiments (Main Paper) - RQ1 & RQ2**

- **RQ1 (Router Superiority):**  
    Does CARSON outperform identity and linear routing under matched architectural and optimization settings?  
- **RQ2 (Cross-Modality Generalization):**  
    Do the performance gains of CARSON persist across different modalities (text, image, audio)?

* * *

All experiments use **precomputed embeddings** to isolate the effect of routing and spiking dynamics.

* * *

## **Controlled Experimental Conditions**

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

## **Router Comparison (RQ1 + RQ2)**

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

### Experimental Grid

| Modality | Dataset         | Router   |
|----------|-----------------|----------|
| Text     | SST-2           | identity |
| Text     | SST-2           | linear   |
| Text     | SST-2           | carson   |
| Image    | CIFAR-10        | identity |
| Image    | CIFAR-10        | linear   |
| Image    | CIFAR-10        | carson   |
| Audio    | Speech Commands | identity |
| Audio    | Speech Commands | linear   |
| Audio    | Speech Commands | carson   |
