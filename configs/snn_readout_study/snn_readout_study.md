# **SNN Readout Study (CIFAR-10)**

> **Status: Completed, but excluded from the thesis.** This study is exploratory — it is kept here as a starting point for **future research** focused specifically on the SNN field. It is *not* part of the core CATS/CARSON thesis results in `Experimental_Proposal.md` or `README.md`.

## Goal

The CATS classifier head can read out from the spiking encoder's membrane potential and/or spike trains in several ways. This study compares **six readout strategies** for the SNN classifier head, with everything else held fixed (CARSON routing, CIFAR-10, full adaptive spiking configuration — same as `full_routing` in the ablation study, but with an SNN classifier head instead of an ANN head).

## Experiments

| Variant                | Description                                              | Config |
| ----------------------- | --------------------------------------------------------- | ------ |
| mean_membrane           | Classify from the time-averaged membrane potential        | `cifar10/train_carson_cifar10_mean_membrane.yaml` |
| last_membrane           | Classify from the final time-step's membrane potential    | `cifar10/train_carson_cifar10_last_membrane.yaml` |
| temporal_pool           | Temporal pooling over membrane potentials across steps    | `cifar10/train_carson_cifar10_temporal_pool.yaml` |
| membrane_spike_hybrid   | Concatenation of membrane potential and spike statistics  | `cifar10/train_carson_cifar10_membrane_spike_hybrid.yaml` |
| spike_count             | Classify from total spike counts per neuron               | `cifar10/train_carson_cifar10_spike_count.yaml` |
| mean_spikes             | Classify from the mean spike rate per neuron              | `cifar10/train_carson_cifar10_mean_spikes.yaml` |

> Equivalent configs exist under `configs/snn_readout_study/speech_commands/` for a possible audio-modality follow-up, but only the **CIFAR-10** runs were executed and analyzed.

## Results

Logs: `logs/snn_readout_study/<variant>/cifar10/cats/carson/run_001/`. Analysis notebook & figures: `results/snn_readout_study/`.

| Readout | Val F1 | Val Acc | Best Epoch | Spk/Sample | Firing Rate |
| ------- | ------ | ------- | ---------- | ---------- | ----------- |
| Last membrane (final step) | **0.6736** | 67.5% | 7 | 4,371 | 0.263 |
| Mean membrane (μ) | 0.6694 | 66.9% | 8 | 6,391 | 0.384 |
| Temporal pool | 0.6693 | 67.1% | 6 | 5,755 | 0.346 |
| Membrane + spike hybrid | 0.6628 | 66.5% | 8 | 3,041 | 0.183 |
| Spike count | 0.6519 | 64.9% | 7 | 2,928 | 0.176 |
| Mean spikes (spike rate) | 0.5925 | 60.7% | 10 | 6,114 | 0.367 |

## Key Findings

1. **All six readout strategies converge** — the surrogate gradient propagates through both membrane-potential and spike-based readouts when properly implemented.
2. **`last_membrane` is the best overall readout** and the most spike-efficient membrane-based one — it tops the table while firing at roughly two-thirds the rate of `mean_membrane`.
3. **Membrane-based readouts outperform pure spike-based readouts** by 1.1–8.1 pp.
4. **`spike_count` is the most spike-efficient overall** (lowest firing rate at 0.176), offering the best accuracy-per-spike trade-off among spike-based strategies.
5. **`temporal_pool` converges fastest** (best epoch 6), suggesting a stronger early learning signal.
6. **`mean_spikes` underperforms** — despite firing activity comparable to membrane variants, it produces the lowest val F1, indicating spike rate alone is a weaker task signal than membrane potential at this sequence length.

## Future Work

- Run the equivalent comparison on Speech Commands (configs already exist under `configs/snn_readout_study/speech_commands/`).
- Investigate whether the `last_membrane` advantage holds when combined with the ablation findings (e.g. `fixed_tau_fixed_threshold`).
