# **Diagnostic Subset (Initial Pilot Runs)**

> **Status: Initial / pilot runs only — not part of the thesis results.** These configs were used to validate the training pipeline and the CARSON ablation conditions on a single dataset (Speech Commands) **before** running the full ablation study reported in `configs/ablation_study/`. They are kept here for reference and reproducibility, but the numbers from these runs are **not** analyzed or reported anywhere in the thesis.

## Purpose

Before committing to the full CIFAR-10 + Speech Commands ablation grid (see `configs/ablation_study/carson_ablation_study.md`), each of the 9 ablation conditions was first sanity-checked on **Speech Commands only**. These initial runs confirmed that:

- Each ablation condition's config loads and trains correctly.
- The CARSON router and spiking encoder behave as expected under each modified configuration.
- The training pipeline (logging, checkpointing, metrics) works end-to-end for every condition.

Once these pilot runs confirmed the pipeline was correct, the full ablation study (CIFAR-10 + Speech Commands, with additional analysis) was run separately and is the version reported in the thesis.

## Conditions Covered

All configs are under `configs/diagnostic_subset/<condition>/speech_commands/`:

| Condition | Config(s) |
| --------- | --------- |
| full_router | `train_carson_speech_commands_run_001.yaml` |
| fixed_tau | `train_carson_speech_commands_run_001.yaml`, `train_linear_speech_commands_run_001.yaml` |
| fixed_threshold | `train_carson_speech_commands_run_001.yaml` |
| fixed_tau_fixed_threshold | `train_carson_speech_commands_run_001.yaml` |
| fixed_tau_fixed_threshold_no_adaptive | `train_carson_speech_commands_run_001.yaml` |
| no_adaptive | `train_carson_speech_commands_run_001.yaml` |
| no_inhibition | `train_carson_speech_commands_run_001.yaml` |
| no_positional | `train_carson_speech_commands_run_001.yaml` |
| no_spiking | `train_carson_speech_commands_run_001.yaml` |

> The `fixed_tau` condition additionally includes a `linear` router variant for early comparison; this was not carried forward into the full ablation study.

## Logs

Corresponding logs live under `logs/diagnostic_subset/<condition>/speech_commands/cats/carson/run_00{1,2}/`. Some conditions (e.g. `fixed_tau`, `full_router`) have both `run_001` and `run_002` from repeated pilot runs during pipeline debugging.

## Relationship to Thesis Results

These pilot runs are **not referenced** in `Experimental_Proposal.md` or `README.md`. The thesis-reported ablation results come from the full study in `configs/ablation_study/` and `results/ablation_study/`. This `diagnostic_subset` is retained purely for reproducibility/history of the development process.
