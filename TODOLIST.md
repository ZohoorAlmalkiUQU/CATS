> This file is just for me to keep all tasks in front of me; so please just ignore it :) T7yati Zohoory ✌️

# 🧠 CATS — MASTER TODO PLAN

---

## 🔹 Phase 0 — Foundation

### ☐ Task 1: Verify pipeline works ✅ Done

* ☐ Load saved embeddings correctly
* ☐ Build dummy encoder (no routing, no spiking)
* ☐ Train classifier
* ☐ Check:

  * ☐ shapes are consistent
  * ☐ loss decreases
  * ☐ accuracy > random
* ☐ Confirm no bugs (device / dtype / loading)

---

## 🔹 Phase 1 — Core CATS Architecture

### ☐ Task 2: No Routing + Spiking ⏺️ Ongoing

* ☐ Implement LIF layer
* ☐ Add normalization before LIF
* ☐ Apply mask properly
* ☐ Pipeline:

  ```text
  embedding → LIF → classifier
  ```
* ☐ Train + log:

  * ☐ accuracy / F1
  * ☐ firing rate
  * ☐ spikes per sample

---

### ☐ Task 3: Linear Routing + Spiking

* ☐ Implement linear router
* ☐ Pipeline:

  ```text
  embedding → linear → LIF → classifier
  ```
* ☐ Train and compare vs Task 2

---

### ☐ Task 4: Gating Routing + Spiking

* ☐ Implement gating router
* ☐ Pipeline:

  ```text
  embedding → gating → LIF → classifier
  ```
* ☐ Train and compare

---

### ☐ Task 5: CARSON + Spiking (main contribution)

* ☐ Implement CARSON routing
* ☐ Pipeline:

  ```text
  embedding → CARSON → LIF → classifier
  ```
* ☐ Train and verify:

  * ☐ better performance
  * ☐ better spike efficiency
  * ☐ or both

---

## 🔹 Phase 2 — Clean Ablation (CRITICAL)

### ☐ Run fair comparison across:

* ☐ No routing
* ☐ Linear
* ☐ Gating
* ☐ CARSON

### ☐ Keep everything fixed:

* ☐ dataset
* ☐ embeddings
* ☐ task loss
* ☐ training setup
* ☐ evaluation protocol

---

### ☐ Produce Table 1

* ☐ Accuracy
* ☐ Macro-F1
* ☐ Firing rate
* ☐ Spikes per sample

---

## 🔹 Phase 3 — Efficiency Analysis (CRITICAL)

### ☐ Compute:

* ☐ average firing rate
* ☐ spikes per sample
* ☐ spike sparsity
* ☐ active neuron ratio

---

### ☐ Generate Figure 1

* ☐ x-axis = spikes per sample
* ☐ y-axis = accuracy or Macro-F1
* ☐ plot all models

---

### ☐ Generate Figure 2

* ☐ bar plot of firing rate per model

---

## 🔹 Phase 4 — Routing Behavior Analysis (CRITICAL)

### ☐ Compute route utilization

* ☐ tokens per route
* ☐ normalized route usage

→ ☐ Figure 3 (route utilization bar plot)

---

### ☐ Compute routing entropy

* ☐ average entropy per token
* ☐ compare Gating vs CARSON

---

### ☐ Analyze token position vs routing

→ ☐ Figure 4 (heatmap)

* x = token position
* y = route
* value = average usage / routing weight

---

### ☐ Visualize real examples

→ ☐ Figure 5

* ☐ pick 2–3 samples
* ☐ show token → route mapping

---

### ☐ (Strong) visualize spike activity

→ ☐ Figure 6

* ☐ compare spike patterns across models

---

## 🔹 Phase 5 — Intra-Modality Generalization

### ☐ Validate on multiple text datasets first

* ☐ SST-2
* ☐ IMDB
* ☐ AG News

### ☐ Compare at least:

* ☐ No routing
* ☐ Gating
* ☐ CARSON

### ☐ Produce Table 2

* ☐ cross-dataset performance
* ☐ cross-dataset spike efficiency

### ☐ Verify:

* ☐ CARSON is not only winning on one text dataset
* ☐ routing behavior remains meaningful across text tasks

---

## 🔹 Phase 6 — Embedding Interface Standardization (FUNDAMENTAL)

### ☐ Define the core input contract for CATS

* ☐ all modalities must enter as embedding sequences
* ☐ target canonical shape:

  ```text
  (batch, sequence_length, d_model)
  ```

### ☐ Define masking rules

* ☐ text: attention mask
* ☐ image: patch-validity mask if needed
* ☐ audio: frame/time mask if padded

### ☐ Define embedding projection policy

* ☐ project all backbones to a shared `d_model` when necessary

### ☐ Define normalization policy

* ☐ apply pre-spiking normalization consistently across modalities

### ☐ Verify Task 1 pipeline interface still matches this design

* ☐ make sure Task 1 does not lock me into text-specific assumptions
* ☐ keep encoder API modality-agnostic from the start

---

## 🔹 Phase 7 — Multi-Modality Validation (FUNDAMENTAL)

### ☐ Prove that CATS is actually modality-agnostic

* ☐ do not leave this as a future-work-only claim

### ☐ Add at least one non-text modality

* ☐ image embeddings
* ☐ audio embeddings
* ☐ ideally both if feasible

### ☐ Keep the CATS encoder unchanged

* ☐ same encoder design
* ☐ same routing modules
* ☐ same spiking encoder logic
* ☐ only backbone / embedding source changes

### ☐ Build modality-specific embedding pipelines

* ☐ Text:

  ```text
  text → Transformer → embeddings
  ```
* ☐ Image:

  ```text
  image → ViT/CNN patches/features → embeddings
  ```
* ☐ Audio:

  ```text
  audio → frame/spectrogram encoder → embeddings
  ```

### ☐ Standardize all of them into:

```text
(B, L, D)
```

### ☐ Run cross-modality experiments

* ☐ compare No routing / Gating / CARSON where possible
* ☐ evaluate performance and spike efficiency in each modality

### ☐ Produce Table 3

* ☐ modality
* ☐ dataset
* ☐ model
* ☐ Accuracy / F1
* ☐ firing rate
* ☐ spikes per sample

### ☐ Produce a cross-modality summary figure

* ☐ show whether CARSON remains beneficial beyond text

### ☐ Verify the actual claim:

* ☐ CATS works on embedding sequences, not on one specific modality
* ☐ routing-guided spike encoding transfers across modalities without architecture redesign

---

## 🔹 Phase 8 — (Optional) Objective Extensions

### ☐ Only after the core claim is proven

* ☐ CARSON + task loss only
* ☐ CARSON + rate loss
* ☐ CARSON + entropy regularization
* ☐ CARSON + balance regularization

### ☐ Compare:

* ☐ performance
* ☐ firing rate
* ☐ routing behavior

### ☐ Treat this as enhancement, not the core claim

---

## 🔹 Phase 9 — Paper Writing

### ☐ Section 1: Introduction

* ☐ define the gap
* ☐ explain embedding-level spiking encoder idea
* ☐ motivate modality-agnostic design
* ☐ state contributions clearly

---

### ☐ Section 2: Method

* ☐ CATS encoder
* ☐ routing modules
* ☐ CARSON
* ☐ embedding-sequence interface
* ☐ spiking mechanism

---

### ☐ Section 3: Experimental Setup

* ☐ datasets
* ☐ embedding backbones
* ☐ evaluation metrics
* ☐ training details
* ☐ implementation details

---

### ☐ Section 4: Results

* ☐ Architectural ablation → Table 1
* ☐ Spike efficiency → Figures 1–2
* ☐ Routing behavior → Figures 3–6
* ☐ Intra-modality generalization → Table 2
* ☐ Cross-modality validation → Table 3

---

### ☐ Section 5: Discussion

* ☐ why routing helps
* ☐ why CARSON helps
* ☐ how routing affects spike efficiency
* ☐ what “modality-agnostic” really means in practice
* ☐ limitations

---

# 🔥 My execution priority

1. ☐ Finish Tasks 2–5
2. ☐ Run clean ablation on SST-2
3. ☐ Build Table 1
4. ☐ Add spike-efficiency metrics
5. ☐ Add routing-behavior analysis
6. ☐ Validate across multiple text datasets
7. ☐ Standardize embedding interface across modalities
8. ☐ Run at least one non-text modality
9. ☐ Add second non-text modality if feasible
10. ☐ Only then consider objective extensions

---

# 🎯 Reminder to myself

* ☐ My core claim is not just “CARSON improves accuracy”
* ☐ My core claim is that CATS is a **modality-agnostic spiking encoder for embedding sequences**
* ☐ So I must prove:

  * ☐ architectural benefit
  * ☐ spike efficiency
  * ☐ routing structure
  * ☐ cross-modality transfer without redesign

---