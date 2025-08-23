# Mixture of Aggregators (MoA)

This repository implements a **Mixture-of-Aggregators (MoA)** framework for **Multiple Instance Learning (MIL)**.  
Instead of committing to a single MIL aggregator (e.g., mean pooling or attention), MoA introduces a **router/gating mechanism** that dynamically selects or combines multiple aggregators depending on the input.

---

## 📌 Features

- Core **MoA module** with multiple aggregation modes:
  - Soft Gating (weighted mixture)
  - Hard Gating (single expert)
  - Top-k Gating
  - Uniform Ensemble
- Router/gating network with configurable design (MLP or Transformer).
- Unified dataset loader for `.pt` and `.h5` patient-level features.
- 5-fold cross-validation training pipeline.
- PyTorch implementation, easy to extend with custom aggregators.

---

## 📂 Repository Structure
