# Architecture Recommendations for scGPT

Based on analysis of the original scGPT paper, two critique papers (zero-shot evaluation and perturbation prediction), and the existing codebase.

---

## Root Problems Identified

The critique papers expose three fundamental issues:

1. **scGPT treats genes as a sequence** — they are a set (exchangeable, no natural order)
2. **MLM on discretized bins** is a weak pretraining objective that doesn't match the ZINB data distribution; the model defaults to predicting the mean bin
3. **Perturbation prediction requires causal reasoning** (do-calculus), not observational correlation learning — atlas pretraining cannot bridge this gap

---

## Recommended Changes, Prioritised

### Priority 1 — Quick Wins

#### 1. Quantile Binning (replaces equidistant bins)
- **Where**: `scgpt/preprocess.py:31`, `scgpt/model/model.py:93`
- **Problem**: scRNA-seq is ~90% zeros. Equidistant bins collapse all zeros into bin 0; the model learns to predict the mean bin.
- **Fix**: Bin over the non-zero expression distribution per gene (quantile-based), so each bin receives a meaningful fraction of observations.

#### 2. Attention Pooling for Cell Embeddings (replaces CLS token)
- **Where**: `scgpt/model/model.py:211`
- **Problem**: `cell_emb = layer_output[:, 0, :]` — CLS aggregation is suboptimal when the attention mask is generative/causal.
- **Fix**: Learned query vector attending over all gene token outputs (Pooling by Multihead Attention / PMA). Directly addresses the poor zero-shot clustering shown in the Genome Biology paper.

#### 3. Zero-Shot Silhouette Monitoring During Pretraining
- **Where**: `scgpt/trainer.py` training loop
- **Problem**: Only reconstruction loss is monitored. The zero-shot paper shows this does not correlate with useful cell embeddings.
- **Fix**: Add a periodic callback that computes average silhouette score on a fixed held-out labeled set. Use as an early-stopping / model-selection signal alongside reconstruction loss.

#### 4. ESM-2 Gene Embedding Initialisation
- **Where**: `scgpt/model/model.py:86` — `GeneEncoder`
- **Problem**: Gene embeddings are randomly initialised with no biological prior.
- **Fix**: Initialise from ESM-2 (Meta's protein language model) embeddings for the ~20,000 human genes. Genes with similar evolutionary profiles (paralogs, gene families) start with similar embeddings. Low effort, meaningful biological prior.

---

### Priority 2 — Medium Effort, High Impact

#### 5. Replace MLM with InfoNCE Contrastive + ZINB Reconstruction
- **Where**: `scgpt/loss.py`, `scgpt/trainer.py:36-41`
- **Problem**: Masked MSE on bins is weak. The existing ECS loss (`self.sim = Similarity(temp=0.5)`) uses a hardcoded temperature and threshold.
- **Fix**:
  - Replace binned reconstruction with **ZINB likelihood** loss: `L = -log P_ZINB(x | π, μ, θ)` — matches the actual data distribution including dropout zeros.
  - Add **InfoNCE / NT-Xent contrastive loss**: augmented views of the same cell (random masking already done) as positives; other cells in batch as negatives. Replace fixed `ecs_threshold=0.3` with learnable temperature.

#### 6. Batch-Agnostic Regularisation During Pretraining
- **Where**: `scgpt/model/model.py:105-113` (DSBN, DAB)
- **Problem**: DSBN and DAB require known batch labels at inference time — unavailable in zero-shot settings. The zero-shot paper shows batch effects dominate scGPT's embedding space.
- **Fix**: Add **Maximum Mean Discrepancy (MMD)** loss during pretraining to enforce batch-invariant cell embeddings without requiring batch labels at inference. Keep DSBN/DAB for fine-tuning.

#### 7. Remove Generative Attention Mask → Bidirectional Attention
- **Where**: `scgpt/model/model.py` — the specialised attention mask
- **Problem**: Causal/generative masking forces an arbitrary gene ordering and means each gene only sees prior genes in the sequence. Genes are exchangeable — there is no prior gene. This also creates a train/test mismatch since `TransformerEncoderLayer` at fine-tuning is bidirectional.
- **Fix**: Switch pretraining to full bidirectional attention (BERT-style MLM, no causal mask). Each gene attends to all others, matching biological reality.

---

### Priority 3 — Architectural Redesign

#### 8. Set Transformer Architecture (replaces sequence transformer)
- **Problem**: Transformers with positional encoding (or causal masking) impose sequence structure on set-valued data.
- **Fix**: Replace `TransformerModel` with a **Set Transformer**:
  - **Induced Set Attention Blocks (ISAB)**: m learned inducing points, O(n·m) attention instead of O(n²)
  - **Pooling by Multihead Attention (PMA)**: learned query vector aggregates the gene set into the cell embedding — strictly better than CLS
  - No positional encoding, full permutation invariance by design
- **Reference**: Lee et al., NeurIPS 2019. scTab (2023) already validated this for cell type annotation.

#### 9. GRN-Conditioned GNN for Perturbation (replaces condition tokens)
- **Where**: perturbation conditioning in `scgpt/model/model.py:102-103`
- **Problem**: Perturbation is encoded as a condition token (same mechanism as batch labels). Predictions don't vary across perturbations. The perturbation critique paper shows a simple linear model beats scGPT on all benchmarks.
- **Core issue**: Perturbation prediction requires causal inference (`do(gene=0)`), not observational correlation. Atlas pretraining on 33M normal cells cannot learn interventional effects on cancer cell lines.
- **Fix**:
  - Short term: Dedicated **perturbation cross-attention layer** where perturbation embeddings multiplicatively gate gene representations (not additive).
  - Long term: **GRN-conditioned GNN** — represent gene regulatory network (STRING/RegNetwork/SCENIC) as a directed graph. Perturbation = removing edges from knocked-out gene. Propagate effects downstream via message passing.
  - Data: Add a **perturbation-specific pretraining stage** on Perturb-seq datasets (Norman, Replogle, Adamson). The perturbation critique shows that pretraining on perturbation data helps perturbation prediction far more than atlas pretraining.

#### 10. Hierarchical VAE for Gene Program Representations
- **Problem**: Operating at individual gene level is noisy. Gene programs (co-regulated modules) are the real functional units.
- **Fix**: Hierarchical latent structure:
  - Level 1: Individual gene expression (ZINB decoder)
  - Level 2: Gene program weights (sparse, continuous)
  - Level 3: Cell type / state (discrete or mixture)
- This would improve zero-shot embeddings by encoding cells at the right biological granularity. Reference: scETM, f-scLVM.

---

## Proposed Target Architecture

```
Input: gene set {(gene_id, raw_count)} — no ordering
Gene embeddings: initialised from ESM-2 protein LM

┌─────────────────────────────────────────┐
│         SET TRANSFORMER ENCODER         │
│  ISAB blocks (inducing points)          │
│  O(n·m) attention, no positional enc.  │
└──────────────┬──────────────────────────┘
               │
    ┌──────────┴──────────┐
    │                     │
┌───▼───────┐    ┌────────▼───────────────┐
│  CELL EMB │    │  GENE PROGRAM LAYER    │
│  PMA pool │    │  Sparse inducing pts   │
└───┬───────┘    └────────┬───────────────┘
    │                     │
┌───▼─────────────────────▼───────────────┐
│        PRETRAINING OBJECTIVES           │
│  1. ZINB reconstruction (masked genes)  │
│  2. InfoNCE contrastive (cell-level)   │
│  3. MMD batch invariance               │
└─────────────────────────────────────────┘
               │
    Fine-tuning heads:
    ├── Cell type classifier
    ├── Batch integration (scVI-style)
    └── Perturbation: GRN-conditioned GNN
```

---

## Honest Assessment

No architecture will fully solve perturbation prediction from observational pretraining alone — the causal gap is fundamental. The perturbation critique's key finding: **pretraining on Perturb-seq data outperforms cell atlas pretraining for perturbation tasks**.

Right now, scVI (embeddings) + GEARS (perturbation) + task-specific models still outperform scGPT zero-shot on most benchmarks. Any new architecture must beat this baseline without fine-tuning to claim genuine foundation model status.

The field is likely 2–3 years away from a genuinely general single-cell foundation model. The path there requires:
1. Task-specific pretraining corpora (not one universal pretraining for all tasks)
2. Causal structure as inductive bias (GRNs), not learned from scratch
3. Set-based architecture with proper distributional losses (ZINB)
4. Holdout benchmark datasets never used for pretraining (analogous to MMLU for LLMs)

---

## References

- scGPT: Cui et al., *Nature Methods* 21, 1470–1480 (2024) — `papers/s41592-024-02201-0.pdf`
- Zero-shot critique: Kedzierska et al., *Genome Biology* 26:101 (2025) — `papers/13059_2025_Article_3574.pdf`
- Perturbation critique: Ahlmann-Eltze et al., *Nature Methods* 22, 1657–1661 (2025) — `papers/s41592-025-02772-6.pdf`
- Set Transformer: Lee et al., NeurIPS 2019
- CellOT (optimal transport for perturbation): Bunne et al., NeurIPS 2021
- scVI (ZINB VAE baseline): Lopez et al., *Nature Methods* 15, 1053–1058 (2018)
- ESM-2 (protein LM for gene init): Lin et al., *Science* 379, 1123–1130 (2023)
