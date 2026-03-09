# Papers

Key papers for understanding scGPT and its evaluation.

## scGPT (Original)

**scGPT: toward building a foundation model for single-cell multi-omics using generative AI**
Cui, Wang, Maan, Pang, Luo, Duan & Bo Wang — *Nature Methods* 21, 1470–1480 (August 2024)
DOI: [10.1038/s41592-024-02201-0](https://doi.org/10.1038/s41592-024-02201-0)
File: `s41592-024-02201-0.pdf`

Introduces scGPT: a generative pretrained transformer trained on 33 million normal human cells (CELLxGENE, 51 organs, 441 studies). Claims state-of-the-art performance on cell type annotation, multi-batch integration, perturbation prediction, multi-omic integration, and GRN inference — all with fine-tuning.

---

## Critique 1: Zero-Shot Evaluation

**Zero-shot evaluation reveals limitations of single-cell foundation models**
Kedzierska, Crawford, Amini & Lu — *Genome Biology* 26:101 (2025)
DOI: [10.1186/s13059-025-03574-x](https://doi.org/10.1186/s13059-025-03574-x)
File: `13059_2025_Article_3574.pdf`

Evaluates scGPT and Geneformer in zero-shot settings (no fine-tuning) on cell type clustering and batch integration across 5 datasets. Key findings: both models are outperformed by simply selecting highly variable genes (HVG); scGPT barely beats mean prediction on its own pretraining task; larger pretraining datasets do not reliably help; models fail even on datasets seen during pretraining.

---

## Critique 2: Perturbation Prediction

**Deep-learning-based gene perturbation effect prediction does not yet outperform simple linear baselines**
Ahlmann-Eltze, Huber & Anders — *Nature Methods* 22, 1657–1661 (August 2025)
DOI: [10.1038/s41592-025-02772-6](https://doi.org/10.1038/s41592-025-02772-6)
File: `s41592-025-02772-6.pdf`

Benchmarks scGPT and other foundation models on perturbation prediction (Norman double, Replogle, Adamson datasets). None outperform a simple linear model or even the mean of training examples. scGPT predictions do not vary meaningfully across perturbations. The linear baseline used in the original scGPT paper was a strawman that always predicted "no change" for unseen perturbations.
