# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

scGPT is a foundation model for single-cell multi-omics analysis. It uses transformer-based architecture to process gene expression data for tasks like cell type annotation, perturbation prediction, multi-omics integration, and GRN inference.

## Installation

```bash
pip install scgpt "flash-attn<1.0.5"
# If dependency conflicts occur:
# pip install scgpt "flash-attn<1.0.5" "orbax<0.1.8"
```

For development (note: poetry.lock may be out of sync with pyproject.toml):
```bash
pip install -e .
```

## Common Commands

### Run tests
```bash
pytest                          # all tests
pytest tests/test_tokenizer.py  # single test file
pytest tests/ -k "test_name"    # single test by name
```

### Format code
```bash
black scgpt/
```

## Architecture

### Core Pipeline
Data → `Preprocessor` → `GeneVocab` tokenization → `TransformerModel` → task-specific head

### Key Modules

**`scgpt/model/`** — Model definitions:
- `model.py`: `TransformerModel` — main encoder. Accepts gene tokens + expression values. Supports three cell embedding styles: `"cls"`, `"avg-pool"`, `"w-pool"`. Optional domain-specific batch normalization (DSBN) and flash attention.
- `multiomic_model.py`: `MultiOmicTransformerModel` — extends TransformerModel for multi-modality inputs.
- `generation_model.py`: Generative/perturbation prediction models.

**`scgpt/tokenizer/gene_tokenizer.py`** — `GeneVocab` class manages the gene vocabulary. Pre-built vocabs live in `scgpt/tokenizer/default_gene_vocab.json` (general) and `default_census_vocab.json` (Census).

**`scgpt/trainer.py`** — `prepare_data()`, `prepare_dataloader()`, `train()`, `evaluate()`. Supports tasks: `"annotation"`, `"integration"`, `"perturb"`, `"multiomic"`. Integrates with W&B via `define_wandb_metrics()`.

**`scgpt/preprocess.py`** — `Preprocessor` class handles gene/cell filtering, normalization (total count, log1p), HVG selection, and binning.

**`scgpt/scbank/`** — `DataBank` manages multi-dataset collections on disk or in memory with metadata tracking.

**`scgpt/tasks/`** — Task utilities: `cell_emb.py` for extracting cell embeddings (`get_batch_cell_embeddings()`), `grn.py` for gene regulatory network inference.

**`scgpt/utils/util.py`** — `load_pretrained()` for loading checkpoints, `eval_scib_metrics()`, `set_seed()`, `get_free_gpu()`.

**`scgpt/loss.py`** — `masked_mse_loss()`, `criterion_neg_log_bernoulli()`, `masked_relative_error()`.

### Training Losses
- **MLM**: Masked language modeling on gene expression values
- **MVC**: Masked value correction
- **ECS**: Elastic cell similarity
- **DAB**: Domain adversarial batch correction

### Pretrained Models
Checkpoints are downloaded separately (Google Drive). Each checkpoint comes with a paired `vocab.json`. The recommended general-purpose checkpoint is `whole-human` (33M cells).

## Data Flow for Fine-tuning

1. Load AnnData (`.h5ad`) with scanpy
2. Run `Preprocessor` to filter/normalize/bin
3. Build `GeneVocab` from checkpoint's `vocab.json` via `GeneVocab.from_file()`
4. Tokenize: map gene names → integer IDs, expression values → bin integers
5. Load `TransformerModel` weights with `load_pretrained()`
6. Fine-tune via `trainer.py` or custom loop using `DataCollator`

See `tutorials/` for end-to-end Jupyter notebooks per task type.
