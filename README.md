# Privacy-Preserving Synthetic Tabular Data Generation

Major Technical Project (MTP), B.Tech Final Year (Aug 2025 - Apr 2026)

This project focuses on generating high-quality synthetic tabular data for financial domain while preserving privacy.

We implemented and compared three major pipelines:

1. CTAB-GAN+DP
2. FinDiff
3. HARPOON

We also unified evaluation across models using the metric framework from TabStruct and introduced two key novelties to improve utility, privacy, and structural fidelity.

## Project Highlights

- Implemented three research-paper-based synthetic data frameworks:
	- CTAB-GAN+ with differential privacy
	- FinDiff (diffusion-based tabular synthesis)
	- HARPOON
- Built a combined evaluation workflow by adapting TabStruct-style metrics and integrating them across all model outputs.
- Added novelty in model design and training:
	- Transformer-based cross-attention for conditioning input vectors.
	- Adaptive embeddings with DP-SGD and ghost clipping.
- Added novelty in optimization:
	- Hyperparameter tuning for FinDiff with Bayesian optimization.

## Repository Structure

Top-level folders in this workspace:

- `CTAB-GAN-Plus-DP-main/`:
	CTAB-GAN+ codebase, scripts, real/fake datasets, and model artifacts.
- `Harpoon-master/`:
	HARPOON training/sampling scripts, utilities, experiments, and requirements.
- `Tabstruct/`:
	TabStruct-based structural fidelity and evaluation notebook(s).
- `Datasets/`:
	Source datasets (for example `adult.csv`, `default.csv`).
- `Adult results/`, `Default results/`:
	Generated outputs and experiment results grouped by dataset/model family.
- `Novelty/`:
	Experiment outputs for novel model and optimization contributions.
- `Presentations/`, `Research Papers/`:
	Supporting project material.

## Implemented Baselines

### 1) CTAB-GAN+DP

- Used CTAB-GAN+ as a GAN-based tabular synthesizer.
- Extended with differential privacy training setup.
- Source code and scripts are under `CTAB-GAN-Plus-DP-main/`.

### 2) FinDiff

- Diffusion-based tabular data synthesis pipeline.
- Used as a core baseline and as a target for advanced hyperparameter tuning.
- Related outputs can be found in dataset-specific result folders (for example, `Adult results/Findiff/`).

### 3) HARPOON

- Implemented and experimented with HARPOON variants and sampling/training scripts.
- Main implementation is in `Harpoon-master/`.

## Unified Evaluation Framework

To make comparisons fair and reproducible across all three model families, we used TabStruct-based evaluation ideas and combined metrics into one common analysis flow.

Evaluation includes:

- Data fidelity and distribution-level similarity.
- Structural fidelity metrics inspired by TabStruct.
- Downstream utility and divergence-style comparisons (where applicable in experiment notebooks).

Relevant evaluation artifacts are available in:

- `Tabstruct/`
- `Adult results/Findiff/`
- `Default results/`

## Novel Contributions

### Novelty 1: Architecture + Privacy Enhancements

We introduced a transformer-style cross-attention mechanism to pass and condition on input vectors more effectively, then combined it with adaptive embeddings.

Privacy-aware training additions:

- DP-SGD integration
- Ghost clipping for stable and efficient per-sample gradient clipping

Objective:

- Improve synthetic data quality under privacy constraints.
- Better retain complex feature relationships without exposing sensitive records.

### Novelty 2: FinDiff Hyperparameter Optimization

We performed targeted hyperparameter tuning on FinDiff using Bayesian optimization.

Objective:

- Improve convergence and sample quality.
- Find robust configurations across datasets with fewer trial runs than grid/random search.

## Typical Workflow

1. Prepare dataset from `Datasets/` (or dataset-specific copies in model folders).
2. Train/generate synthetic data using one of:
	 - `CTAB-GAN-Plus-DP-main/`
	 - `Harpoon-master/`
	 - FinDiff scripts/notebooks in result folders
3. Store generated outputs in corresponding `Adult results/` or `Default results/` directories.
4. Run evaluation notebooks/scripts (including TabStruct metrics) to compare models.
5. Analyze novelty experiments from `Novelty/` and summarize trade-offs.

## Quick Start

Because this repository combines multiple research codebases, dependencies are managed per submodule.

### A) HARPOON environment

```bash
cd Harpoon-master
pip install -r requirements.txt
```

Use provided shell scripts (for example `train_baselines.sh`, `sample_baselines.sh`) or run Python training/sampling scripts directly.

### B) CTAB-GAN+DP environment

```bash
cd CTAB-GAN-Plus-DP-main
# install dependencies based on this folder's README / project setup
```

Then run the experiment script(s), for example:

- `Experiment_Script_Adult.ipynb`
- `run_ctab_gan`

### C) FinDiff + evaluation notebooks

Use notebooks/scripts under result folders and `Tabstruct/` for metric computation and comparison.

## Datasets Used

- Adult dataset
- Default dataset

Location:

- `Datasets/adult.csv`
- `Datasets/default.csv`

Additional dataset copies may appear in model/result folders for experiment isolation.

## Outputs and Artifacts

The repository contains:

- Synthetic datasets generated by each method.
- Model checkpoints (`.pth`) for selected runs.
- Evaluation CSVs and notebooks with metric results.
- Comparative results organized by dataset and method.

## Reproducibility Notes

- Fix random seeds in each framework where possible.
- Keep train/eval splits consistent across models.
- Use the same preprocessing for fair cross-model comparisons.
- Document exact hyperparameters for Bayesian-optimized FinDiff runs.
