# Deep Coffee Project

Professional deep learning repository for **coffee bean image classification** with four classes:

- dark
- green
- light
- medium

This project is designed for academic rigor and production-style engineering, with a staged roadmap from baseline modeling to deployment.

## Problem Statement

Coffee bean quality and roast characteristics are visually distinguishable but hard to classify consistently at scale. This repository provides a reproducible deep learning workflow for classifying coffee bean images into four roast/color categories.

## Objective

- Build a robust baseline in Stage 1 (MLP).
- Evolve into stronger architectures (CNN, transfer learning).
- Add generative components for data enhancement.
- Reach partial fine-tuning and deployment-ready inference.

## Repository Structure

```text
deep_coffee_project/
├── configs/                # Central configuration and logging
├── data/                   # raw, interim, processed datasets
├── figures/                # Saved plots and visual diagnostics
├── models/                 # Saved model checkpoints
├── notebooks/              # Stage-by-stage exploratory notebooks
├── reports/                # Technical and presentation outlines
├── scripts/                # CLI scripts for train/eval/demo
├── src/                    # Reusable source code (data/model/train/eval/deploy)
├── tests/                  # Unit tests
├── Makefile                # Development shortcuts
├── pyproject.toml          # Project metadata and tooling config
└── requirements.txt        # Python dependencies
```

## Installation

### 1) Create environment (Python 3.11+)

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

Or with Make:

```bash
make install
```

## Expected Dataset Structure

Place your data under:

```text
data/raw/coffee_beans/
├── train/
│   ├── dark/
│   ├── green/
│   ├── light/
│   └── medium/
└── test/
	├── dark/
	├── green/
	├── light/
	└── medium/
```

Supported formats: `.jpg`, `.jpeg`, `.png`

## Stage 1: Train Baseline MLP

```bash
python scripts/train_stage1.py
```

or

```bash
make train
```

Outputs:

- `models/stage1_mlp_best.pt`
- `results/stage1_training_history.json`
- `figures/stage1_training_curves.png`

## Stage 1: Evaluate on Test Set

```bash
python scripts/evaluate_stage1.py
```

or

```bash
make eval
```

Outputs:

- `results/stage1_test_metrics.json`
- `figures/stage1_confusion_matrix.png`

## Key Metrics Tracked

- Accuracy
- F1 weighted
- F1 macro
- Classification report
- Confusion matrix
- Training curves

## Five-Stage Roadmap

1. **Stage 1** — Baseline MLP + EDA + reproducible training/evaluation
2. **Stage 2** — Specialized CNN architectures
3. **Stage 3** — Transfer learning with pretrained backbones
4. **Stage 4** — Generative component for data augmentation/analysis
5. **Stage 5** — Partial fine-tuning and deployment (Gradio/Streamlit)

## Reproducibility and Engineering Practices

- Centralized dataclass config (`configs/config.py`)
- Deterministic seed setup (`src/utils/seed.py`)
- Minimal script entrypoints (`scripts/`)
- Modular source code in `src/`
- Test suite with `pytest`

## Future Work

- Hyperparameter sweep infrastructure
- Model registry and experiment tracking
- Hugging Face model integration
- Deployment hardening with monitoring and CI/CD

## License

MIT License (see `LICENSE`).
