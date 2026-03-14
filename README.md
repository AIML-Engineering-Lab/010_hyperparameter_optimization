# Post 010 — Hyperparameter Optimization: Grid, Random, Bayesian (Optuna)

**AI Engineering Lab Series** | Era 1: Classic Machine Learning

## Overview

This project demonstrates the three main hyperparameter optimization strategies — Grid Search, Random Search, and Bayesian Optimization (Optuna) — applied to two regression problems with 5-dimensional search spaces.

| Method | Strategy | Evaluations | Best For |
|---|---|---|---|
| **Grid Search** | Exhaustive — tries every combination | All combinations | Small search spaces |
| **Random Search** | Random sampling from distributions | Fixed budget | Medium spaces |
| **Bayesian (Optuna)** | Builds a surrogate model, samples intelligently | Fixed budget | Large/expensive spaces |

## Datasets

### Dataset A: 3D Printer Extrusion Quality
- **Rows:** 5,000 | **Target:** Print quality score (0-1)
- **Task:** Find the optimal combination of temperature, speed, and layer height

### Dataset B: PLL Loop Filter Tuning (Post-Silicon Validation)
- **Rows:** 5,000 | **Target:** Lock time (µs) — minimize
- **Task:** Find optimal PLL parameters to minimize phase-locked loop lock time

## Quick Start

```bash
git clone https://github.com/AIML-Engineering-Lab/010_hyperparameter_optimization.git
cd 010_hyperparameter_optimization
pip install -r requirements.txt
python src/data_generator.py
jupyter notebook notebooks/
```

*Part of the [AI Engineering Lab](https://github.com/AIML-Engineering-Lab) series.*
