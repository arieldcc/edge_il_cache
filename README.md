# Edge IL Cache Experiments

This repository contains code for IL-based edge caching experiments (guard/no-guard variants), overhead measurement, and analysis notebooks.

## 1. Project Setup

Create the Conda environment from `environment.yml`:

```bash
conda env create -f environment.yml
```

Activate the environment:

```bash
conda activate edge_il_cache
```

## 2. Dataset Preparation

Important: the experiment scripts expect datasets under `data/raw/...` (not `data/row/...`).

Create dataset folders:

```bash
mkdir -p data/raw/wikipedia_september_2007
mkdir -p data/raw/wiki2018
```

Download datasets:

```bash
# Wikipedia September 2007
curl -L "http://www.globule.org/wiki/2007-09/wiki.1190153705.gz" \
  -o data/raw/wikipedia_september_2007/wiki.1190153705.gz

# Wiki2018 CDN trace (tar.gz)
curl -L "http://lrb.cs.princeton.edu/wiki2018.tr.tar.gz" \
  -o data/raw/wiki2018/wiki2018.tr.tar.gz
```

Convert `wiki2018.tr.tar.gz` to `wiki2018.gz` (10M prefix) using the provided script:

```bash
python scripts/convert_wiki2018.py
```

This script writes:

- `data/raw/wiki2018/wiki2018.gz`

## 3. Run Main Experiments

### 3.1 No Guard

```bash
python src/experiments/run_il_cache_opt022_guard_compare_nb_svm_dt.py \
  --model guard_no_guard \
  --feature-sets A2 \
  --base-learners nb \
  --datasets wikipedia_september_2007 wiki2018
```

### 3.2 Guard

```bash
python src/experiments/run_il_cache_opt022_guard_compare_nb_svm_dt.py \
  --model guard_full \
  --feature-sets A2 \
  --base-learners nb \
  --datasets wikipedia_september_2007 wiki2018
```

## 4. Run Overhead Experiments

```bash
python src/experiments/run_il_cache_overhead_guard.py \
  --dataset wiki2018 \
  --feature-set A2 \
  --base-learner nb

python src/experiments/run_il_cache_overhead_no_guard.py \
  --dataset wiki2018 \
  --feature-set A2 \
  --base-learner nb

python src/experiments/run_gdbt_cache_overhead.py \
  --dataset wiki2018 \
  --feature-set A2
```

## 5. Outputs

- Main experiment outputs: `results/<dataset>/...`
- Overhead outputs: `results/overhead/<dataset>/...`

## 6. Evaluation and Visualization

Evaluation and visualization notebooks are available in:

- `notebooks/`

You can open them with Jupyter Lab/Notebook after activating the Conda environment.
