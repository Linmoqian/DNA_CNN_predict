# DNA_CNN_predict

**[中文](README.md)**

A lightweight CNN-Transformer hybrid model for predicting gene expression from DNA promoter sequences. Using GM12878 cell line data (hg19 coordinates) to classify genes as high or low expression (binary). Through 4 iterations, the final model (v4) achieves 82.74% test accuracy with only 96.8K parameters.

## Model Versions

| Version | Architecture | Parameters | Test Acc | File |
|---------|-------------|------------|---------|------|
| v1 | CNN + Attention | 41M | 0.81 | `model/modelv1.py` |
| v2 | Multi-scale CNN + Dilated Conv | 22K | 0.79 | `model/modelv2.py` |
| v3 | CNN + 1-layer Transformer | 45.7K | 0.8131 | `model/modelv3.py` |
| **v4** | **Three-branch (promoter+halflife+epigenomic)** | **96.8K** | **0.8274** | `model/modelv4.py` |

Key finding: With only 16K samples, overfitting is the main bottleneck. Lightweight design (96.8K params) outperforms large models (41M), and ENCODE epigenomic signals contribute +2.9% accuracy.

## Quick Start

### 1. Install Dependencies

```bash
conda env create -f environment.yml
conda activate dna-cnn
```

### 2. Prepare Data

Download the GM12878 dataset from AISCCC:

```bash
# 1. Visit http://www.aisccc.cn/database/data-details?id=121 to download the dataset
# 2. Extract and place train.h5, valid.h5, test.h5 into data/
# 3. Verify data integrity
python script/setup_data.py --check
```

### 3. Train Models

**v3 baseline** (no extra data needed):

```bash
python script/train_v3.py              # Augmentation + AMP + Label Smoothing + TTA
python script/train_v3.py --no-augment  # No augmentation baseline
```

**v4 full features** (requires ENCODE data):

```bash
# Prepare ENCODE epigenomic signals (~880MB bigWig download → data/epigenomic.pt)
python script/prepare_epigenomic.py

# Pre-compute sequence features (GC/CpG → data/seq_features_*.pt)
python script/precompute_seq_features.py

# Ablation study
python script/train_v4.py --features baseline    # v3-equivalent baseline
python script/train_v4.py --features encode      # +ENCODE epigenomic signals
python script/train_v4.py --features all         # Full features (best)
```

### 4. Interpretability Analysis

```bash
python script/xai_analyze.py                    # v3 DeepLIFT + attention analysis
python script/xai_analyze_v4.py --features all  # v4 XAI + ENCODE channel contribution
```

### One-Click Run

```bash
bash run_all.sh              # Full pipeline (data check → v3 → ENCODE → v4 → XAI → figures)
bash run_all.sh --quick      # v3 baseline only
bash run_all.sh --step 4     # Start from v4 training
```

## Key Results

### Model Architecture — V4 Three-Branch Design

![V4 Architecture](paper/figures/fig1_architecture_v4.png)

### Ablation Study

| Config | Features | Test Acc |
|--------|----------|---------|
| baseline | promoter + halflife | 0.7959 |
| +ENCODE | baseline + H3K4me3/H3K27ac/DNase | 0.8252 |
| **all** | **full features** | **0.8274** |

### Interpretability Analysis

DeepLIFT attribution reveals that key regions learned by the model align with known regulatory elements (TATA box, CAAT box). Cross-validation with Integrated Gradients yields r = 0.92.

![Global Attribution](paper/figures/fig_global_attribution.png)

![Method Validation](paper/figures/fig_dl_ig_correlation.png)

## Dataset

- Source: http://www.aisccc.cn/database/data-details?id=121
- Format: HDF5 (train.h5, valid.h5, test.h5)
- Contents: `gene_id` (Ensembl ID), `halflife` (8-dim normalized features), `promoter` (20,000bp one-hot DNA sequence), `label` (0=low, 1=high expression)
- Encoding: `{'A':0, 'C':1, 'G':2, 'T':3}`
- Scale: Train 16,215 / Valid 989 / Test 990 samples

## Project Structure

```
DNA_CNN_predict/
├── data/          # Data files
├── logs/          # Experiment logs
├── model/         # Model definitions (v1-v4)
├── results/       # Results (XAI figures, CSVs)
├── script/        # Training/analysis scripts
└── utils/         # Utility functions
```

## Paper

Full paper (Chinese): [`paper/paper.md`](paper/paper.md) — 8 sections, 10 figures, 21 references (GB/T 7714 format).
