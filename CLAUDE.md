# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

基于 CNN + Transformer 的 DNA 序列基因表达预测模型。使用 GM12878 细胞系数据（hg19 坐标系），预测基因为高表达或低表达（二分类）。

## 模型版本

| 版本 | 架构 | 参数量 | 测试 Acc | 文件 |
|------|------|-------|---------|------|
| v1 | CNN + Attention | 41M | 0.81 | model/modelv1.py |
| v2 | 多尺度CNN + 空洞卷积 | 22K | 0.79 | model/modelv2.py |
| **v3** | **CNN + 1层Transformer** | **22.5K** | **0.805** | model/modelv3.py |

v3 为当前最优，核心发现：数据量仅 16K 样本时，过拟合是主要瓶颈，1 层 Transformer 比 2 层泛化更好。

## 目录结构

```
DNA_CNN_predict/
├── data/          # 数据文件 + data.yaml 配置
├── docs/          # 文档（数据说明等）
├── model/         # 模型定义（modelv1/v2/v3）
├── script/        # 训练脚本（train_v1/v2/v3）
└── utils/         # 工具函数
```

## Data

- 数据来源: http://www.aisccc.cn/database/data-details?id=121
- 格式: HDF5（train.h5, valid.h5, test.h5）
- 数据配置: `data/data.yaml`
- 每个 HDF5 文件包含: `gene_id`(Ensembl ID), `halflife`(8维标准化特征), `promoter`(20000bp one-hot DNA序列), `label`(0=低表达, 1=高表达)
- promoter 的 one-hot 编码: `{'A':0, 'C':1, 'G':2, 'T':3}`
- halflife 8 维特征顺序: UTR5LEN, CDSLEN, INTRONLEN, UTR3LEN, UTR5GC, CDSGC, UTR3GC, ORFEXONDENSITY

## Run

```bash
# 需要先将 train.h5, valid.h5, test.h5 放入 data/ 目录
python script/train_v3.py
```

## v3 Architecture (GeneExpressTransformer)

`model/modelv3.py`，22,514 参数：

- **CNN 前端**: 多尺度并行卷积(k=8/16/32, 各16通道) → Pool(8) → 融合Conv(48→32) → Pool(8)
- **Token 压缩**: AdaptiveAvgPool1d(64) + 可学习位置编码(1, 64, 32)
- **Transformer**: 1层 EncoderLayer(d_model=32, nhead=4, ff=64, dropout=0.1, gelu)
- **Halflife 分支**: FC(8→32→32)
- **分类头**: cat(promoter_GAP, halflife) → FC(64→32) → Dropout(0.5) → FC(32→2)

训练配置: Adam(lr=5e-5, weight_decay=1e-4) + ReduceLROnPlateau + early stopping(patience=8) + seed=42

## Dependencies

PyTorch, h5py, scikit-learn, numpy
