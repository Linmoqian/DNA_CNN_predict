# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

基于 CNN + 注意力机制的 DNA 序列基因表达预测模型。使用 GM12878 细胞系数据（hg19 坐标系），预测基因为高表达或低表达（二分类）。训练 10 个 epoch 后准确率约 81%。

## 目录结构

```
DNA_CNN_predict/
├── data/          # 数据文件 + data.yaml 配置
├── docs/          # 文档（数据说明等）
├── model/         # 模型定义（modelv1.py 为参考版本）
├── script/        # 训练/评估脚本
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
python script/train.py
```

## Architecture

参考版本: `model/modelv1.py`（原 start.py），核心组件:

- **ConvModel**: CNN 模型，双分支输入
  - promoter 分支: 2层 Conv1d(4→32→64) + BN + MaxPool + 注意力机制(sigmoid 加权)
  - halflife 分支: 全连接(8→32)
  - 融合层: cat(promoter_flat, halflife) → FC(→128) → Dropout(0.8) → FC(→2)
- **训练流程**: Adam(lr=1e-4) + CrossEntropyLoss + ReduceLROnPlateau + 10 epochs + batch_size=32
- **评估指标**: Accuracy, AUC, F1 Score

## Dependencies

PyTorch, h5py, scikit-learn, numpy, matplotlib
