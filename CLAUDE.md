# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

基于 CNN + Transformer 的 DNA 序列基因表达预测模型。使用 GM12878 细胞系数据（hg19 坐标系），预测基因为高表达或低表达（二分类）。

## 模型版本

| 版本 | 架构 | 参数量 | 测试 Acc | 文件 |
|------|------|-------|---------|------|
| v1 | CNN + Attention | 41M | 0.81 | model/modelv1.py |
| v2 | 多尺度CNN + 空洞卷积 | 22K | 0.79 | model/modelv2.py |
| **v3** | **CNN + 1层Transformer (d48)** | **45.7K** | **0.8131** | model/modelv3.py |
| v4 | 三分支(promoter+halflife+epigenomic) | 96.8K | 0.8273 | model/modelv4.py |

核心发现：数据量仅 16K 样本时，过拟合是主要瓶颈。数据增强（+0.018）和 TTA（反向互补预测）是最有效的提升手段。
v4 新增 epigenomic 分支（ENCODE H3K4me3/H3K27ac/DNase），消融实验确认表观信号是最大增量（+2.9%），全特征 Acc 0.8273。

## 目录结构

```
DNA_CNN_predict/
├── data/          # 数据文件 + data.yaml 配置 + epigenomic.pt
├── docs/          # 文档（数据说明等）
├── logs/          # 实验日志（experiments.csv, fold checkpoints）
├── model/         # 模型定义（modelv1/v2/v3/v4）
├── script/        # 训练脚本（train_v1/v2/v3/v4, prepare_epigenomic）
└── utils/         # 工具函数（augment.py, feature_engineering.py, encode_downloader.py）
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
python script/train_v3.py              # 增强 + AMP + Label Smoothing + TTA
python script/train_v3.py --no-augment  # 无增强基线
python script/xai_analyze.py           # XAI 可解释性分析（DeepLIFT + 注意力 + 6张图表）

# v4: 额外特征实验
python script/prepare_epigenomic.py              # 下载 ENCODE 数据 + 提取表观特征 → data/epigenomic.pt
python script/train_v4.py --features baseline    # v3 等价基线
python script/train_v4.py --features seq         # +序列内在特征(GC/CpG)
python script/train_v4.py --features encode      # +ENCODE 表观信号
python script/train_v4.py --features all         # 全特征
```

## v3 Architecture (GeneExpressTransformer)

`model/modelv3.py`，45,674 参数：

- **CNN 前端**: 多尺度并行卷积(k=8/16/32, 各24通道) → Pool(8) → 融合Conv(72→48) → Pool(8)
- **Token 压缩**: AdaptiveAvgPool1d(64) + 可学习位置编码(1, 64, 48)
- **Transformer**: 1层 EncoderLayer(d_model=48, nhead=4, ff=96, dropout=0.1, gelu)
- **Halflife 分支**: FC(8→48→48)
- **分类头**: cat(promoter_GAP, halflife) → FC(96→48) → Dropout(0.5) → FC(48→2)

## 训练配置

- **优化器**: Adam(lr=5e-5, weight_decay=1e-4)
- **调度器**: ReduceLROnPlateau(mode=min, factor=0.5, patience=3)
- **早停**: patience=8, max 35 epochs
- **数据增强**: 反向互补(50%) + 随机平移(±200bp) + 随机遮蔽(1%)
- **Label Smoothing**: 0.1
- **AMP**: 混合精度训练（GPU 自动启用）
- **TTA**: 测试时正向+反向互补 logits 取平均
- **seed**: 42

## 已验证无效的方法（16K样本下）

- K-Fold 集成：每折样本更少，反而降低性能
- SWA 随机权重平均：小样本下效果不佳
- Mixup 样本插值：进一步稀释训练信号
- 大 batch(128) + CosineWarmRestarts：步数不足 + LR 衰减过快

## Dependencies

PyTorch, h5py, scikit-learn, numpy, captum, seaborn, scipy, statsmodels, logomaker, pyBigWig
