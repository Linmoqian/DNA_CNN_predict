# DNA_CNN_predict

基于 CNN + Transformer 的 DNA 启动子序列基因表达预测模型。使用 GM12878 细胞系数据（hg19 坐标系），预测基因为高表达或低表达（二分类）。通过 4 个版本的迭代优化，最终模型（v4）以 96.8K 参数达到 82.74% 测试准确率。

## 模型版本

| 版本 | 架构 | 参数量 | 测试 Acc | 文件 |
|------|------|--------|---------|------|
| v1 | CNN + Attention | 41M | 0.81 | `model/modelv1.py` |
| v2 | 多尺度CNN + 空洞卷积 | 22K | 0.79 | `model/modelv2.py` |
| v3 | CNN + 1层Transformer | 45.7K | 0.8131 | `model/modelv3.py` |
| **v4** | **三分支 (promoter+halflife+epigenomic)** | **96.8K** | **0.8274** | `model/modelv4.py` |

核心发现：数据量仅 16K 样本时，过拟合是主要瓶颈。轻量化设计（96.8K 参数）优于大模型（41M），ENCODE 表观信号贡献 +2.9% 准确率。

## 代表性成果

### 模型架构 — V4 三分支设计

![V4 架构](paper/figures/fig1_architecture_v4.png)

### 消融实验

| 配置 | 特征组合 | 测试 Acc |
|------|---------|---------|
| baseline | promoter + halflife | 0.7959 |
| +ENCODE | baseline + H3K4me3/H3K27ac/DNase | 0.8252 |
| **all** | **全特征** | **0.8274** |

### 可解释性分析

DeepLIFT 归因分析显示模型学习到的关键区域与已知调控元件（TATA box、CAAT box）高度吻合，DeepLIFT 与 IG 相关系数 r = 0.92。

![全局归因](paper/figures/fig_global_attribution.png)

![TSS 区域](paper/figures/fig_tss_zoom.png)

![方法验证](paper/figures/fig_dl_ig_correlation.png)

## 数据集

数据下载：http://www.aisccc.cn/database/data-details?id=121

格式：HDF5（train.h5, valid.h5, test.h5），包含 `gene_id`、`halflife`(8维)、`promoter`(20000bp one-hot)、`label`(0/1)。

## 快速开始

```bash
# 将 train.h5, valid.h5, test.h5 放入 data/ 目录

# v3 基线训练
python script/train_v3.py

# v4 表观信号预处理（需下载 ENCODE bigWig 文件）
python script/prepare_epigenomic.py

# v4 全特征训练
python script/train_v4.py --features all

# 可解释性分析
python script/xai_analyze.py
```

## 目录结构

```
DNA_CNN_predict/
├── data/          # 数据文件 + data.yaml + epigenomic.pt
├── docs/          # 论文(paper.md) + 图表 + XAI报告
├── logs/          # 实验日志(experiments.csv)
├── model/         # 模型定义(v1~v4)
├── paper/         # 独立论文目录(paper.md + figures/)
├── results/       # 实验结果(XAI图表、CSV)
├── script/        # 训练/分析脚本
└── utils/         # 工具函数
```

## 论文

完整论文见 [`paper/paper.md`](paper/paper.md)，包含 8 个章节、10 张图表、21 篇参考文献（GB/T 7714 格式）。

## Dependencies

PyTorch, h5py, scikit-learn, numpy, captum, seaborn, scipy, statsmodels, logomaker, pyBigWig
