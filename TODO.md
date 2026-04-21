# TODO

## 模型迭代
- [x] modelv1: CNN 基线 (41M参数, Acc 0.81)
- [x] modelv2: 多尺度CNN+空洞卷积 (22K参数, Acc 0.79)
- [x] modelv3: CNN+1层Transformer (22.5K参数, Acc 0.805)
- [x] modelv3 优化: d_model48 + 增强 + Label Smoothing + TTA (45.7K参数, Acc 0.8131)

## modelv3 实验记录（13组，已完成）
- [x] ExpA: +2Conv无Pool → 0.728（加深CNN损害性能）
- [x] ExpB: 1层Transformer → **0.805**（最优，减少过拟合）
- [x] ExpC: 2层+强正则化 → 0.803（正则化有效但不如减层）
- [x] ExpD: 残差块 → 0.790（增加复杂度无收益）
- [x] ExpE: 双路径特征 → 0.799（略好于基线）
- [x] ExpF~I, lr/wd/bs/seed 消融 → 均未超过 ExpB

## modelv3 二次优化实验（已完成）
- [x] 数据增强(反向互补/平移/遮蔽) → +0.018, Acc 0.804
- [x] AMP 混合精度 → 微升, Acc 0.8051
- [x] d_model48 架构扩展 + Label Smoothing + TTA → Acc 0.8131（当前最优）
- [x] K-Fold 5折集成 → 无效(0.8051)，样本太少
- [x] SWA + Mixup → 无效，已回退

## 可解释性分析（已完成）
- [x] XAI 模型变体：注意力权重捕获 + DeepLIFT + IG 归因
- [x] 6 张可视化图表（全局归因、TSS 放大、注意力热力图、序列 logo、关键区域、分组对比）
- [x] 关键碱基片段识别（2230 区域，top: 9775-9781bp）
- [x] DeepLIFT vs IG 相关性 0.907，TSS 区域高/低表达归因 2.2x 差异
- [x] CNN 卷积核 motif 提取（3 组滤波器权重）

## modelv4 额外特征实验
- [x] Phase A: 序列内在特征 (GC/CpG/k-mer) — utils/feature_engineering.py
- [x] Phase A: 三分支模型 v4 — model/modelv4.py
- [x] Phase A: v4 训练脚本 — script/train_v4.py
- [x] Phase B: ENCODE 数据下载器 — utils/encode_downloader.py
- [x] Phase B: 表观特征提取 — script/prepare_epigenomic.py
- [x] 消融实验: baseline(0.796) / +seq(0.799) / +encode(0.825) / +all(0.827)
- [x] 记录实验结果到 logs/experiments.csv
- [x] 特征实验报告 — docs/v4_feature_report.md

## 待探索
- [ ] 其他细胞系数据迁移学习
- [ ] 多 seed 集成（seed=0/1/42 投票）
