# 研究思路

## 一、问题分析

### 任务本质
二分类问题：给定基因的启动子 DNA 序列（20000bp）和 mRNA 半衰期特征（8维），预测该基因是高表达（label=1）还是低表达（label=0）。

### 输入特征
| 特征 | 维度 | 说明 |
|------|------|------|
| promoter | 4 × 20000 | TSS ±10000bp 的 one-hot 编码 |
| halflife | 8 | 标准化后的序列结构特征 |

### 核心难点
- 启动子序列长达 20000bp，有效调控元件可能只占很小片段
- 远距离碱基间存在依赖关系（增强子-启动子互作）
- 8 维半衰期特征与表达水平的非线性关系

---

## 二、子任务1：预测基因表达

### 2.1 模型架构：双分支融合网络

**设计思路**：DNA 序列和半衰期特征的信息密度差异大，采用双分支分别提取，再融合预测。

#### Promoter 分支：多尺度 CNN + Transformer

```
Input: (batch, 4, 20000)
  │
  ├─ Conv1d(4→64, k=8)  → BN → ReLU → MaxPool(2)   # 局部 motif 检测
  ├─ Conv1d(4→64, k=16) → BN → ReLU → MaxPool(2)   # 中等尺度特征
  └─ Conv1d(4→64, k=32) → BN → ReLU → MaxPool(2)   # 较大片段特征
  │
  concat → Conv1d(192→128, k=3) → BN → ReLU → MaxPool(4)
  │
  Reshape → Transformer Encoder (2层, 4头)            # 捕获长程依赖
  │
  Attention Pooling → (batch, 128)
```

**关键选择**：
- **多尺度卷积核**（8/16/32）：同时捕获短 motif（如 TF 结合位点 ~6-20bp）和中等尺度调控模式
- **Transformer Encoder**：比纯 CNN 更擅长建模远距离碱基间的交互关系
- **Attention Pooling**：自适应地关注重要位置，为子任务2的可解释性打基础

#### Halflife 分支

```
Input: (batch, 8)
  │
  FC(8→32) → ReLU → Dropout(0.3)
  FC(32→32) → ReLU
  │
  Output: (batch, 32)
```

#### 融合与分类

```
concat(promoter_feat, halflife_feat)  # (batch, 160)
  │
  FC(160→64) → ReLU → Dropout(0.5)
  FC(64→2)
  │
  Output: logits
```

### 2.2 训练策略

| 项目 | 方案 |
|------|------|
| 损失函数 | CrossEntropyLoss + Label Smoothing (0.1) |
| 优化器 | AdamW, lr=1e-3, weight_decay=1e-4 |
| 学习率调度 | CosineAnnealingWarmRestarts (T_0=10) |
| 正则化 | Dropout + BatchNorm + weight_decay |
| 训练轮次 | 50 epochs, early stopping (patience=10) |
| 批大小 | 64 |

### 2.3 数据处理
- promoter 输入保持原始 one-hot 格式
- halflife 已标准化，直接使用
- 按 train/valid/test 划分，**测试集绝不参与训练**

---

## 三、子任务2：模型可解释性分析

### 目标
识别启动子 DNA 序列中与高表达相关的碱基片段。

### 方法：梯度加权注意力图 (Grad-Attention)

1. **Attention 权重提取**：从 Promoter 分支的 Transformer 和 Attention Pooling 层获取 attention 权重矩阵
2. **梯度加权**：对高表达类别（label=1）的预测输出求梯度，用梯度对 attention 权重加权
3. **位置重要性得分**：将加权后的得分映射回 20000bp 序列的每个位置
4. **片段识别**：对重要性得分进行峰值检测，提取连续高得分区域作为关键碱基片段

### 输出
- 每个测试样本的位置重要性热图
- 高表达相关的 Top-K 关键片段列表（起止位置 + 重要性得分）
- 与已知转录因子结合位点数据库（JASPAR）的比对分析

---

## 四、子任务3：增加额外数据特征

### 候选特征（按可行性排序）

| 优先级 | 特征 | 来源 | 预期收益 |
|--------|------|------|----------|
| 1 | 转录因子结合位点 (ChIP-seq) | ENCODE / ChIP-Atlas | 高，直接调控表达 |
| 2 | 组蛋白修饰 (H3K4me3, H3K27ac) | ENCODE GM12878 | 高，标记活跃启动子 |
| 3 | DNA 甲基化 | ENCODE / Roadmap | 中，表观沉默标记 |
| 4 | GC 含量 (sliding window) | 序列本身直接计算 | 中，启动子活性相关 |
| 5 | 染色质开放性 (ATAC-seq/DNase) | ENCODE GM12878 | 高，标志可及区域 |

### 整合方式
- 将额外特征作为第三分支输入，或与 promoter 特征在通道维度拼接
- 转录因子结合位点：按位置对齐到 promoter 序列，生成 binary mask
- 组蛋白修饰：在 promoter 区域内的信号强度，生成连续值向量

---

## 五、评估方案

### 主要指标
- **AUC**：衡量模型区分高/低表达的整体能力
- **F1-Score**：衡量分类精确度，兼顾 precision 和 recall

### 辅助指标
- 参数量（模型复杂度）
- 训练时间 / 推理时间
- 可解释性结果与已知生物学知识的吻合度

---

## 六、开发计划

1. **基线模型**：实现双分支 CNN+Transformer 架构，跑通训练流程
2. **调优**：超参搜索、消融实验（验证多尺度 CNN、Transformer 各自贡献）
3. **可解释性**：实现 Grad-Attention 方法，生成测试集的关键片段
4. **额外特征**：获取 ENCODE GM12878 数据，整合到模型中
5. **文档与提交**：整理代码、模型、结果报告
