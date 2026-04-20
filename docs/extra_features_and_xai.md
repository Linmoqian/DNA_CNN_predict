# 额外特征与可解释性分析

## 一、ENCODE 中 GM12878 可用数据

| 数据类型 | 说明 | 用途 |
|----------|------|------|
| ChIP-seq（组蛋白修饰） | H3K4me1/3, H3K27ac, H3K27me3, H3K36me3, H3K9me3 | 标注启动子、增强子、异染色质 |
| ChIP-seq（转录因子） | 55-120 种 TF 的结合数据 | 识别 TF 结合位点 |
| DNase-seq | DNase I 超敏感位点图谱 | 基因组范围染色质开放性 |
| ATAC-seq | Tn5 转座酶插入开放区域 | 染色质开放性 |
| WGBS/RRBS | 全基因组亚硫酸氢盐测序 | CpG 甲基化状态 |
| Hi-C / CHiA-PET | 三维染色质互作 | 增强子-启动子互作、TAD 结构 |
| RAMPAGE / CAGE | 转录起始位点标注 | 精确定位 TSS |

数据获取：[ENCODE Portal](https://www.encodeproject.org)（搜索 biosample "GM12878"）

---

## 二、转录因子结合位点（TFBS）

### 生物学意义
转录因子通过识别特定 DNA motif 结合到调控元件上，招募或阻断 RNA 聚合酶来调控转录。

### 获取流程
1. 从 ENCODE 下载 GM12878 的 TF ChIP-seq 峰值文件（BED 格式）
2. 用 bedtools intersect 将峰值与 TSS +/-10kb 取交集
3. 可进一步用 JASPAR PWM 扫描获得高分辨率结合位点

### GM12878 中活跃的关键 TF

| 转录因子 | 功能 |
|---------|------|
| CTCF | 绝缘子蛋白，染色质结构边界 |
| POLR2A | RNA 聚合酶 II，标记活跃转录 |
| EP300 | 组蛋白乙酰转移酶，转录共激活因子 |
| YY1 | 多功能 TF，参与增强子-启动子成环 |
| ELF1 | ETS 家族，淋巴细胞活跃 |
| IRF4 | 干扰素调节因子，B 细胞发育关键 |
| PAX5 | B 细胞分化主调控因子 |
| SPI1 (PU.1) | 造血系统关键 TF |
| EBF1 | 早期 B 细胞因子 |

---

## 三、组蛋白修饰详解

| 修饰 | 定位 | 与表达的关系 |
|------|------|-------------|
| **H3K4me3** | 活跃启动子，TSS 附近 | **激活** — 活跃启动子经典标志 |
| **H3K27ac** | 活跃启动子和活跃增强子 | **激活** — 区分活跃增强子与准备态 |
| **H3K4me1** | 增强子区域 | **增强子标记** — 与 H3K27ac 共存为活跃增强子 |
| **H3K27me3** | Polycomb 抑制区域 | **抑制** — 沉默基因表达 |
| **H3K9me3** | 结构性异染色质 | **抑制** — 永久沉默 |
| **H3K36me3** | 活跃转录延伸区域 | **激活** — 标记正在转录的基因体 |

### 关键规律
```
H3K4me3 + H3K27ac = 活跃启动子 → 高表达
H3K4me1 + H3K27ac = 活跃增强子 → 促进远距离基因表达
H3K4me1 单独     = 准备态增强子
H3K27me3         = Polycomb 沉默 → 低表达
H3K9me3          = 异染色质 → 永久沉默
```

H3K27ac 与 H3K27me3 呈此消彼长的拮抗关系，其动态切换是细胞命运转变的关键事件。

---

## 四、染色质开放性

### 含义
基因组中核小体排列松散、DNA 可被转录因子访问的区域。开放染色质是活跃调控元件的标志。

### 与表达的关系
- 启动子区域的染色质开放性是基因表达的必要前提
- 开放区域富集转录因子结合位点
- DNase/ATAC 信号强度与基因表达水平正相关
- 约 95% 的 DNase I 超敏感位点位于活性调控元件

### 推荐
对 GM12878 推荐使用 DNase-seq 数据（ENCODE 中最全面成熟）。

---

## 五、特征对齐与整合方案

### 流程
1. 从 gene_id（Ensembl ID）映射到 hg19 坐标，定义 TSS +/-10kb 窗口
2. 从 ENCODE 下载 bigWig 文件，用 pyBigWig 按窗口提取信号
3. 每种表观信号作为独立通道（1bp 或 bin 分辨率）

### 三种整合方案

| 方案 | 描述 | 优点 | 缺点 |
|------|------|------|------|
| **A. 多通道输入** | 表观信号拼接 one-hot（4+6=10 通道） | 简单直接 | 离散/连续信号性质不同 |
| **B. 多分支** | 序列 CNN + 表观独立分支 + halflife FC | 各分支独立处理，灵活 | 模型更复杂 |
| **C. 信号加权** | 表观信号对 one-hot 加权 | 保留序列结构 | 信息表达有限 |

**推荐方案 B**：与 halflife 分支设计理念一致，增加表观信号分支，最后在融合层拼接。

### TF 结合特征
- 用 bedtools intersect 计算窗口内各 TF 的结合位点数量
- 生成固定维度 TF 活性向量（如 20 个活跃 TF 的结合计数）
- 与 halflife 特征拼接

---

## 六、可解释性分析方法

### 6.1 DeepLIFT
- 将模型输出分解为各输入特征的贡献
- 与"参考输入"比较，衡量每个位置对预测的影响
- 碱基级别归因，计算速度快
- 工具：`captum` 库的 `DeepLift` 类

### 6.2 Integrated Gradients（积分梯度）
- 从 baseline 到实际输入之间积分梯度
- 理论基础严谨（满足完备性和敏感性公理）
- Baseline 选择：全零向量或随机 shuffled 序列
- 工具：`captum` 库的 `IntegratedGradients` 类

### 6.3 Grad-CAM（1D 版本）
- 利用最后卷积层的梯度生成热力图
- 适合定位启动子中哪些区段对预测贡献最大
- 分辨率受限于池化层步长

### 6.4 Attention 可视化
- 直接提取模型已有的注意力权重
- 映射回输入序列位置
- 对比高/低表达基因的注意力模式差异
- 计算成本为零

### 6.5 方法比较与推荐

| 方法 | 分辨率 | 难度 | 推荐优先级 |
|------|--------|------|-----------|
| Attention 可视化 | 中（受池化限制） | 低 | **首选** — 现成可用 |
| DeepLIFT | 碱基级 | 中 | **次选** — 精细归因 |
| Integrated Gradients | 碱基级 | 中 | **并行** — 交叉验证 |
| Grad-CAM | 区域级 | 中 | **补充** — 宏观定位 |

---

## 七、与已知生物学知识比对

### 比对流程
1. 用 DeepLIFT/IG 计算位置重要性分数，提取 Top-k 关键区域
2. 用 JASPAR PWM（FIMO 工具）扫描关键区域
3. 统计高/低表达基因关键区域中显著富集的 motif
4. 与 ENCODE ChIP-seq/DNase-seq 峰值交叉验证
5. 分析关键区域与 TSS 的距离分布

### JASPAR 数据库
- 最大开放获取 TF 结合谱数据库
- 网址：https://jaspar.elixir.no/
- 提供 PWM/PCM/PPM 格式结合谱

---

## 八、Motif 分析

### 方法 A — First Layer Filter 可视化
- CNN 第一层卷积核可学习短序列模式
- 提取高激活序列，用 WebLogo/logomaker 生成序列 logo
- 与 JASPAR motif 比较（Tomtom 工具）

### 方法 B — TF-MoDISco（推荐）
- Kundaje Lab 专门为基因组学开发的 motif 发现算法
- 输入碱基级重要性分数
- 流程：切分 seqlets → 聚类 → 生成 consensus motif (PWM)
- 可发现已知和全新 motif
- 工具：[tfmodisco](https://github.com/kundajelab/tfmodisco)

### 方法 C — In Silico Mutagenesis
- 逐位置单碱基突变，观察预测分数变化
- 精确但计算成本高（20000 × 3 = 60000 次前向传播）
- 适合小规模验证

### 推荐流程
```
碱基级归因 (DeepLIFT/IG)
       ↓
  TF-MoDISco 聚类
       ↓
  生成 consensus motif (PWM)
       ↓
  Tomtom 与 JASPAR 比对 → 已知 motif 匹配
       ↓
  未匹配的 novel motif → 候选新调控元件
```

### 工具汇总

| 工具 | 用途 | 语言 |
|------|------|------|
| Captum | PyTorch 可解释性（DeepLIFT, IG, GradCAM） | Python |
| TF-MoDISco | 从归因分数发现 motif | Python |
| logomaker | 生成序列 logo | Python |
| FIMO (MEME Suite) | 用 PWM 扫描序列 | 命令行 |
| Tomtom (MEME Suite) | motif 相似性比较 | 命令行 |
| JASPAR API | 获取已知 TF 结合谱 | Python |
| pyBigWig | 读取基因组信号文件 | Python |
| bedtools | 基因组区间操作 | 命令行 |

---

**参考来源：**
- [ENCODE Portal](https://www.encodeproject.org)
- [JASPAR 2026](https://jaspar.elixir.no/)
- [TF-MoDISco](https://github.com/kundajelab/tfmodisco)
- [Enformer - Nature Methods](https://www.nature.com/articles/s41592-021-01252-x)
- [基因组学可解释AI综述](https://academic.oup.com/bib/article/25/5/bbae449/7759907)
- [组蛋白修饰与基因表达](https://pmc.ncbi.nlm.nih.gov/articles/PMC3842134/)
- [ATAC-seq 综述](https://pmc.ncbi.nlm.nih.gov/articles/PMC9189070/)
