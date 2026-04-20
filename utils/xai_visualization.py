"""XAI 可视化：6 张 Nature/Science 风格科研图表。

配色方案：高表达=#E63946，低表达=#457B9D，TSS=#F4A261
输出格式：PNG 300dpi + PDF 矢量
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
from scipy import stats
import seaborn as sns
import pandas as pd
from pathlib import Path

# 配色方案
COLOR_HIGH = "#E63946"
COLOR_LOW = "#457B9D"
COLOR_TSS = "#F4A261"
COLOR_NEUTRAL = "#2A9D8F"
COLOR_BG = "#FAFAFA"
COLOR_GRID = "#E0E0E0"

# matplotlib 全局字体
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica"],
    "font.size": 10,
    "axes.linewidth": 1.2,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})


def _save_fig(fig, output_dir: Path, name: str):
    """保存 PNG + PDF。"""
    fig.savefig(output_dir / f"{name}.png", dpi=300)
    fig.savefig(output_dir / f"{name}.pdf")
    plt.close(fig)


def fig1_global_attribution(
    attr_high: np.ndarray,
    attr_low: np.ndarray,
    output_dir: Path,
    bin_size: int = 100,
    fdr_alpha: float = 0.05,
):
    """fig1: 全局 20000bp 归因分布（高 vs 低表达）+ 显著性条带。

    双面板：上方为折线图，下方为 -log10(p-value) 显著性条带。

    Args:
        attr_high: (N_high, 20000) 高表达组归因
        attr_low: (N_low, 20000) 低表达组归因
        output_dir: 输出目录
        bin_size: 滑动窗口大小
        fdr_alpha: FDR 校正阈值
    """
    n_bins = 20000 // bin_size
    positions = np.arange(n_bins) * bin_size + bin_size // 2

    mean_high = attr_high.mean(axis=0)
    mean_low = attr_low.mean(axis=0)

    # 滑动窗口平滑
    smooth_high = np.convolve(mean_high, np.ones(bin_size) / bin_size, mode="valid")[:n_bins]
    smooth_low = np.convolve(mean_low, np.ones(bin_size) / bin_size, mode="valid")[:n_bins]

    # 逐 bin Mann-Whitney U 检验
    p_values = []
    for i in range(n_bins):
        start = i * bin_size
        end = start + bin_size
        h_vals = attr_high[:, start:end].mean(axis=1)
        l_vals = attr_low[:, start:end].mean(axis=1)
        try:
            _, p = stats.mannwhitneyu(h_vals, l_vals, alternative="two-sided")
        except ValueError:
            p = 1.0
        p_values.append(p)

    p_values = np.array(p_values)

    # FDR 校正
    try:
        from statsmodels.stats.multitest import multipletests
        _, p_corrected, _, _ = multipletests(p_values, alpha=fdr_alpha, method="fdr_bh")
    except ImportError:
        p_corrected = p_values

    neg_log_p = -np.log10(np.clip(p_corrected, 1e-300, 1.0))
    sig_threshold = -np.log10(fdr_alpha)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), height_ratios=[3, 1],
                                     sharex=True, facecolor=COLOR_BG)

    # 上方：归因曲线
    ax1.plot(positions, smooth_high, color=COLOR_HIGH, linewidth=1.5,
             label="High expression", alpha=0.9)
    ax1.plot(positions, smooth_low, color=COLOR_LOW, linewidth=1.5,
             label="Low expression", alpha=0.9)
    ax1.fill_between(positions, smooth_high, smooth_low,
                     where=smooth_high > smooth_low,
                     color=COLOR_HIGH, alpha=0.15, interpolate=True)
    ax1.fill_between(positions, smooth_low, smooth_high,
                     where=smooth_low > smooth_high,
                     color=COLOR_LOW, alpha=0.15, interpolate=True)

    ax1.set_ylabel("Mean attribution score")
    ax1.legend(loc="upper right", frameon=True, fancybox=True, shadow=False)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.grid(True, alpha=0.3, color=COLOR_GRID)

    # 标记 TSS 位置（假设在中间 10000bp）
    tss_pos = 10000
    ax1.axvline(x=tss_pos, color=COLOR_TSS, linestyle="--", linewidth=1.2, alpha=0.8, label="TSS")

    # 下方：显著性条带
    ax2.bar(positions, neg_log_p, width=bin_size * 0.8,
            color=np.where(neg_log_p > sig_threshold, COLOR_HIGH, COLOR_GRID),
            alpha=0.7)
    ax2.axhline(y=sig_threshold, color="black", linestyle=":", linewidth=1)
    ax2.set_ylabel(r"$-\log_{10}(p_{FDR})$")
    ax2.set_xlabel("Position in promoter region (bp)")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.suptitle("Global Attribution Profile: High vs Low Expression Genes",
                 fontsize=13, fontweight="bold", y=0.98)

    _save_fig(fig, output_dir, "fig1_global_attribution")


def fig2_tss_zoom(
    attr_high: np.ndarray,
    attr_low: np.ndarray,
    promoters_high: np.ndarray,
    output_dir: Path,
    tss_center: int = 10000,
    window: int = 1000,
    top_k: int = 50,
):
    """fig2: TSS +-1000bp 放大视图 + top-50 样本热力图。

    上方：折线图（高/低表达平均归因）
    下方：top-50 高表达样本热力图

    Args:
        attr_high: (N_high, 20000)
        attr_low: (N_low, 20000)
        promoters_high: (N_high, 20000, 4) 高表达组原始序列
        output_dir: 输出目录
        tss_center: TSS 位置
        window: 放大窗口半宽
        top_k: 热力图展示样本数
    """
    start = tss_center - window
    end = tss_center + window
    positions = np.arange(start, end)

    zoom_high = attr_high[:, start:end]
    zoom_low = attr_low[:, start:end]

    mean_h = zoom_high.mean(axis=0)
    mean_l = zoom_low.mean(axis=0)
    sem_h = zoom_high.std(axis=0) / np.sqrt(zoom_high.shape[0])
    sem_l = zoom_low.std(axis=0) / np.sqrt(zoom_low.shape[0])

    # 选 top-k 高表达样本（按 TSS 区域总归因排序）
    tss_scores = zoom_high.sum(axis=1)
    top_idx = np.argsort(tss_scores)[-top_k:][::-1]
    heatmap_data = zoom_high[top_idx]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), height_ratios=[2, 1.5],
                                     sharex=True, facecolor=COLOR_BG)

    # 上方折线
    ax1.plot(positions, mean_h, color=COLOR_HIGH, linewidth=1.5, label="High expression")
    ax1.fill_between(positions, mean_h - sem_h, mean_h + sem_h,
                     color=COLOR_HIGH, alpha=0.2)
    ax1.plot(positions, mean_l, color=COLOR_LOW, linewidth=1.5, label="Low expression")
    ax1.fill_between(positions, mean_l - sem_l, mean_l + sem_l,
                     color=COLOR_LOW, alpha=0.2)

    ax1.axvline(x=tss_center, color=COLOR_TSS, linestyle="--", linewidth=1.5, alpha=0.8)
    ax1.annotate("TSS", xy=(tss_center, ax1.get_ylim()[1] * 0.9),
                 fontsize=10, fontweight="bold", color=COLOR_TSS,
                 ha="center")
    ax1.set_ylabel("Attribution score")
    ax1.legend(loc="upper right", frameon=True)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.grid(True, alpha=0.3, color=COLOR_GRID)

    # 下方热力图
    sns.heatmap(heatmap_data, ax=ax2, cmap="Reds",
                xticklabels=100, yticklabels=False,
                cbar_kws={"label": "Attribution", "shrink": 0.8})
    n_ticks = 5
    tick_positions = np.linspace(0, len(positions) - 1, n_ticks, dtype=int)
    ax2.set_xticks(tick_positions + 0.5)
    ax2.set_xticklabels([str(positions[i]) for i in tick_positions])
    ax2.set_xlabel(f"Position (TSS ±{window}bp)")
    ax2.set_ylabel(f"Top-{top_k} samples")

    fig.suptitle("TSS Proximal Attribution Landscape",
                 fontsize=13, fontweight="bold", y=0.98)

    _save_fig(fig, output_dir, "fig2_tss_zoom")


def fig3_attention_heads(
    attn_weights: np.ndarray,
    label: str,
    sample_idx: int,
    output_dir: Path,
):
    """fig3: 单样本 4 head 注意力 + 平均注意力热力图。

    5 子图：head 0-3 + mean attention。

    Args:
        attn_weights: (4, 64, 64) 单样本注意力矩阵
        label: "high" 或 "low"
        sample_idx: 样本索引
        output_dir: 输出目录
    """
    fig, axes = plt.subplots(1, 5, figsize=(18, 3.5), facecolor=COLOR_BG)

    head_labels = [f"Head {i}" for i in range(4)] + ["Mean"]
    all_attn = [attn_weights[i] for i in range(4)]
    all_attn.append(attn_weights.mean(axis=0))

    vmax = max(a.max() for a in all_attn)

    for ax, data, title in zip(axes, all_attn, head_labels):
        sns.heatmap(data, ax=ax, cmap="viridis", vmin=0, vmax=vmax,
                    xticklabels=8, yticklabels=8,
                    cbar=(title == "Mean"),
                    cbar_kws={"shrink": 0.6} if title == "Mean" else None)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel("Token (key)")
        ax.set_ylabel("Token (query)" if title == "Head 0" else "")

    fig.suptitle(
        f"Attention Weights — {label.capitalize()} Expression Sample #{sample_idx}",
        fontsize=13, fontweight="bold", y=1.02
    )

    _save_fig(fig, output_dir, f"fig3_attention_{label}_sample{sample_idx}")


def fig4_sequence_logo_attribution(
    attribution: np.ndarray,
    promoter: np.ndarray,
    output_dir: Path,
    region_start: int = 0,
    region_end: int = 200,
    name_suffix: str = "",
):
    """fig4: 碱基级归因序列 logo 风格可视化。

    类似基因组浏览器 track，碱基高度与归因分数成正比。

    Args:
        attribution: (20000,) 单样本归因
        promoter: (20000, 4) one-hot 序列
        output_dir: 输出目录
        region_start: 放大区域起始
        region_end: 放大区域终止
        name_suffix: 文件名后缀
    """
    try:
        import logomaker
        has_logomaker = True
    except ImportError:
        has_logomaker = False

    bases = ["A", "C", "G", "T"]
    region_attr = attribution[region_start:region_end]
    region_seq = promoter[region_start:region_end]
    positions = np.arange(region_start, region_end)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 5), facecolor=COLOR_BG,
                                     height_ratios=[3, 1])

    # 上方：碱基级归因 bar
    base_colors = {"A": "#4CAF50", "C": "#2196F3", "G": "#FF9800", "T": "#9C27B0"}
    for i in range(len(region_attr)):
        base_idx = region_seq[i].argmax()
        base_name = bases[base_idx]
        color = base_colors[base_name]
        ax1.bar(positions[i], region_attr[i], width=1.0, color=color, alpha=0.8)

    ax1.set_ylabel("Attribution score")
    ax1.set_title(f"Base-level Attribution ({region_start}-{region_end}bp)", fontweight="bold")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=b, alpha=0.8) for b, c in base_colors.items()]
    ax1.legend(handles=legend_elements, loc="upper right", ncol=4, frameon=True)

    # 下方：logomaker sequence logo 或简单文本序列
    if has_logomaker:
        # 构建 PWM-like 矩阵
        df_data = np.abs(region_attr[:, None]) * region_seq
        df = pd.DataFrame(df_data, columns=bases)
        logomaker.Logo(df, ax=ax2, shade_below=0.5, fade_below=0.5)
        ax2.set_ylabel("Importance")
        ax2.set_xlabel("Position (bp)")
    else:
        # 简单文本序列展示
        seq_str = "".join([bases[int(s.argmax())] for s in region_seq])
        step = max(1, len(seq_str) // 60)
        display_pos = list(range(0, len(seq_str), step))
        ax2.scatter(display_pos, [0] * len(display_pos), s=1, alpha=0)
        for p in display_pos:
            ax2.text(p, 0, seq_str[p:p+step], fontsize=5, fontfamily="monospace",
                     ha="center", va="center")
        ax2.set_xlabel("Position (bp)")
        ax2.set_ylim(-0.5, 0.5)
        ax2.set_yticks([])

    _save_fig(fig, output_dir, f"fig4_sequence_logo{name_suffix}")


def fig5_key_regions(
    all_regions: list[dict],
    output_dir: Path,
    tss_pos: int = 10000,
    top_k: int = 10,
):
    """fig5: Top-10 关键区域柱状图 + 距 TSS 距离分布。

    Args:
        all_regions: 所有样本的关键区域列表
        output_dir: 输出目录
        tss_pos: TSS 位置
        top_k: 展示前 k 个区域
    """
    # 汇总所有区域，按平均分数排序
    df = pd.DataFrame(all_regions)
    top_regions = df.nlargest(top_k, "mean_score")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), facecolor=COLOR_BG)

    # 左：柱状图
    labels = [f"{r['start']}-{r['end']}" for _, r in top_regions.iterrows()]
    colors = [COLOR_HIGH if abs(r["start"] - tss_pos) < 2000
              else COLOR_LOW for _, r in top_regions.iterrows()]
    bars = ax1.barh(range(top_k), top_regions["mean_score"], color=colors, alpha=0.85)
    ax1.set_yticks(range(top_k))
    ax1.set_yticklabels(labels, fontsize=8)
    ax1.set_xlabel("Mean attribution score")
    ax1.set_title(f"Top-{top_k} Key Regions", fontweight="bold")
    ax1.invert_yaxis()
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # 标注距 TSS 距离
    for i, (_, r) in enumerate(top_regions.iterrows()):
        center = (r["start"] + r["end"]) / 2
        dist = center - tss_pos
        ax1.text(r["mean_score"] * 0.98, i, f"{dist:+.0f}bp",
                 va="center", ha="right", fontsize=7, color="white", fontweight="bold")

    # 右：距 TSS 距离分布
    distances = df["start"].apply(lambda s: s - tss_pos).values
    ax2.hist(distances, bins=50, color=COLOR_NEUTRAL, alpha=0.7, edgecolor="white")
    ax2.axvline(x=0, color=COLOR_TSS, linestyle="--", linewidth=1.5, label="TSS")
    ax2.set_xlabel("Distance from TSS (bp)")
    ax2.set_ylabel("Count")
    ax2.set_title("Distribution of Key Regions vs TSS", fontweight="bold")
    ax2.legend(frameon=True)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    _save_fig(fig, output_dir, "fig5_key_regions")


def fig6_group_comparison(
    attr_high: np.ndarray,
    attr_low: np.ndarray,
    key_regions: list[dict],
    output_dir: Path,
):
    """fig6: 高/低表达组关键区域归因分数箱线图对比。

    Args:
        attr_high: (N_high, 20000)
        attr_low: (N_low, 20000)
        key_regions: 关键区域列表
        output_dir: 输出目录
    """
    top_regions = sorted(key_regions, key=lambda r: r["mean_score"], reverse=True)[:8]

    data = []
    for region in top_regions:
        s, e = region["start"], region["end"]
        for val in attr_high[:, s:e].mean(axis=1):
            data.append({
                "region": f"{s}-{e}",
                "score": val,
                "group": "High expression",
            })
        for val in attr_low[:, s:e].mean(axis=1):
            data.append({
                "region": f"{s}-{e}",
                "score": val,
                "group": "Low expression",
            })

    df = pd.DataFrame(data)

    fig, ax = plt.subplots(figsize=(12, 6), facecolor=COLOR_BG)

    palette = {"High expression": COLOR_HIGH, "Low expression": COLOR_LOW}
    sns.boxplot(data=df, x="region", y="score", hue="group",
                palette=palette, ax=ax, width=0.6, fliersize=2)

    # 添加显著性标注
    for i, region in enumerate(top_regions):
        s, e = region["start"], region["end"]
        h_vals = attr_high[:, s:e].mean(axis=1)
        l_vals = attr_low[:, s:e].mean(axis=1)
        try:
            _, p = stats.mannwhitneyu(h_vals, l_vals, alternative="two-sided")
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        except ValueError:
            sig = "ns"
        y_max = max(h_vals.max(), l_vals.max())
        ax.text(i, y_max * 1.05, sig, ha="center", fontsize=8, fontweight="bold")

    ax.set_xlabel("Region (bp)")
    ax.set_ylabel("Mean attribution score")
    ax.set_title("Key Region Attribution: High vs Low Expression", fontweight="bold")
    ax.legend(loc="upper right", frameon=True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.xticks(rotation=30, ha="right")

    _save_fig(fig, output_dir, "fig6_group_comparison")
