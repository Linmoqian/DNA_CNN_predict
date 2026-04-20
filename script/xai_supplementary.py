"""补充图表生成：模型架构、DeepLIFT vs IG 相关性、卷积核 motif、TSS 富集。"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

COLOR_HIGH = "#E63946"
COLOR_LOW = "#457B9D"
COLOR_TSS = "#F4A261"
COLOR_NEUTRAL = "#2A9D8F"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica"],
    "font.size": 10,
    "axes.linewidth": 1.2,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

output_dir = Path(__file__).resolve().parent.parent / "results" / "xai"


def fig_architecture():
    """fig_architecture: GeneExpressTransformer 架构示意图。"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 7))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis("off")

    def draw_box(x, y, w, h, text, color, fontsize=8, alpha=0.85):
        rect = mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.1",
            facecolor=color, edgecolor="#333333", linewidth=1.2, alpha=alpha
        )
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color="white")

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                     arrowprops=dict(arrowstyle="->", color="#555", lw=1.5))

    ax.text(7, 7.6, "GeneExpressTransformer (v3) Architecture",
            ha="center", fontsize=14, fontweight="bold")

    # Input
    draw_box(0.2, 5.5, 2.2, 1.2, "Promoter\n(20000bp, 4ch)", "#4A90D9", 9)
    draw_box(0.2, 3.5, 2.2, 1.2, "Halflife\n(8-dim features)", "#7B68AE", 9)

    # Multi-scale CNN
    ax.text(3.8, 7.0, "Multi-scale CNN Frontend", ha="center", fontsize=10,
            fontweight="bold", color="#333")
    draw_box(2.8, 5.8, 1.8, 0.9, "Conv k=8\n(24ch)", "#E67E22", 8)
    draw_box(2.8, 4.6, 1.8, 0.9, "Conv k=16\n(24ch)", "#E74C3C", 8)
    draw_box(2.8, 3.4, 1.8, 0.9, "Conv k=32\n(24ch)", "#C0392B", 8)

    arrow(2.4, 6.1, 2.8, 6.25)
    arrow(2.4, 6.1, 2.8, 5.05)
    arrow(2.4, 6.1, 2.8, 3.85)

    # Concat + Fuse
    draw_box(5.0, 4.6, 1.6, 1.6, "Concat\n(72ch)\nFuse Conv\n(72->48)", "#27AE60", 8)
    arrow(4.6, 6.25, 5.0, 5.7)
    arrow(4.6, 5.05, 5.0, 5.4)
    arrow(4.6, 3.85, 5.0, 5.1)

    # Token compress + Transformer
    draw_box(7.0, 5.0, 2.0, 1.0, "Token Pool\n(->64 tokens)", "#8E44AD", 8)
    draw_box(7.0, 3.5, 2.0, 1.0, "Transformer\n(d=48, h=4, L=1)", "#2C3E50", 8)
    arrow(6.6, 5.4, 7.0, 5.5)
    arrow(8.0, 5.0, 8.0, 4.5)

    # Halflife branch
    draw_box(5.0, 1.8, 1.6, 1.2, "FC(8->48)\nFC(48->48)", "#7B68AE", 8)
    arrow(2.4, 4.1, 5.0, 2.4)

    # GAP + Concat
    draw_box(9.5, 4.3, 1.4, 1.2, "Global\nAvg Pool", "#16A085", 8)
    draw_box(9.5, 2.3, 1.4, 1.2, "Concat\n(96-dim)", "#2C3E50", 8)
    arrow(9.0, 4.0, 9.5, 4.9)
    arrow(9.0, 4.0, 9.5, 2.9)
    arrow(6.6, 2.4, 9.5, 2.9)

    # Classifier
    draw_box(11.3, 2.8, 1.8, 1.2, "FC(96->48)\nDropout(0.5)\nFC(48->2)", "#C0392B", 8)
    arrow(10.9, 2.9, 11.3, 3.4)

    # Output
    draw_box(13.0, 3.0, 0.8, 0.8, "High\n/ Low", "#333333", 8)
    arrow(13.1, 3.4, 13.0, 3.4)

    ax.text(7, 0.8, "Total parameters: 45,674  |  Test Accuracy: 0.8131  |  AUC: 0.8131",
            ha="center", fontsize=10, fontstyle="italic", color="#666")

    fig.savefig(output_dir / "fig_architecture.png", dpi=300)
    fig.savefig(output_dir / "fig_architecture.pdf")
    plt.close(fig)
    print("fig_architecture done")


def fig_dl_ig_correlation():
    """fig_dl_ig: DeepLIFT vs IG 归因相关性散点图。"""
    import torch
    import h5py
    from model.modelv3 import GeneExpressTransformer
    from model.modelv3_xai import GeneExpressTransformerXAI
    from utils.xai_attribution import compute_deeplift, compute_integrated_gradients

    project_root = Path(__file__).resolve().parent.parent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base = GeneExpressTransformer(num_classes=2)
    base.load_state_dict(torch.load(
        project_root / "data" / "modelv3_best.pt", map_location="cpu", weights_only=True))
    xai = GeneExpressTransformerXAI(num_classes=2)
    xai.load_from_base(base.state_dict())
    xai.to(device).eval()

    with h5py.File(project_root / "data" / "test.h5", "r") as f:
        halflife = torch.tensor(np.array(f["halflife"]), dtype=torch.float32)
        promoter = torch.tensor(np.array(f["promoter"]), dtype=torch.float32)
        labels = torch.tensor(np.array(f["label"]), dtype=torch.long)

    high_mask = labels.numpy() == 1
    np.random.seed(42)
    sample_idx = np.random.choice(high_mask.sum(), 10, replace=False)

    p_sample = promoter[high_mask][sample_idx]
    h_sample = halflife[high_mask][sample_idx]

    dl_all, ig_all = [], []
    for i in range(len(sample_idx)):
        dl_attr = compute_deeplift(xai, p_sample[i:i+1], h_sample[i:i+1],
                                    target_class=1, device=device)
        ig_attr = compute_integrated_gradients(xai, p_sample[i:i+1], h_sample[i:i+1],
                                                target_class=1, n_steps=30, device=device)
        dl_all.append(dl_attr[0])
        ig_all.append(ig_attr[0])

    dl_arr = np.concatenate(dl_all)
    ig_arr = np.concatenate(ig_all)

    np.random.seed(42)
    idx = np.random.choice(len(dl_arr), min(5000, len(dl_arr)), replace=False)
    dl_sample = dl_arr[idx]
    ig_sample = ig_arr[idx]
    corr = np.corrcoef(dl_sample, ig_sample)[0, 1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.scatter(dl_sample, ig_sample, alpha=0.3, s=5, color=COLOR_NEUTRAL)
    lim = max(abs(dl_sample).max(), abs(ig_sample).max()) * 0.95
    ax1.plot([-lim, lim], [-lim, lim], '--', color='gray', alpha=0.5, label='y=x')
    ax1.set_xlabel("DeepLIFT Attribution")
    ax1.set_ylabel("Integrated Gradients Attribution")
    ax1.set_title(f"Per-base Attribution Correlation (r = {corr:.4f})", fontweight="bold")
    ax1.legend(frameon=True)

    dl_mean = dl_arr.reshape(-1, 20000).mean(axis=0)
    ig_mean = ig_arr.reshape(-1, 20000).mean(axis=0)
    win = 100
    dl_smooth = np.convolve(dl_mean, np.ones(win)/win, mode="valid")
    ig_smooth = np.convolve(ig_mean, np.ones(win)/win, mode="valid")
    pos = np.arange(len(dl_smooth)) + win // 2

    ax2.plot(pos, dl_smooth, color=COLOR_HIGH, linewidth=1.5, label="DeepLIFT", alpha=0.9)
    ax2.plot(pos, ig_smooth, color=COLOR_LOW, linewidth=1.5, label="Integrated Gradients", alpha=0.9)
    ax2.axvline(x=10000, color=COLOR_TSS, linestyle="--", linewidth=1.2, alpha=0.8, label="TSS")
    ax2.set_xlabel("Position (bp)")
    ax2.set_ylabel("Mean Attribution")
    ax2.set_title("Method Agreement: Global Attribution Profile", fontweight="bold")
    ax2.legend(frameon=True)

    fig.suptitle("Cross-validation of Attribution Methods", fontsize=13,
                 fontweight="bold", y=1.02)
    fig.savefig(output_dir / "fig_dl_ig_correlation.png", dpi=300)
    fig.savefig(output_dir / "fig_dl_ig_correlation.pdf")
    plt.close(fig)
    print("fig_dl_ig_correlation done")


def fig_filter_motifs():
    """fig_filter: CNN 第一层卷积核 motif 可视化。"""
    from scipy.special import softmax

    data = np.load(output_dir / "conv_filters.npz")

    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    bases = ["A", "C", "G", "T"]
    base_colors = ["#4CAF50", "#2196F3", "#FF9800", "#9C27B0"]

    for ax, (name, weights) in zip(axes, data.items()):
        avg_w = weights.mean(axis=0).T  # (k, 4)
        ppm = softmax(avg_w, axis=1)

        x = np.arange(ppm.shape[0])
        bottom = np.zeros(ppm.shape[0])
        for b in range(4):
            ax.bar(x, ppm[:, b], bottom=bottom, color=base_colors[b],
                   label=bases[b], width=0.9, alpha=0.85, edgecolor="white", linewidth=0.3)
            bottom += ppm[:, b]

        kernel_size = weights.shape[2]
        ax.set_ylabel("Probability")
        ax.set_xlabel("Position")
        ax.set_title(f"Conv {name} (k={kernel_size}): Average Filter Motif (n={weights.shape[0]})",
                     fontweight="bold")
        ax.legend(loc="upper right", ncol=4, fontsize=8)
        ax.set_ylim(0, 1.05)

    fig.suptitle("Learned Sequence Motifs from First-layer CNN Filters",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.savefig(output_dir / "fig_filter_motifs.png", dpi=300)
    fig.savefig(output_dir / "fig_filter_motifs.pdf")
    plt.close(fig)
    print("fig_filter_motifs done")


def fig_tss_enrichment():
    """fig_enrichment: 关键区域富集分析。"""
    import pandas as pd
    from scipy import stats

    regions_df = pd.read_csv(output_dir / "key_regions.csv")
    tss_pos = 10000
    regions_df["center"] = (regions_df["start"] + regions_df["end"]) / 2
    regions_df["dist_to_tss"] = regions_df["center"] - tss_pos

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # (a) 位置分布
    ax = axes[0, 0]
    bins = np.linspace(0, 20000, 41)
    ax.hist(regions_df["center"], bins=bins, color=COLOR_NEUTRAL, alpha=0.75,
            edgecolor="white", linewidth=0.5)
    ax.axvline(x=tss_pos, color=COLOR_TSS, linestyle="--", linewidth=2, label="TSS")
    ax.set_xlabel("Position in promoter (bp)")
    ax.set_ylabel("Number of key regions")
    ax.set_title("(a) Key Region Distribution Along Promoter", fontweight="bold")
    ax.legend(frameon=True)

    # (b) 距 TSS 距离分布
    ax = axes[0, 1]
    ax.hist(regions_df["dist_to_tss"], bins=50, color=COLOR_HIGH, alpha=0.7, edgecolor="white")
    ax.axvline(x=0, color=COLOR_TSS, linestyle="--", linewidth=2, label="TSS")
    ax.set_xlabel("Distance from TSS (bp)")
    ax.set_ylabel("Count")
    ax.set_title("(b) Distance from TSS", fontweight="bold")
    ax.legend(frameon=True)

    # (c) 长度分布
    ax = axes[1, 0]
    ax.hist(regions_df["length"], bins=range(5, 22), color=COLOR_LOW, alpha=0.75,
            edgecolor="white")
    ax.set_xlabel("Region length (bp)")
    ax.set_ylabel("Count")
    median_len = regions_df["length"].median()
    ax.axvline(x=median_len, color="red", linestyle=":", linewidth=1.5,
               label=f"Median = {median_len:.0f}bp")
    ax.set_title("(c) Key Region Length Distribution", fontweight="bold")
    ax.legend(frameon=True)

    # (d) 近端 vs 远端
    ax = axes[1, 1]
    proximal_mask = regions_df["dist_to_tss"].abs() <= 500
    distal_mask = regions_df["dist_to_tss"].abs() > 500
    prox_scores = regions_df.loc[proximal_mask, "mean_score"]
    dist_scores = regions_df.loc[distal_mask, "mean_score"]

    bp = ax.boxplot([prox_scores, dist_scores],
                     labels=["Proximal\n(|d| <= 500bp)", "Distal\n(|d| > 500bp)"],
                     patch_artist=True, widths=0.5)
    bp["boxes"][0].set_facecolor(COLOR_HIGH)
    bp["boxes"][0].set_alpha(0.7)
    bp["boxes"][1].set_facecolor(COLOR_LOW)
    bp["boxes"][1].set_alpha(0.7)

    try:
        _, p_val = stats.mannwhitneyu(prox_scores, dist_scores, alternative="greater")
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
    except ValueError:
        p_val = 1.0
        sig = "ns"

    y_max = max(prox_scores.max(), dist_scores.max())
    ax.text(1.5, y_max * 0.95, f"p = {p_val:.2e} {sig}", ha="center", fontsize=9,
            fontweight="bold")
    ax.set_ylabel("Mean attribution score")
    ax.set_title("(d) Proximal vs Distal Attribution", fontweight="bold")

    fig.suptitle("Enrichment Analysis of Key Regulatory Regions",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.savefig(output_dir / "fig_tss_enrichment.png", dpi=300)
    fig.savefig(output_dir / "fig_tss_enrichment.pdf")
    plt.close(fig)
    print("fig_tss_enrichment done")


if __name__ == "__main__":
    print("Generating supplementary figures...")
    fig_architecture()
    fig_filter_motifs()
    fig_tss_enrichment()
    fig_dl_ig_correlation()
    print("All supplementary figures generated.")
