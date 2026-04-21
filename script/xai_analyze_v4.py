"""v4 XAI 可解释性分析：DeepLIFT 归因 + 注意力分析 + epigenomic 分支贡献。

新增 v4 专属分析：
- ENCODE 三通道信号贡献对比（H3K4me3 vs H3K27ac vs DNase）
- 表观信号区域重要性
- 分支消融归因（逐通道遮蔽）
"""

import argparse
import csv
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.modelv4_xai import GeneExpressTransformerV4XAI
from utils.xai_attribution import (
    compute_deeplift_batched,
    extract_conv_filters,
    identify_key_regions,
)
from utils.xai_visualization import (
    fig1_global_attribution,
    fig2_tss_zoom,
    fig3_attention_heads,
    fig4_sequence_logo_attribution,
    fig5_key_regions,
    fig6_group_comparison,
)


class C:
    GREEN = "\033[32m"; YELLOW = "\033[33m"; RED = "\033[31m"
    CYAN = "\033[36m"; BLUE = "\033[34m"; GRAY = "\033[90m"
    BOLD = "\033[1m"; RST = "\033[0m"

    @staticmethod
    def ok(msg): return f"{C.GREEN}{msg}{C.RST}"
    @staticmethod
    def warn(msg): return f"{C.YELLOW}{msg}{C.RST}"
    @staticmethod
    def err(msg): return f"{C.RED}{msg}{C.RST}"
    @staticmethod
    def info(msg): return f"{C.CYAN}{msg}{C.RST}"
    @staticmethod
    def hi(msg): return f"{C.BLUE}{msg}{C.RST}"
    @staticmethod
    def dim(msg): return f"{C.GRAY}{msg}{C.RST}"
    @staticmethod
    def bold(msg): return f"{C.BOLD}{msg}{C.RST}"


def load_hdf5(file_path: str):
    with h5py.File(file_path, "r") as f:
        halflife = torch.tensor(np.array(f["halflife"]), dtype=torch.float32)
        promoter = torch.tensor(np.array(f["promoter"]), dtype=torch.float32)
        labels = torch.tensor(np.array(f["label"]), dtype=torch.long)
    return promoter, halflife, labels


def parse_args():
    parser = argparse.ArgumentParser(description="v4 XAI 分析")
    parser.add_argument("--features", choices=["encode", "all"], default="all")
    parser.add_argument("--batch-size", type=int, default=4)
    return parser.parse_args()


def analyze_encode_contribution(model, test_p, test_h, test_enc, test_sf, test_l, device, output_dir):
    """逐通道遮蔽 ENCODE 信号，测量预测概率变化。"""
    print(C.info("分析 ENCODE 三通道贡献..."))
    model.eval()
    high_mask = test_l.numpy() == 1
    n = min(200, int(high_mask.sum()))
    p = test_p[high_mask][:n].to(device)
    h = test_h[high_mask][:n].to(device)
    e = test_enc[high_mask][:n].to(device)
    sf = test_sf[high_mask][:n].to(device) if test_sf is not None else None
    names = ["H3K4me3", "H3K27ac", "DNase"]

    with torch.no_grad():
        full_prob = torch.softmax(model(p, h, e, sf)[0], dim=1)[:, 1].cpu().numpy()
        results = {}
        for ch in range(3):
            e_m = e.clone()
            e_m[:, ch, :] = 0.0
            prob_m = torch.softmax(model(p, h, e_m, sf)[0], dim=1)[:, 1].cpu().numpy()
            delta = full_prob - prob_m
            results[names[ch]] = {"mean": float(delta.mean()), "std": float(delta.std()), "max": float(delta.max())}

        e_zero = torch.zeros_like(e)
        prob_zero = torch.softmax(model(p, h, e_zero, sf)[0], dim=1)[:, 1].cpu().numpy()
        results["ALL"] = {"mean": float((full_prob - prob_zero).mean()), "std": float((full_prob - prob_zero).std()), "max": float((full_prob - prob_zero).max())}

    print(f"  完整模型高表达概率: {C.bold(f'{full_prob.mean():.4f}')}")
    for name, s in results.items():
        print(f"  遮蔽 {name}: delta {s['mean']:+.4f}")

    csv_path = output_dir / "encode_channel_contribution.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["channel", "mean_delta", "std_delta", "max_delta"])
        for name, s in results.items():
            writer.writerow([name, f"{s['mean']:.6f}", f"{s['std']:.6f}", f"{s['max']:.6f}"])
    print(C.ok(f"通道贡献 CSV: {csv_path}"))


def analyze_encode_profiles(test_enc, test_l, output_dir):
    """高/低表达基因的 ENCODE 信号分布差异。"""
    print(C.info("分析 ENCODE 信号分布（高 vs 低表达）..."))
    high_mask = test_l.numpy() == 1
    low_mask = test_l.numpy() == 0
    names = ["H3K4me3", "H3K27ac", "DNase"]
    enc_h = test_enc[high_mask].numpy()
    enc_l = test_enc[low_mask].numpy()

    print("  TSS 区域信号对比 (TSS +/- 500bp):")
    for ch, name in enumerate(names):
        h_tss = enc_h[:, ch, 95:105].mean()
        l_tss = enc_l[:, ch, 95:105].mean()
        ratio = h_tss / (l_tss + 1e-8)
        print(f"    {name}: 高表达 {C.hi(f'{h_tss:.3f}')}  低表达 {C.dim(f'{l_tss:.3f}')}  比值 {C.bold(f'{ratio:.2f}x')}")

    csv_path = output_dir / "encode_signal_profiles.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["channel", "group", "mean_signal", "tss_mean"])
        for ch, name in enumerate(names):
            writer.writerow([name, "high", f"{enc_h[:, ch].mean():.6f}", f"{enc_h[:, ch, 95:105].mean():.6f}"])
            writer.writerow([name, "low", f"{enc_l[:, ch].mean():.6f}", f"{enc_l[:, ch, 95:105].mean():.6f}"])
    print(C.ok(f"信号分布 CSV: {csv_path}"))


def main():
    args = parse_args()
    feat_mode = args.features
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data"
    output_dir = project_root / "results" / "xai_v4"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(C.info(f"设备: {device}  特征: {C.bold(feat_mode)}"))

    use_encode = feat_mode in ("encode", "all")
    use_seq = feat_mode == "all"
    encode_channels = 3 if use_encode else 0
    seq_dim = 588 if use_seq else 0

    # 1. 加载模型
    print(C.dim("加载 v4 模型..."))
    model = GeneExpressTransformerV4XAI(encode_channels=encode_channels, seq_feat_dim=seq_dim)
    weight_path = data_dir / f"modelv4_{feat_mode}_best.pt"
    if not weight_path.exists():
        print(C.err(f"权重不存在: {weight_path}"))
        sys.exit(1)
    state_dict = torch.load(weight_path, map_location="cpu", weights_only=True)
    model.load_from_base(state_dict)
    model.to(device).eval()
    print(C.ok(f"模型加载完成  参数: {C.bold(f'{sum(p.numel() for p in model.parameters()):,}')}"))

    # 2. 加载数据
    print(C.dim("加载测试集..."))
    test_p, test_h, test_l = load_hdf5(str(data_dir / "test.h5"))
    high_mask = test_l.numpy() == 1
    low_mask = test_l.numpy() == 0
    n_high, n_low = int(high_mask.sum()), int(low_mask.sum())
    print(C.ok(f"测试集 {C.bold(len(test_l))} 样本  高 {n_high}  低 {n_low}"))

    test_enc = torch.tensor(torch.load(data_dir / "epigenomic.pt", weights_only=False)["test"], dtype=torch.float32) if use_encode else None
    test_sf = torch.load(data_dir / "seq_features_test.pt", weights_only=True) if use_seq else None

    # 3. ENCODE 分支分析
    if test_enc is not None:
        analyze_encode_contribution(model, test_p, test_h, test_enc, test_sf, test_l, device, output_dir)
        analyze_encode_profiles(test_enc, test_l, output_dir)

    # 4. 注意力分析
    print(C.info("计算注意力权重..."))
    attn_h, attn_l = [], []
    with torch.no_grad():
        for idx in np.where(high_mask)[0][:50]:
            e = test_enc[idx:idx+1].to(device) if test_enc is not None else None
            sf = test_sf[idx:idx+1].to(device) if test_sf is not None else None
            _, a = model(test_p[idx:idx+1].to(device), test_h[idx:idx+1].to(device), e, sf)
            attn_h.append(a[0].cpu().numpy())
        for idx in np.where(low_mask)[0][:50]:
            e = test_enc[idx:idx+1].to(device) if test_enc is not None else None
            sf = test_sf[idx:idx+1].to(device) if test_sf is not None else None
            _, a = model(test_p[idx:idx+1].to(device), test_h[idx:idx+1].to(device), e, sf)
            attn_l.append(a[0].cpu().numpy())
    attn_high = np.concatenate(attn_h)
    attn_low = np.concatenate(attn_l)

    tss_tokens = slice(28, 36)
    best_hi = int(np.argmax(attn_high[:, :, tss_tokens, tss_tokens].sum(axis=(1, 2, 3))))
    fig3_attention_heads(attn_high[best_hi], "v4_high", best_hi, output_dir)
    best_lo = int(np.argmax(attn_low[:, :, tss_tokens, tss_tokens].sum(axis=(1, 2, 3))))
    fig3_attention_heads(attn_low[best_lo], "v4_low", best_lo, output_dir)
    print(C.ok("注意力热力图"))

    # 5. DeepLIFT
    print(C.info(f"DeepLIFT 归因 (batch={args.batch_size})..."))
    mean_hl = test_h.mean(0, keepdim=True).to(device)
    mean_enc = test_enc.mean(0, keepdim=True).to(device) if test_enc is not None else None
    mean_sf = test_sf.mean(0, keepdim=True).to(device) if test_sf is not None else None
    model.set_fixed_inputs(mean_hl, mean_enc, mean_sf)

    from utils.xai_attribution import PromoterOnlyWrapper
    from captum.attr import DeepLift

    wrapper = PromoterOnlyWrapper(model, mean_hl)
    wrapper.to(device).eval()
    dl = DeepLift(wrapper)

    attr_high_parts = []
    for s in range(0, n_high, args.batch_size):
        bp = test_p[high_mask][s:s+args.batch_size].to(device)
        a = dl.attribute(bp, baselines=torch.zeros_like(bp), target=1)
        attr_high_parts.append((a * bp).sum(-1).detach().cpu().numpy())
    attr_high_np = np.concatenate(attr_high_parts)

    attr_low_parts = []
    for s in range(0, n_low, args.batch_size):
        bp = test_p[low_mask][s:s+args.batch_size].to(device)
        a = dl.attribute(bp, baselines=torch.zeros_like(bp), target=1)
        attr_low_parts.append((a * bp).sum(-1).detach().cpu().numpy())
    attr_low_np = np.concatenate(attr_low_parts)
    print(C.ok(f"归因完成  高 {attr_high_np.shape}  低 {attr_low_np.shape}"))

    # IG 交叉验证
    corr = None
    try:
        from captum.attr import IntegratedGradients
        ig = IntegratedGradients(wrapper)
        ig_idx = np.random.choice(n_high, min(20, n_high), replace=False)
        ig_parts = []
        for i in ig_idx:
            bp = test_p[high_mask][i:i+1].to(device)
            a = ig.attribute(bp, baselines=torch.zeros_like(bp), target=1, n_steps=30)
            ig_parts.append((a * bp).sum(-1).cpu().numpy())
        ig_np = np.concatenate(ig_parts)
        corr = float(np.corrcoef(attr_high_np[ig_idx].mean(0), ig_np.mean(0))[0, 1])
        print(f"  DeepLIFT vs IG 相关性: {C.bold(f'{corr:.4f}')}")
    except RuntimeError as e:
        if "memory" in str(e).lower():
            print(C.warn("IG OOM，跳过"))
            torch.cuda.empty_cache()
        else:
            raise

    # 6. 图表
    print(C.info("生成图表..."))
    fig1_global_attribution(attr_high_np, attr_low_np, output_dir)
    fig2_tss_zoom(attr_high_np, attr_low_np, test_p[high_mask].numpy(), output_dir)

    top_idx = int(np.argmax(attr_high_np.sum(1)))
    top_attr = attr_high_np[top_idx]
    top_prom = test_p[high_mask][top_idx].numpy()
    smooth = np.convolve(np.abs(top_attr), np.ones(200) / 200, mode="valid")
    ps = int(np.argmax(smooth))
    fig4_sequence_logo_attribution(top_attr, top_prom, output_dir, region_start=ps, region_end=min(ps+200, 20000), name_suffix="_v4_top")
    fig4_sequence_logo_attribution(top_attr, top_prom, output_dir, region_start=9900, region_end=10100, name_suffix="_v4_tss")

    all_regions = []
    for i in range(attr_high_np.shape[0]):
        all_regions.extend(identify_key_regions(attr_high_np[i], percentile=95, min_length=6))
    print(C.ok(f"关键区域: {C.bold(str(len(all_regions)))}"))
    fig5_key_regions(all_regions, output_dir)
    fig6_group_comparison(attr_high_np, attr_low_np, all_regions, output_dir)

    # 7. 保存
    filters = extract_conv_filters(model, top_k=5)
    np.savez(output_dir / "conv_filters_v4.npz", **filters)

    with open(output_dir / "key_regions_v4.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["start", "end", "length", "mean_score", "max_score"])
        w.writeheader()
        for r in sorted(all_regions, key=lambda x: x["mean_score"], reverse=True)[:100]:
            w.writerow(r)

    with open(output_dir / "analysis_summary_v4.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerow(["model", f"v4_{feat_mode}"])
        w.writerow(["n_high", n_high])
        w.writerow(["n_low", n_low])
        w.writerow(["mean_attr_high", f"{attr_high_np.mean():.6f}"])
        w.writerow(["mean_attr_low", f"{attr_low_np.mean():.6f}"])
        tss = slice(9500, 10500)
        w.writerow(["tss_attr_high", f"{attr_high_np[:, tss].mean():.6f}"])
        w.writerow(["tss_attr_low", f"{attr_low_np[:, tss].mean():.6f}"])
        w.writerow(["n_key_regions", len(all_regions)])
        if all_regions:
            tr = max(all_regions, key=lambda x: x["mean_score"])
            w.writerow(["top_region", f"{tr['start']}-{tr['end']}"])
        if corr is not None:
            w.writerow(["dl_ig_corr", f"{corr:.4f}"])

    print()
    print(C.ok("v4 XAI 分析完成") + f"  结果: {C.hi(str(output_dir))}")


if __name__ == "__main__":
    main()
