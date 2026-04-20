"""XAI 可解释性分析主入口。

加载最佳模型权重 -> 转换为 XAI 模型 -> 注意力分析 + DeepLIFT 归因 ->
关键区域识别 -> 生成全部图表 + CSV。
"""

import argparse
import csv
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.modelv3 import GeneExpressTransformer
from model.modelv3_xai import GeneExpressTransformerXAI
from utils.xai_attribution import (
    compute_deeplift_batched,
    compute_integrated_gradients,
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
    """语义化颜色输出。"""
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RED = "\033[31m"
    CYAN = "\033[36m"
    BLUE = "\033[34m"
    GRAY = "\033[90m"
    BOLD = "\033[1m"
    RST = "\033[0m"

    @staticmethod
    def ok(msg):
        return f"{C.GREEN}{msg}{C.RST}"

    @staticmethod
    def warn(msg):
        return f"{C.YELLOW}{msg}{C.RST}"

    @staticmethod
    def err(msg):
        return f"{C.RED}{msg}{C.RST}"

    @staticmethod
    def info(msg):
        return f"{C.CYAN}{msg}{C.RST}"

    @staticmethod
    def hi(msg):
        return f"{C.BLUE}{msg}{C.RST}"

    @staticmethod
    def dim(msg):
        return f"{C.GRAY}{msg}{C.RST}"

    @staticmethod
    def bold(msg):
        return f"{C.BOLD}{msg}{C.RST}"


def load_hdf5(file_path: str):
    with h5py.File(file_path, "r") as f:
        halflife = torch.tensor(np.array(f["halflife"]), dtype=torch.float32)
        promoter = torch.tensor(np.array(f["promoter"]), dtype=torch.float32)
        labels = torch.tensor(np.array(f["label"]), dtype=torch.long)
    return promoter, halflife, labels


def parse_args():
    parser = argparse.ArgumentParser(description="XAI 可解释性分析")
    parser.add_argument("--batch-size", type=int, default=8, help="DeepLIFT batch size")
    parser.add_argument("--ig-samples", type=int, default=20, help="IG 交叉验证样本数")
    parser.add_argument("--skip-deeplift", action="store_true", help="跳过 DeepLIFT")
    return parser.parse_args()


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data"
    output_dir = project_root / "results" / "xai"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(C.info(f"设备: {device}"))

    # ---- 1. 加载模型 ----
    print(C.dim("加载模型权重..."))
    base_model = GeneExpressTransformer(num_classes=2)
    weight_path = data_dir / "modelv3_best.pt"
    if not weight_path.exists():
        print(C.err(f"权重文件不存在: {weight_path}"))
        print(C.warn("请先运行 python script/train_v3.py 训练模型"))
        sys.exit(1)

    base_model.load_state_dict(torch.load(weight_path, map_location="cpu", weights_only=True))
    base_model.to(device)

    xai_model = GeneExpressTransformerXAI(num_classes=2)
    state_dict = base_model.state_dict()
    xai_model.load_from_base(state_dict)
    xai_model.to(device)
    xai_model.eval()
    print(C.ok("XAI 模型加载完成"))

    # ---- 2. 加载测试集数据 ----
    print(C.dim("加载测试集数据..."))
    test_p, test_h, test_l = load_hdf5(str(data_dir / "test.h5"))
    print(C.ok(f"测试集 {C.bold(len(test_l))} 样本"))

    high_mask = test_l.numpy() == 1
    low_mask = test_l.numpy() == 0
    n_high = int(high_mask.sum())
    n_low = int(low_mask.sum())
    print(f"  高表达: {C.hi(n_high)}  低表达: {C.hi(n_low)}")

    # ---- 3. 注意力权重分析 ----
    print(C.info("计算注意力权重..."))
    all_attn_high = []
    all_attn_low = []
    with torch.no_grad():
        high_indices = np.where(high_mask)[0][:50]
        low_indices = np.where(low_mask)[0][:50]

        for idx in high_indices:
            p = test_p[idx:idx+1].to(device)
            h = test_h[idx:idx+1].to(device)
            _, attn = xai_model(p, h)
            all_attn_high.append(attn[0].cpu().numpy())

        for idx in low_indices:
            p = test_p[idx:idx+1].to(device)
            h = test_h[idx:idx+1].to(device)
            _, attn = xai_model(p, h)
            all_attn_low.append(attn[0].cpu().numpy())

    attn_high = np.concatenate(all_attn_high, axis=0)
    attn_low = np.concatenate(all_attn_low, axis=0)
    print(C.ok(f"注意力矩阵: 高表达 {attn_high.shape}, 低表达 {attn_low.shape}"))

    # fig3: 单样本注意力热力图
    print(C.dim("生成注意力热力图..."))
    tss_tokens = slice(28, 36)
    high_tss_attn = attn_high[:, :, tss_tokens, tss_tokens].sum(axis=(1, 2, 3))
    best_high_idx = int(np.argmax(high_tss_attn))
    fig3_attention_heads(attn_high[best_high_idx], "high", best_high_idx, output_dir)

    low_tss_attn = attn_low[:, :, tss_tokens, tss_tokens].sum(axis=(1, 2, 3))
    best_low_idx = int(np.argmax(low_tss_attn))
    fig3_attention_heads(attn_low[best_low_idx], "low", best_low_idx, output_dir)
    print(C.ok("fig3 注意力热力图已生成"))

    # ---- 4. DeepLIFT 归因 ----
    attr_high_np = None
    attr_low_np = None

    if not args.skip_deeplift:
        print(C.info(f"计算 DeepLIFT 归因 (batch_size={args.batch_size})..."))

        attr_high_np = compute_deeplift_batched(
            xai_model, test_p[high_mask], test_h[high_mask],
            target_class=1, batch_size=args.batch_size, device=device,
        )
        print(f"  高表达归因: {C.ok(str(attr_high_np.shape))}")

        attr_low_np = compute_deeplift_batched(
            xai_model, test_p[low_mask], test_h[low_mask],
            target_class=1, batch_size=args.batch_size, device=device,
        )
        print(f"  低表达归因: {C.ok(str(attr_low_np.shape))}")

        # IG 交叉验证（逐样本避免 OOM）
        print(C.dim(f"IG 交叉验证 ({args.ig_samples} 样本)..."))
        try:
            ig_idx = np.random.choice(n_high, min(args.ig_samples, n_high), replace=False)
            ig_parts = []
            for i in ig_idx:
                ig_single = compute_integrated_gradients(
                    xai_model,
                    test_p[high_mask][i:i+1],
                    test_h[high_mask][i:i+1],
                    target_class=1, n_steps=30, device=device,
                )
                ig_parts.append(ig_single)
            ig_high = np.concatenate(ig_parts, axis=0)
            dl_mean = attr_high_np[ig_idx].mean(axis=0)
            ig_mean = ig_high.mean(axis=0)
            corr = float(np.corrcoef(dl_mean, ig_mean)[0, 1])
            print(f"  DeepLIFT vs IG 相关性: {C.bold(f'{corr:.4f}')}")
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(C.warn("IG 交叉验证 GPU 内存不足，跳过"))
                torch.cuda.empty_cache()
            else:
                raise
    else:
        print(C.warn("跳过 DeepLIFT 归因"))

    # ---- 5. 生成全部图表 ----
    if attr_high_np is not None and attr_low_np is not None:
        print(C.info("生成可视化图表..."))

        print(C.dim("  fig1 全局归因分布..."))
        fig1_global_attribution(attr_high_np, attr_low_np, output_dir)

        print(C.dim("  fig2 TSS 放大视图..."))
        fig2_tss_zoom(
            attr_high_np, attr_low_np,
            test_p[high_mask].numpy(), output_dir,
        )

        # fig4: 序列 logo
        top_high_idx = int(np.argmax(attr_high_np.sum(axis=1)))
        top_attribution = attr_high_np[top_high_idx]
        top_promoter = test_p[high_mask][top_high_idx].numpy()

        smooth = np.convolve(np.abs(top_attribution), np.ones(200) / 200, mode="valid")
        peak_start = int(np.argmax(smooth))
        peak_end = min(peak_start + 200, 20000)
        print(C.dim(f"  fig4 序列 logo (区域 {peak_start}-{peak_end})..."))
        fig4_sequence_logo_attribution(
            top_attribution, top_promoter, output_dir,
            region_start=peak_start, region_end=peak_end,
            name_suffix="_top_high",
        )
        fig4_sequence_logo_attribution(
            top_attribution, top_promoter, output_dir,
            region_start=9900, region_end=10100,
            name_suffix="_tss",
        )

        # ---- 6. 关键区域识别 ----
        print(C.info("识别关键碱基片段..."))
        all_key_regions = []
        for i in range(attr_high_np.shape[0]):
            regions = identify_key_regions(attr_high_np[i], percentile=95, min_length=6)
            all_key_regions.extend(regions)
        print(C.ok(f"共识别 {C.bold(str(len(all_key_regions)))} 个关键区域"))

        print(C.dim("  fig5 关键区域..."))
        fig5_key_regions(all_key_regions, output_dir)

        print(C.dim("  fig6 分组对比..."))
        fig6_group_comparison(attr_high_np, attr_low_np, all_key_regions, output_dir)

        # ---- 7. 输出 CSV ----
        print(C.info("输出分析结果..."))

        csv_path = output_dir / "key_regions.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["start", "end", "length", "mean_score", "max_score"]
            )
            writer.writeheader()
            for r in sorted(all_key_regions, key=lambda x: x["mean_score"], reverse=True)[:100]:
                writer.writerow(r)
        print(C.ok(f"关键区域 CSV: {csv_path}"))

        summary_path = output_dir / "analysis_summary.csv"
        with open(summary_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value"])
            writer.writerow(["n_high", n_high])
            writer.writerow(["n_low", n_low])
            writer.writerow(["mean_attr_high", f"{attr_high_np.mean():.6f}"])
            writer.writerow(["mean_attr_low", f"{attr_low_np.mean():.6f}"])
            tss_region = slice(9500, 10500)
            writer.writerow(["tss_mean_attr_high", f"{attr_high_np[:, tss_region].mean():.6f}"])
            writer.writerow(["tss_mean_attr_low", f"{attr_low_np[:, tss_region].mean():.6f}"])
            writer.writerow(["n_key_regions", len(all_key_regions)])
            if all_key_regions:
                top_r = max(all_key_regions, key=lambda x: x["mean_score"])
                writer.writerow(["top_region", f"{top_r['start']}-{top_r['end']}"])
                writer.writerow(["top_region_mean_score", f"{top_r['mean_score']:.6f}"])
        print(C.ok(f"统计摘要: {summary_path}"))

    # ---- 8. CNN 卷积核 motif 提取 ----
    print(C.info("提取 CNN 第一层卷积核 motif..."))
    filters = extract_conv_filters(xai_model, top_k=5)
    for name, weights in filters.items():
        print(f"  {name}: {weights.shape}")

    np.savez(output_dir / "conv_filters.npz", **filters)
    print(C.ok("卷积核权重已保存"))

    # 完成
    print()
    print(C.ok("分析完成") + f"  结果保存在 {C.hi(str(output_dir))}")
    print(C.dim("生成的文件:"))
    for f in sorted(output_dir.iterdir()):
        size_kb = f.stat().st_size / 1024
        print(f"  {f.name} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
