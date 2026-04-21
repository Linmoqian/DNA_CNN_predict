"""从 ENCODE bigWig 提取表观基因组特征。

按基因 TSS ±10000bp 窗口提取 H3K4me3、H3K27ac、DNase-seq 信号，
分箱为 200 个 bin（每 bin 100bp），保存为 data/epigenomic.pt。
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.encode_downloader import download_encode_data


# GTF 文件 URL (GENCODE v19, hg19)
GTF_URL = (
    "https://ftp.ebi.ac.uk/pub/databases/gencode/"
    "Gencode_human/release_19/gencode.v19.annotation.gtf.gz"
)


def download_gtf(data_dir: Path) -> Path:
    """下载 GENCODE v19 GTF 注释文件。"""
    gtf_path = data_dir / "encode" / "gencode.v19.annotation.gtf.gz"
    if gtf_path.exists():
        return gtf_path
    gtf_path.parent.mkdir(exist_ok=True)
    print(f"下载 GTF: {GTF_URL}")
    import subprocess

    subprocess.run(
        ["wget", "-c", "-q", "--show-progress", "-O", str(gtf_path), GTF_URL],
        check=True,
    )
    return gtf_path


def parse_gtf_for_genes(gtf_path: Path) -> dict:
    """解析 GTF 文件，提取基因 TSS 坐标。

    Returns:
        dict: {ensembl_id: {"chr": str, "tss": int, "strand": str}}
    """
    import gzip

    gene_coords = {}

    opener = gzip.open if str(gtf_path).endswith(".gz") else open
    with opener(gtf_path, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split("\t")
            if len(parts) < 9:
                continue
            if parts[2] != "gene":
                continue

            chrom = parts[0]
            if not chrom.startswith("chr") or len(chrom) > 5:
                continue  # 跳过 scaffold 等

            start = int(parts[3])
            end = int(parts[4])
            strand = parts[6]

            # 从 attributes 提取 gene_id
            attrs = parts[8]
            gene_id = None
            for attr in attrs.split(";"):
                attr = attr.strip()
                if attr.startswith("gene_id"):
                    gene_id = attr.split('"')[1]
                    # 去掉版本号 (ENSG00000118257.3 -> ENSG00000118257)
                    gene_id = gene_id.split(".")[0]
                    break

            if gene_id and gene_id.startswith("ENSG"):
                tss = start if strand == "+" else end
                gene_coords[gene_id] = {
                    "chr": chrom,
                    "tss": tss,
                    "strand": strand,
                }

    print(f"  解析到 {len(gene_coords)} 个基因坐标")
    return gene_coords


def extract_signal(
    bigwig_path: Path,
    chrom: str,
    start: int,
    end: int,
    n_bins: int = 200,
) -> np.ndarray:
    """从 bigWig 提取区间信号并分箱。

    Args:
        bigwig_path: bigWig 文件路径
        chrom: 染色体 (如 "chr1")
        start: 起始位置
        end: 终止位置
        n_bins: 分箱数量

    Returns:
        signal: (n_bins,) 信号值数组
    """
    import pyBigWig

    bw = pyBigWig.open(str(bigwig_path))
    bin_size = (end - start) / n_bins

    signal = np.zeros(n_bins, dtype=np.float32)
    for i in range(n_bins):
        bin_start = int(start + i * bin_size)
        bin_end = int(start + (i + 1) * bin_size)
        try:
            val = bw.stats(chrom, bin_start, bin_end, type="mean")[0]
            signal[i] = val if val is not None else 0.0
        except Exception:
            signal[i] = 0.0

    bw.close()
    return signal


def main():
    parser = argparse.ArgumentParser(description="提取表观基因组特征")
    parser.add_argument(
        "--skip-download", action="store_true", help="跳过下载步骤"
    )
    parser.add_argument(
        "--window", type=int, default=10000, help="TSS 侧翼窗口大小 (bp)"
    )
    parser.add_argument(
        "--n-bins", type=int, default=200, help="信号分箱数"
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data"

    # Step 1: 下载 ENCODE 数据
    if not args.skip_download:
        print("下载 ENCODE bigWig 数据...")
        bigwig_files = download_encode_data(data_dir)
    else:
        encode_dir = data_dir / "encode"
        bigwig_files = {
            "H3K4me3": encode_dir / "H3K4me3.bigWig",
            "H3K27ac": encode_dir / "H3K27ac.bigWig",
            "DNase": encode_dir / "DNase.bigWig",
        }

    # Step 2: 下载 GTF
    print("下载 GENCODE v19 GTF...")
    gtf_path = download_gtf(data_dir)
    print(f"  GTF 路径: {gtf_path}")

    # Step 3: 解析基因坐标
    print("解析基因坐标...")
    gene_coords = parse_gtf_for_genes(gtf_path)

    # Step 4: 加载 HDF5 获取 gene_id
    print("加载 HDF5 数据...")
    all_genes = {}
    for split in ["train", "valid", "test"]:
        h5_path = data_dir / f"{split}.h5"
        if not h5_path.exists():
            print(f"  跳过 {h5_path}（不存在）")
            continue
        with h5py.File(str(h5_path), "r") as f:
            gene_ids = [gid.decode() if isinstance(gid, bytes) else gid for gid in f["gene_id"]]
        all_genes[split] = gene_ids
        print(f"  {split}: {len(gene_ids)} 基因")

    # Step 5: 提取信号
    print("提取表观信号...")
    window = args.window
    n_bins = args.n_bins
    mark_names = ["H3K4me3", "H3K27ac", "DNase"]

    result = {}
    for split, gene_ids in all_genes.items():
        n_genes = len(gene_ids)
        signals = np.zeros((n_genes, 3, n_bins), dtype=np.float32)
        matched = 0

        for i, gid in enumerate(gene_ids):
            if gid not in gene_coords:
                continue
            coord = gene_coords[gid]
            chrom = coord["chr"]
            tss = coord["tss"]
            start = max(0, tss - window)
            end = tss + window

            for j, mark in enumerate(mark_names):
                bw_path = bigwig_files.get(mark)
                if bw_path and bw_path.exists():
                    signals[i, j] = extract_signal(bw_path, chrom, start, end, n_bins)
            matched += 1

            if (i + 1) % 1000 == 0:
                print(f"  {split}: {i + 1}/{n_genes}")

        match_rate = matched / n_genes * 100 if n_genes > 0 else 0
        print(f"  {split}: {matched}/{n_genes} 匹配 ({match_rate:.1f}%)")
        result[split] = signals

    # Step 6: 标准化
    print("标准化信号...")
    # 用训练集统计做标准化
    if "train" in result:
        train_signals = result["train"]
        # 每个 mark 独立标准化
        for j in range(3):
            vals = train_signals[:, j, :]
            mean = vals.mean()
            std = vals.std() + 1e-8
            for split in result:
                result[split][:, j, :] = (result[split][:, j, :] - mean) / std

    # Step 7: 保存
    output_path = data_dir / "epigenomic.pt"
    torch.save(result, output_path)
    print(f"保存到 {output_path}")
    for split, signals in result.items():
        print(f"  {split}: {signals.shape}")


if __name__ == "__main__":
    main()
