"""ENCODE GM12878 表观基因组数据下载和预处理。

下载 GM12878 (hg19) 的 bigWig 信号文件：
- H3K4me3: 活跃启动子标记
- H3K27ac: 活跃启动子/增强子
- DNase-seq: 染色质开放性
"""

import os
import subprocess
from pathlib import Path

# GM12878 hg19 数据集（UCSC Broad Histone + ENCODE DNase-seq）
ENCODE_DATASETS = {
    "H3K4me3": {
        "url": "http://hgdownload.cse.ucsc.edu/goldenpath/hg19/encodeDCC/wgEncodeRegMarkH3k4me3/wgEncodeBroadHistoneGm12878H3k4me3StdSig.bigWig",
    },
    "H3K27ac": {
        "url": "http://hgdownload.cse.ucsc.edu/goldenpath/hg19/encodeDCC/wgEncodeRegMarkH3k27ac/wgEncodeBroadHistoneGm12878H3k27acStdSig.bigWig",
    },
    "DNase": {
        "url": "https://www.encodeproject.org/files/ENCFF001CUH/@@download/ENCFF001CUH.bigWig",
    },
}


def download_file(url: str, dest: Path, force: bool = False) -> Path:
    """下载文件，支持断点续传。"""
    if dest.exists() and not force:
        print(f"  已存在: {dest.name}")
        return dest

    print(f"  下载: {dest.name}")
    cmd = ["wget", "-c", "-q", "--show-progress", "-O", str(dest), url]
    subprocess.run(cmd, check=True)
    return dest


def download_encode_data(data_dir: Path, force: bool = False) -> dict[str, Path]:
    """下载全部 ENCODE bigWig 文件。

    Args:
        data_dir: 数据目录
        force: 是否强制重新下载

    Returns:
        dict: {name: bigWig_path}
    """
    encode_dir = data_dir / "encode"
    encode_dir.mkdir(exist_ok=True)

    bigwig_files = {}
    for name, info in ENCODE_DATASETS.items():
        dest = encode_dir / f"{name}.bigWig"
        download_file(info["url"], dest, force=force)
        bigwig_files[name] = dest

    print(f"  ENCODE 数据下载完成: {encode_dir}")
    return bigwig_files


if __name__ == "__main__":
    from pathlib import Path

    data_dir = Path(__file__).resolve().parent.parent / "data"
    download_encode_data(data_dir)
