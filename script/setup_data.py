"""数据集下载与校验脚本。

下载 GM12878 基因表达数据集并校验完整性。

用法:
    python script/setup_data.py           # 下载 + 校验
    python script/setup_data.py --check   # 仅校验已有文件
"""

import hashlib
import sys
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# 数据集信息
DATA_URL = "http://www.aisccc.cn/database/data-details?id=121"
FILES = {
    "train.h5": None,  # 无固定 hash，网站可能更新
    "valid.h5": None,
    "test.h5": None,
}
MIN_SIZES = {
    "train.h5": 100_000_000,  # > 100MB
    "valid.h5": 5_000_000,   # > 5MB
    "test.h5": 5_000_000,    # > 5MB
}

# ANSI 颜色
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
CYAN = "\033[36m"
BLUE = "\033[34m"
GRAY = "\033[90m"
RESET = "\033[0m"


def check_files() -> bool:
    """校验数据文件是否存在且大小合理。"""
    print(f"\n{CYAN}校验数据文件{RESET}")
    print(f"{GRAY}路径: {DATA_DIR}{RESET}\n")

    all_ok = True
    for fname, min_size in MIN_SIZES.items():
        fpath = DATA_DIR / fname
        if not fpath.exists():
            print(f"  {RED}缺失{RESET} {fname}")
            all_ok = False
        else:
            size = fpath.stat().st_size
            if size < min_size:
                print(f"  {YELLOW}异常{RESET} {fname} ({size / 1e6:.1f}MB, 期望 >{min_size / 1e6:.0f}MB)")
                all_ok = False
            else:
                print(f"  {GREEN}正常{RESET} {fname} ({size / 1e6:.1f}MB)")

    return all_ok


def verify_h5(fname: str) -> bool:
    """校验 HDF5 文件基本结构。"""
    try:
        import h5py

        fpath = DATA_DIR / fname
        with h5py.File(fpath, "r") as f:
            keys = set(f.keys())
            expected = {"gene_id", "halflife", "promoter", "label"}
            if not expected.issubset(keys):
                missing = expected - keys
                print(f"  {YELLOW}缺少键: {missing}{RESET}")
                return False
            n = len(f["label"])
            print(f"  {GREEN}HDF5 结构正确{RESET} ({n} 样本, 键: {sorted(keys)})")
            return True
    except ImportError:
        print(f"  {YELLOW}跳过 HDF5 校验 (h5py 未安装){RESET}")
        return True
    except Exception as e:
        print(f"  {RED}HDF5 校验失败: {e}{RESET}")
        return False


def main():
    check_only = "--check" in sys.argv

    print(f"\n{BLUE}DNA 基因表达预测 - 数据集准备{RESET}")
    print(f"{GRAY}{'=' * 50}{RESET}")

    if check_files():
        print(f"\n{GREEN}文件校验通过，验证 HDF5 结构...{RESET}")
        all_valid = True
        for fname in MIN_SIZES:
            if not verify_h5(fname):
                all_valid = False

        if all_valid:
            print(f"\n{GREEN}所有数据文件就绪{RESET}")
            return 0
        else:
            print(f"\n{RED}部分文件损坏，请重新下载{RESET}")
            return 1

    if check_only:
        print(f"\n{RED}数据文件不完整，请手动下载{RESET}")
        print(f"\n{CYAN}下载地址:{RESET}")
        print(f"  {BLUE}{DATA_URL}{RESET}")
        print(f"\n{CYAN}将以下文件放入 {DATA_DIR}/ 目录:{RESET}")
        for fname in MIN_SIZES:
            print(f"  - {fname}")
        return 1

    # 引导用户下载
    print(f"\n{YELLOW}数据文件缺失，需手动下载{RESET}")
    print(f"\n{CYAN}步骤:{RESET}")
    print(f"  1. 访问 {BLUE}{DATA_URL}{RESET}")
    print(f"  2. 下载压缩包并解压")
    print(f"  3. 将以下文件放入 {DATA_DIR}/ 目录:")
    for fname in MIN_SIZES:
        print(f"     {fname}")

    print(f"\n{CYAN}完成后重新运行校验:{RESET}")
    print(f"  python script/setup_data.py --check")

    return 1


if __name__ == "__main__":
    sys.exit(main())
