"""预计算序列内在特征并保存到磁盘，避免训练时 OOM。

分批计算 GC + CpG O/E + CpG 岛密度，保存为 data/seq_features.pt。
"""

import sys
from pathlib import Path

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def gc_content_cumsum(one_hot: torch.Tensor, window: int = 500, step: int = 100) -> torch.Tensor:
    """用 cumsum 快速计算滑动窗口 GC 含量。"""
    gc = (one_hot[:, :, 1] + one_hot[:, :, 2])  # (B, L)
    cs = gc.cumsum(dim=1)
    B, L = gc.shape
    n_w = (L - window) // step + 1
    out = torch.zeros(B, n_w)
    for i in range(n_w):
        s = i * step
        e = s + window
        val = cs[:, e - 1] - (cs[:, s - 1] if s > 0 else 0)
        out[:, i] = val / window
    return out


def cpg_oe_cumsum(one_hot: torch.Tensor, window: int = 500, step: int = 100) -> torch.Tensor:
    """用 cumsum 快速计算滑动窗口 CpG O/E。"""
    is_c = one_hot[:, :, 1]
    is_g = one_hot[:, :, 2]
    cpg = is_c[:, :-1] * is_g[:, 1:]  # (B, L-1)

    cpg_cs = cpg.cumsum(dim=1)
    c_cs = is_c.cumsum(dim=1)
    g_cs = is_g.cumsum(dim=1)

    B, L = one_hot.shape[0], one_hot.shape[1]
    n_w = (L - window) // step + 1
    out = torch.zeros(B, n_w)
    for i in range(n_w):
        s = i * step
        e = s + window
        n_cpg = cpg_cs[:, min(e - 2, cpg_cs.size(1) - 1)] - (cpg_cs[:, s - 1] if s > 0 else 0)
        n_c = c_cs[:, e - 1] - (c_cs[:, s - 1] if s > 0 else 0)
        n_g = g_cs[:, e - 1] - (g_cs[:, s - 1] if s > 0 else 0)
        out[:, i] = (n_cpg * window) / (n_c * n_g + 1e-8)
    return out


def cpg_island_score(one_hot: torch.Tensor, window: int = 500, step: int = 100) -> torch.Tensor:
    """CpG 岛密度分数（窗口满足 GC>50% 且 O/E>0.6 的比例）。"""
    gc = gc_content_cumsum(one_hot, window, step)
    cpg = cpg_oe_cumsum(one_hot, window, step)
    return ((gc > 0.5) & (cpg > 0.6)).float()


def load_hdf5(file_path: str):
    with h5py.File(file_path, "r") as f:
        halflife = torch.tensor(np.array(f["halflife"]), dtype=torch.float32)
        promoter = torch.tensor(np.array(f["promoter"]), dtype=torch.float32)
        labels = torch.tensor(np.array(f["label"]), dtype=torch.long)
    return promoter, halflife, labels


def compute_for_split(promoters: torch.Tensor, batch_size: int = 512) -> torch.Tensor:
    """分批计算序列特征，控制内存。"""
    B = promoters.shape[0]
    all_feats = []
    for start in range(0, B, batch_size):
        end = min(start + batch_size, B)
        chunk = promoters[start:end]
        gc = gc_content_cumsum(chunk)
        cpg = cpg_oe_cumsum(chunk)
        island = cpg_island_score(chunk)
        all_feats.append(torch.cat([gc, cpg, island], dim=1))
        print(f"  {end}/{B}", flush=True)
    return torch.cat(all_feats, dim=0)


def main():
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data"

    for split in ["train", "valid", "test"]:
        print(f"处理 {split}...", flush=True)
        p, _, _ = load_hdf5(str(data_dir / f"{split}.h5"))
        feats = compute_for_split(p)
        torch.save(feats, data_dir / f"seq_features_{split}.pt")
        print(f"  保存 seq_features_{split}.pt  形状: {feats.shape}", flush=True)

    print("全部完成", flush=True)


if __name__ == "__main__":
    main()
