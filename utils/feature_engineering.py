"""从 DNA one-hot 序列计算内在特征。

Phase A 特征（无需外部数据）：
- GC 含量（滑动窗口）
- CpG 岛检测
- k-mer 频率统计
"""

import torch
import numpy as np


def compute_gc_content(
    one_hot: torch.Tensor,
    window: int = 500,
    step: int = 100,
) -> torch.Tensor:
    """滑动窗口计算 GC 含量。

    Args:
        one_hot: (B, L, 4) one-hot 编码，通道顺序 A=0, C=1, G=2, T=3
        window: 滑动窗口大小 (bp)
        step: 滑动步长 (bp)

    Returns:
        gc: (B, N_windows) GC 含量向量
    """
    # G + C 比例
    gc_sum = one_hot[:, :, 1] + one_hot[:, :, 2]  # (B, L)
    B, L = gc_sum.shape
    n_windows = (L - window) // step + 1

    gc_vals = []
    for i in range(n_windows):
        start = i * step
        gc_vals.append(gc_sum[:, start : start + window].mean(dim=1))

    return torch.stack(gc_vals, dim=1)  # (B, n_windows)


def compute_cpg_ratio(
    one_hot: torch.Tensor,
    window: int = 500,
    step: int = 100,
) -> torch.Tensor:
    """滑动窗口计算 CpG 观测/期望比。

    CpG O/E = (count_CG * L) / (count_C * count_G)

    Args:
        one_hot: (B, L, 4) one-hot 编码
        window: 滑动窗口大小 (bp)
        step: 滑动步长 (bp)

    Returns:
        cpg_ratio: (B, N_windows) CpG O/E 比值
    """
    # 提取 C 和 G 位置
    is_c = one_hot[:, :, 1]  # (B, L)
    is_g = one_hot[:, :, 2]  # (B, L)

    # 检测 CpG 二核苷酸：位置 i 为 C 且位置 i+1 为 G
    cpg = is_c[:, :-1] * is_g[:, 1:]  # (B, L-1)

    B, L = is_c.shape
    n_windows = (L - window) // step + 1

    ratios = []
    for i in range(n_windows):
        start = i * step
        end = start + window
        count_cpg = cpg[:, start : end - 1].sum(dim=1)
        count_c = is_c[:, start:end].sum(dim=1)
        count_g = is_g[:, start:end].sum(dim=1)
        # O/E = (count_CG * L) / (count_C * count_G)
        denom = count_c * count_g + 1e-8
        ratio = (count_cpg * window) / denom
        ratios.append(ratio)

    return torch.stack(ratios, dim=1)  # (B, n_windows)


def detect_cpg_islands(
    one_hot: torch.Tensor,
    window: int = 500,
    step: int = 100,
) -> torch.Tensor:
    """简化版 CpG 岛密度：直接在滑动窗口内计算 GC>50% 且 CpG O/E>0.6 的比例。

    Args:
        one_hot: (B, L, 4) one-hot 编码
        window: 滑动窗口大小 (bp)
        step: 滑动步长 (bp)

    Returns:
        density: (B, N_windows) 每个窗口满足 CpG 岛条件的比例
    """
    B, L = one_hot.shape[0], one_hot.shape[1]
    n_windows = (L - window) // step + 1

    is_c = one_hot[:, :, 1]
    is_g = one_hot[:, :, 2]
    cpg = is_c[:, :-1] * is_g[:, 1:]  # (B, L-1)

    # 预计算累积和用于快速滑动窗口统计
    gc_sum = (is_c + is_g).cumsum(dim=1)  # (B, L)
    cpg_sum = cpg.cumsum(dim=1)  # (B, L-1)
    c_count = is_c.cumsum(dim=1)
    g_count = is_g.cumsum.cumsum(dim=1) if False else is_g.cumsum(dim=1)

    densities = []
    for i in range(n_windows):
        s = i * step
        e = s + window
        gc = (gc_sum[:, e - 1] - (gc_sum[:, s - 1] if s > 0 else 0)) / window
        n_cpg = cpg_sum[:, min(e - 2, cpg_sum.size(1) - 1)] - (cpg_sum[:, s - 1] if s > 0 else 0)
        n_c = c_count[:, e - 1] - (c_count[:, s - 1] if s > 0 else 0)
        n_g = g_count[:, e - 1] - (g_count[:, s - 1] if s > 0 else 0)
        cpg_oe = (n_cpg * window) / (n_c * n_g + 1e-8)
        # 窗口满足 CpG 岛条件的位置比例（这里直接返回连续浮点值）
        score = ((gc > 0.5) & (cpg_oe > 0.6)).float()
        densities.append(score)

    return torch.stack(densities, dim=1)  # (B, n_windows)


def compute_kmer_counts(
    one_hot: torch.Tensor,
    k: int = 4,
    window: int = 500,
    step: int = 100,
    top_k: int = 64,
) -> torch.Tensor:
    """滑动窗口统计 k-mer 频率并取 top_k 维。

    由于完整 4-mer = 256 维、6-mer = 4096 维，使用 top_k 截断。

    Args:
        one_hot: (B, L, 4) one-hot 编码
        k: k-mer 的 k 值
        window: 滑动窗口大小
        step: 滑动步长
        top_k: 保留最高频 k-mer 数量

    Returns:
        kmer_feat: (B, N_windows * top_k) 展平的 k-mer 特征
    """
    B, L, _ = one_hot.shape
    n_windows = (L - window) // step + 1

    # 将 one-hot 转为碱基索引序列
    base_idx = one_hot.argmax(dim=-1)  # (B, L)

    all_feats = []
    for wi in range(n_windows):
        start = wi * step
        end = start + window
        window_bases = base_idx[:, start:end]  # (B, window)

        # 编码 k-mer 为整数: sum(base_i * 4^i)
        kmers = torch.zeros(B, end - start - k + 1, dtype=torch.long, device=one_hot.device)
        for j in range(k):
            kmers = kmers * 4 + window_bases[:, j : end - start - k + 1 + j]

        # 统计每个 k-mer 出现次数
        n_kmers = 4 ** k
        counts = torch.zeros(B, min(n_kmers, top_k), device=one_hot.device)
        for b in range(B):
            unique, cnts = kmers[b].unique(return_counts=True)
            # 按频率排序取 top_k
            sorted_idx = cnts.argsort(descending=True)[:top_k]
            vals = unique[sorted_idx]
            freqs = cnts[sorted_idx].float() / (end - start - k + 1)
            for idx in range(min(len(vals), top_k)):
                counts[b, idx] = freqs[idx]

        all_feats.append(counts)

    # (B, n_windows, top_k) -> (B, n_windows * top_k)
    result = torch.cat(all_feats, dim=1)
    return result


def compute_sequence_features(
    one_hot: torch.Tensor,
    include_gc: bool = True,
    include_cpg_density: bool = True,
    include_cpg_island: bool = True,
    include_kmer: bool = False,
    batch_size: int = 256,
) -> torch.Tensor:
    """计算全部序列内在特征并拼接。分批处理避免 OOM。

    Args:
        one_hot: (B, L, 4) one-hot 编码
        include_gc: 是否包含 GC 含量
        include_cpg_density: 是否包含 CpG O/E 比
        include_cpg_island: 是否包含 CpG 岛密度
        include_kmer: 是否包含 k-mer 频率
        batch_size: 分批大小

    Returns:
        features: (B, N_total_feat) 拼接的特征向量
    """
    B = one_hot.shape[0]
    results = []

    for start in range(0, B, batch_size):
        end = min(start + batch_size, B)
        chunk = one_hot[start:end]
        parts = []

        if include_gc:
            parts.append(compute_gc_content(chunk))
        if include_cpg_density:
            parts.append(compute_cpg_ratio(chunk))
        if include_cpg_island:
            parts.append(detect_cpg_islands(chunk))
        if include_kmer:
            parts.append(compute_kmer_counts(chunk, k=4, top_k=64))

        results.append(torch.cat(parts, dim=1))

    return torch.cat(results, dim=0)


# 预计算特征维度（不含 k-mer）
# GC: (20000 - 500) // 100 + 1 = 196
# CpG O/E: 196
# CpG island density: 196
# Total: 588
SEQ_FEAT_DIM = 588  # GC(196) + CpG_OE(196) + CpG_island(196)


if __name__ == "__main__":
    # 单元测试
    B, L = 4, 20000
    x = torch.zeros(B, L, 4)
    # 随机 one-hot
    idx = torch.randint(0, 4, (B, L))
    x.scatter_(2, idx.unsqueeze(2), 1.0)

    print("序列内在特征计算测试")
    gc = compute_gc_content(x)
    print(f"  GC 含量: {gc.shape}  范围 [{gc.min():.3f}, {gc.max():.3f}]")

    cpg = compute_cpg_ratio(x)
    print(f"  CpG O/E: {cpg.shape}  范围 [{cpg.min():.3f}, {cpg.max():.3f}]")

    island = detect_cpg_islands(x)
    print(f"  CpG 岛密度: {island.shape}  范围 [{island.min():.3f}, {island.max():.3f}]")

    feats = compute_sequence_features(x)
    print(f"  全部特征: {feats.shape}  (预期 {SEQ_FEAT_DIM})")
    assert feats.shape == (B, SEQ_FEAT_DIM), f"维度不匹配: {feats.shape} vs {(B, SEQ_FEAT_DIM)}"

    # k-mer 测试（可选，较慢）
    kmer = compute_kmer_counts(x, k=4, top_k=64)
    print(f"  k-mer 特征: {kmer.shape}")

    print("全部测试通过")
