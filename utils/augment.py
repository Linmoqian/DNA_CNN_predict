"""DNA 序列数据增强。

三种生物学合理的增强方法：
1. 反向互补 — DNA 双链对称，正/反链调控功能等价
2. 随机平移 — 模拟启动子注释边界微小偏移
3. 随机遮蔽 — 模拟测序缺失，增强鲁棒性
"""

import torch
from torch.utils.data import Dataset


def augment_promoter(
    promoter: torch.Tensor,
    rc_prob: float = 0.5,
    shift_max: int = 200,
    mask_prob: float = 0.01,
) -> torch.Tensor:
    """对单个 promoter 样本应用随机增强。

    Args:
        promoter: (20000, 4) one-hot tensor
        rc_prob: 反向互补概率
        shift_max: 最大平移碱基数
        mask_prob: 逐位置遮蔽概率

    Returns:
        augmented: (20000, 4) tensor
    """
    p = promoter.clone()

    # 反向互补: 翻转序列维度 + 翻转碱基维度 (A↔T, C↔G)
    if torch.rand(1).item() < rc_prob:
        p = p.flip(0).flip(1)

    # 随机平移: 循环移位 ±shift_max bp
    if shift_max > 0:
        offset = torch.randint(-shift_max, shift_max + 1, (1,)).item()
        if offset != 0:
            p = torch.roll(p, shifts=offset, dims=0)

    # 随机遮蔽: 将 mask_prob 比例位置置零
    if mask_prob > 0:
        mask = torch.rand(p.size(0)) < mask_prob
        p[mask] = 0.0

    return p


class AugmentedDataset(Dataset):
    """带增强的数据集，仅对训练集启用。"""

    def __init__(
        self,
        promoters: torch.Tensor,
        halflifes: torch.Tensor,
        labels: torch.Tensor,
        augment: bool = True,
        rc_prob: float = 0.5,
        shift_max: int = 200,
        mask_prob: float = 0.01,
    ):
        self.promoters = promoters
        self.halflifes = halflifes
        self.labels = labels
        self.augment = augment
        self.rc_prob = rc_prob
        self.shift_max = shift_max
        self.mask_prob = mask_prob

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        p = self.promoters[idx]
        h = self.halflifes[idx]
        l = self.labels[idx]
        if self.augment:
            p = augment_promoter(p, self.rc_prob, self.shift_max, self.mask_prob)
        return p, h, l
