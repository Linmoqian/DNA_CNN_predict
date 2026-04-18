"""GeneExpressNet: 基于生物学先验的轻量级 DNA 序列基因表达预测模型。

多尺度卷积核捕获不同长度 TF motif，空洞卷积扩大感受野，
多层级特征融合保留浅层局部信号与深层全局信号。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GeneExpressNet(nn.Module):
    """双分支 CNN 模型：promoter 多尺度卷积 + halflife 全连接分支。

    Args:
        num_classes: 分类数，默认 2（高/低表达）。
    """

    def __init__(self, num_classes: int = 2):
        super().__init__()

        # --- 多尺度卷积（并行三分支）---
        # k=8:  TATA box 等短 motif
        # k=16: CAAT/GC box 等中等 motif
        # k=32: TF 结合位点等较长 motif
        self.conv_a = nn.Conv1d(4, 16, kernel_size=8)
        self.conv_b = nn.Conv1d(4, 16, kernel_size=16)
        self.conv_c = nn.Conv1d(4, 16, kernel_size=32)
        self.bn_a = nn.BatchNorm1d(16)
        self.bn_b = nn.BatchNorm1d(16)
        self.bn_c = nn.BatchNorm1d(16)
        self.pool1 = nn.MaxPool1d(8)

        # --- 融合压缩 ---
        self.conv_fuse = nn.Conv1d(48, 32, 3, padding=1)
        self.bn_fuse = nn.BatchNorm1d(32)
        self.pool2 = nn.MaxPool1d(8)

        # --- 空洞卷积（扩大感受野，捕获远端调控）---
        # d=2: 覆盖 ~320bp, d=4: 覆盖 ~576bp 原始序列范围
        self.conv_d1 = nn.Conv1d(32, 32, 3, dilation=2, padding=2)
        self.bn_d1 = nn.BatchNorm1d(32)
        self.conv_d2 = nn.Conv1d(32, 32, 3, dilation=4, padding=4)
        self.bn_d2 = nn.BatchNorm1d(32)

        # --- SE 通道注意力 ---
        self.se_fc1 = nn.Linear(32, 8)
        self.se_fc2 = nn.Linear(8, 32)

        # --- Halflife 分支 ---
        self.hl_fc1 = nn.Linear(8, 32)
        self.hl_fc2 = nn.Linear(32, 32)

        # --- 分类头 ---
        # 浅层(融合) 64 + 深层(空洞) 64 + halflife 32 = 160
        self.fc1 = nn.Linear(160, 32)
        self.fc2 = nn.Linear(32, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, promoter: torch.Tensor, halflife: torch.Tensor) -> torch.Tensor:
        """
        Args:
            promoter: (B, 20000, 4) one-hot DNA 序列。
            halflife: (B, 8) 8 维标准化特征。

        Returns:
            logits: (B, num_classes)
        """
        x = promoter.permute(0, 2, 1)  # (B, 4, 20000)

        # 多尺度并行卷积
        xa = self.pool1(F.relu(self.bn_a(self.conv_a(x))))
        xb = self.pool1(F.relu(self.bn_b(self.conv_b(x))))
        xc = self.pool1(F.relu(self.bn_c(self.conv_c(x))))

        min_len = min(xa.size(2), xb.size(2), xc.size(2))
        x = torch.cat(
            [xa[:, :, :min_len], xb[:, :, :min_len], xc[:, :, :min_len]], dim=1
        )  # (B, 48, ~2496)

        # 融合压缩（浅层特征：局部 motif 信号）
        x_shallow = self.pool2(F.relu(self.bn_fuse(self.conv_fuse(x))))  # (B, 32, ~312)

        # 空洞卷积 + 残差连接（深层特征：远端调控信号）
        identity = x_shallow
        x_deep = F.relu(self.bn_d1(self.conv_d1(x_shallow)))
        x_deep = F.relu(self.bn_d2(self.conv_d2(x_deep))) + identity  # (B, 32, ~312)

        # SE 通道注意力（作用于深层）
        w = F.adaptive_avg_pool1d(x_deep, 1).squeeze(-1)  # (B, 32)
        w = torch.sigmoid(self.se_fc2(F.relu(self.se_fc1(w))))  # (B, 32)
        x_deep = x_deep * w.unsqueeze(-1)

        # 多层级双路径池化
        shallow_avg = F.adaptive_avg_pool1d(x_shallow, 1).squeeze(-1)  # (B, 32)
        shallow_mx = F.adaptive_max_pool1d(x_shallow, 1).squeeze(-1)  # (B, 32)
        deep_avg = F.adaptive_avg_pool1d(x_deep, 1).squeeze(-1)  # (B, 32)
        deep_mx = F.adaptive_max_pool1d(x_deep, 1).squeeze(-1)  # (B, 32)
        x = torch.cat([shallow_avg, shallow_mx, deep_avg, deep_mx], dim=1)  # (B, 128)

        # Halflife 分支
        h = F.relu(self.hl_fc2(F.relu(self.hl_fc1(halflife))))  # (B, 32)

        # 融合分类
        x = torch.cat([x, h], dim=1)  # (B, 160)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)


if __name__ == "__main__":
    model = GeneExpressNet()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"GeneExpressNet 参数量: {total_params:,}")

    promoter = torch.randn(4, 20000, 4)
    halflife = torch.randn(4, 8)
    output = model(promoter, halflife)
    print(f"输入: promoter {promoter.shape}, halflife {halflife.shape}")
    print(f"输出: logits {output.shape}")
    assert output.shape == (4, 2), f"输出形状错误: {output.shape}"
    print("验证通过")
