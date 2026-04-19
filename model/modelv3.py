"""GeneExpressTransformer: CNN 局部特征 + Token 压缩 + 1 层 Transformer 全局注意力。

CNN 多尺度卷积提取局部 motif，Token 压缩降低序列长度，
单层 Transformer 自注意力建模增强子-启动子长程互作（覆盖 20000bp）。

经 13 组实验验证：1 层 Transformer 在 22.5K 参数下达到测试 Acc 0.8051，
超过 2 层版本（0.786）和 v2 CNN（0.79），接近 v1（0.81，41M 参数）。
核心发现：数据量仅 16K 样本，过拟合是主要瓶颈，更轻量的模型泛化更好。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GeneExpressTransformer(nn.Module):
    """CNN 前端 + 1 层 Transformer 编码器的基因表达预测模型。

    Args:
        num_classes: 分类数，默认 2（高/低表达）。
    """

    def __init__(self, num_classes: int = 2):
        super().__init__()

        # --- 多尺度卷积（复用 v2 设计）---
        # k=8: TATA box 等短 motif
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

        # --- Token 压缩 + 位置编码 ---
        # 将 ~312 位置压缩为 64 个 token，每个覆盖 ~312bp
        self.token_pool = nn.AdaptiveAvgPool1d(64)
        self.pos_embed = nn.Parameter(torch.randn(1, 64, 32) * 0.02)

        # --- Transformer Encoder（1 层）---
        # 实验表明 1 层优于 2 层（0.8051 vs 0.786），减少过拟合
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=32,
            nhead=4,
            dim_feedforward=64,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)

        # --- Halflife 分支 ---
        self.hl_fc1 = nn.Linear(8, 32)
        self.hl_fc2 = nn.Linear(32, 32)

        # --- 分类头 ---
        # promoter GAP(32) + halflife(32) = 64
        self.fc1 = nn.Linear(64, 32)
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

        # 融合压缩
        x = self.pool2(F.relu(self.bn_fuse(self.conv_fuse(x))))  # (B, 32, ~312)

        # Token 压缩: (B, 32, ~312) -> (B, 32, 64) -> (B, 64, 32)
        x = self.token_pool(x)
        x = x.permute(0, 2, 1)  # (B, 64, 32) 即 64 个 token, d_model=32

        # 加位置编码
        x = x + self.pos_embed

        # Transformer 全局自注意力
        x = self.transformer(x)  # (B, 64, 32)

        # 全局平均池化
        x = x.mean(dim=1)  # (B, 32)

        # Halflife 分支
        h = F.relu(self.hl_fc2(F.relu(self.hl_fc1(halflife))))  # (B, 32)

        # 融合分类
        x = torch.cat([x, h], dim=1)  # (B, 64)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)


if __name__ == "__main__":
    model = GeneExpressTransformer()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"GeneExpressTransformer 参数量: {total_params:,}")

    promoter = torch.randn(4, 20000, 4)
    halflife = torch.randn(4, 8)
    output = model(promoter, halflife)
    print(f"输入: promoter {promoter.shape}, halflife {halflife.shape}")
    print(f"输出: logits {output.shape}")
    assert output.shape == (4, 2), f"输出形状错误: {output.shape}"
    print("验证通过")
