"""GeneExpressTransformerV4: 三分支模型 (promoter + halflife + epigenomic)。

在 v3 基础上新增 epigenomic 分支，支持：
- 序列内在特征（GC、CpG 等）：FC 分支
- ENCODE 表观信号（3 通道 x 200 bin）：Conv1d 分支
- 两者可独立或联合使用

分类头：三分支拼接 144 维 → FC(144→64) → FC(64→2)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.modelv3 import GeneExpressTransformer


class EpigenomicBranch(nn.Module):
    """表观基因组信号分支。

    支持 ENCODE 3 通道信号 (B, 3, 200) 和序列内在特征 (B, seq_feat_dim)。
    两种输入可同时使用，输出统一为 48 维。
    """

    def __init__(
        self,
        encode_channels: int = 3,
        encode_bins: int = 200,
        seq_feat_dim: int = 0,
        out_dim: int = 48,
    ):
        super().__init__()
        self.use_encode = encode_channels > 0
        self.use_seq_feat = seq_feat_dim > 0

        encode_out_dim = 0

        if self.use_encode:
            # 3 通道信号 → Conv1d → GAP → FC
            self.encode_conv = nn.Sequential(
                nn.Conv1d(encode_channels, 32, kernel_size=3, padding=1),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.MaxPool1d(4),
                nn.Conv1d(32, 32, kernel_size=3, padding=1),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
            )
            encode_out_dim = 32

        if self.use_seq_feat:
            # 序列内在特征 → FC
            self.seq_fc = nn.Sequential(
                nn.Linear(seq_feat_dim, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, 32),
                nn.ReLU(),
            )
            encode_out_dim += 32

        # 统一映射到 out_dim
        if encode_out_dim > 0:
            self.fc_out = nn.Linear(encode_out_dim, out_dim)

        self._out_dim = out_dim if (self.use_encode or self.use_seq_feat) else 0

    @property
    def output_dim(self) -> int:
        return self._out_dim

    def forward(
        self,
        encode_signal: torch.Tensor | None = None,
        seq_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """前向传播。

        Args:
            encode_signal: (B, 3, 200) ENCODE 信号
            seq_features: (B, seq_feat_dim) 序列内在特征

        Returns:
            (B, out_dim) 表观特征向量
        """
        parts = []

        if self.use_encode and encode_signal is not None:
            x = self.encode_conv(encode_signal)  # (B, 32, 1)
            x = x.squeeze(2)  # (B, 32)
            parts.append(x)

        if self.use_seq_feat and seq_features is not None:
            x = self.seq_fc(seq_features)  # (B, 32)
            parts.append(x)

        if not parts:
            raise ValueError("epigenomic 分支无有效输入")

        combined = torch.cat(parts, dim=1)
        return self.fc_out(combined)


class GeneExpressTransformerV4(nn.Module):
    """三分支模型: promoter CNN+Transformer + halflife FC + epigenomic。

    Args:
        num_classes: 分类数
        encode_channels: ENCODE 信号通道数 (0=不使用)
        encode_bins: ENCODE 信号 bin 数
        seq_feat_dim: 序列内在特征维度 (0=不使用)
        promoter_dim: promoter 分支输出维度
        halflife_dim: halflife 分支输出维度
        epi_dim: epigenomic 分支输出维度
    """

    def __init__(
        self,
        num_classes: int = 2,
        encode_channels: int = 3,
        encode_bins: int = 200,
        seq_feat_dim: int = 0,
        promoter_dim: int = 48,
        halflife_dim: int = 48,
        epi_dim: int = 48,
    ):
        super().__init__()

        # --- Promoter 分支 (复用 v3 的 CNN + Transformer) ---
        self.promoter_dim = promoter_dim
        # 导入 v3 的 backbone 部分
        self.conv_a = nn.Conv1d(4, 24, kernel_size=8)
        self.conv_b = nn.Conv1d(4, 24, kernel_size=16)
        self.conv_c = nn.Conv1d(4, 24, kernel_size=32)
        self.bn_a = nn.BatchNorm1d(24)
        self.bn_b = nn.BatchNorm1d(24)
        self.bn_c = nn.BatchNorm1d(24)
        self.pool1 = nn.MaxPool1d(8)

        self.conv_fuse = nn.Conv1d(72, 48, 3, padding=1)
        self.bn_fuse = nn.BatchNorm1d(48)
        self.pool2 = nn.MaxPool1d(8)

        self.token_pool = nn.AdaptiveAvgPool1d(64)
        self.pos_embed = nn.Parameter(torch.randn(1, 64, 48) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=48,
            nhead=4,
            dim_feedforward=96,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)

        # --- Halflife 分支 ---
        self.halflife_dim = halflife_dim
        self.hl_fc1 = nn.Linear(8, halflife_dim)
        self.hl_fc2 = nn.Linear(halflife_dim, halflife_dim)

        # --- Epigenomic 分支 ---
        self.epi_dim = epi_dim
        use_epi = encode_channels > 0 or seq_feat_dim > 0
        if use_epi:
            self.epi_branch = EpigenomicBranch(
                encode_channels=encode_channels,
                encode_bins=encode_bins,
                seq_feat_dim=seq_feat_dim,
                out_dim=epi_dim,
            )
        else:
            self.epi_branch = None

        # --- 分类头 ---
        total_dim = promoter_dim + halflife_dim + (epi_dim if use_epi else 0)
        self.fc1 = nn.Linear(total_dim, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(
        self,
        promoter: torch.Tensor,
        halflife: torch.Tensor,
        encode_signal: torch.Tensor | None = None,
        seq_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """前向传播。

        Args:
            promoter: (B, 20000, 4)
            halflife: (B, 8)
            encode_signal: (B, 3, 200) 或 None
            seq_features: (B, seq_feat_dim) 或 None

        Returns:
            logits: (B, num_classes)
        """
        # Promoter 分支
        x = promoter.permute(0, 2, 1)  # (B, 4, 20000)
        xa = self.pool1(F.relu(self.bn_a(self.conv_a(x))))
        xb = self.pool1(F.relu(self.bn_b(self.conv_b(x))))
        xc = self.pool1(F.relu(self.bn_c(self.conv_c(x))))
        min_len = min(xa.size(2), xb.size(2), xc.size(2))
        x = torch.cat(
            [xa[:, :, :min_len], xb[:, :, :min_len], xc[:, :, :min_len]], dim=1
        )
        x = self.pool2(F.relu(self.bn_fuse(self.conv_fuse(x))))
        x = self.token_pool(x)
        x = x.permute(0, 2, 1)
        x = x + self.pos_embed
        x = self.transformer(x)
        x = x.mean(dim=1)  # (B, 48)

        # Halflife 分支
        h = F.relu(self.hl_fc2(F.relu(self.hl_fc1(halflife))))

        # 拼接
        parts = [x, h]

        # Epigenomic 分支
        if self.epi_branch is not None:
            epi_out = self.epi_branch(encode_signal, seq_features)
            parts.append(epi_out)

        combined = torch.cat(parts, dim=1)
        combined = self.dropout(F.relu(self.fc1(combined)))
        return self.fc2(combined)


if __name__ == "__main__":
    # 测试 v4 模型（含全部特征）
    print("GeneExpressTransformerV4 测试")
    print()

    # 配置 1: 全部特征
    model = GeneExpressTransformerV4(
        encode_channels=3,
        seq_feat_dim=588,
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"全特征模型: {total_params:,} 参数")

    promoter = torch.randn(4, 20000, 4)
    halflife = torch.randn(4, 8)
    encode = torch.randn(4, 3, 200)
    seq_feat = torch.randn(4, 588)
    out = model(promoter, halflife, encode, seq_feat)
    print(f"  输出: {out.shape}")
    assert out.shape == (4, 2)

    # 配置 2: 仅 ENCODE
    model2 = GeneExpressTransformerV4(
        encode_channels=3,
        seq_feat_dim=0,
    )
    params2 = sum(p.numel() for p in model2.parameters())
    print(f"仅 ENCODE: {params2:,} 参数")
    out2 = model2(promoter, halflife, encode)
    assert out2.shape == (4, 2)

    # 配置 3: 仅序列特征
    model3 = GeneExpressTransformerV4(
        encode_channels=0,
        seq_feat_dim=588,
    )
    params3 = sum(p.numel() for p in model3.parameters())
    print(f"仅序列特征: {params3:,} 参数")
    out3 = model3(promoter, halflife, seq_features=seq_feat)
    assert out3.shape == (4, 2)

    # 配置 4: 无 epigenomic (等价 v3)
    model4 = GeneExpressTransformerV4(
        encode_channels=0,
        seq_feat_dim=0,
    )
    params4 = sum(p.numel() for p in model4.parameters())
    print(f"无 epigenomic: {params4:,} 参数")
    out4 = model4(promoter, halflife)
    assert out4.shape == (4, 2)

    print()
    print("全部配置测试通过")
