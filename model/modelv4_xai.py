"""GeneExpressTransformerV4 XAI 变体：暴露注意力权重 + 支持三分支 Captum 归因。

在 modelv4 基础上替换 Transformer 组件以暴露注意力矩阵，
提供 forward_logits_only() 方法供 Captum 使用（固定 halflife/encode/seq 输入）。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.modelv4 import GeneExpressTransformerV4


class TransformerEncoderLayerWithAttn(nn.TransformerEncoderLayer):
    """重写 forward 以捕获注意力权重。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attn_weights: torch.Tensor | None = None

    def forward(self, src: torch.Tensor, **kwargs) -> torch.Tensor:
        x = src
        if self.norm_first:
            nx = self.norm1(x)
            attn_out, weights = self.self_attn(
                nx, nx, nx, need_weights=True, average_attn_weights=False,
            )
            self.attn_weights = weights
            x = x + self.dropout1(attn_out)
            x = x + self._ff_block(self.norm2(x))
        else:
            attn_out, weights = self.self_attn(
                x, x, x, need_weights=True, average_attn_weights=False,
            )
            self.attn_weights = weights
            x = self.norm1(x + self.dropout1(attn_out))
            x = self.norm2(x + self._ff_block(x))
        return x


class TransformerEncoderWithAttn(nn.Module):
    """逐层收集注意力矩阵。"""

    def __init__(self, layer: TransformerEncoderLayerWithAttn):
        super().__init__()
        self.layers = nn.ModuleList([layer])

    def forward(self, src: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        attn_list = []
        x = src
        for layer in self.layers:
            x = layer(x)
            attn_list.append(layer.attn_weights)
        return x, attn_list


class GeneExpressTransformerV4XAI(GeneExpressTransformerV4):
    """V4 XAI 变体：forward 返回 (logits, attn_weights)。

    支持三分支的 DeepLIFT 归因分析：
    - promoter 分支：碱基级归因（主要分析目标）
    - halflife / encode / seq：可固定或分析
    """

    def __init__(self, num_classes=2, encode_channels=3, seq_feat_dim=588):
        super().__init__(
            num_classes=num_classes,
            encode_channels=encode_channels,
            seq_feat_dim=seq_feat_dim,
        )

        # 替换 Transformer 组件
        xai_layer = TransformerEncoderLayerWithAttn(
            d_model=48, nhead=4, dim_feedforward=96,
            dropout=0.1, batch_first=True, activation="gelu",
        )
        orig_layer = self.transformer.layers[0]
        xai_layer.load_state_dict(orig_layer.state_dict())
        self.transformer = TransformerEncoderWithAttn(xai_layer)

        # 固定辅助输入（归因分析时使用）
        self._fixed_halflife: torch.Tensor | None = None
        self._fixed_encode: torch.Tensor | None = None
        self._fixed_seq: torch.Tensor | None = None

    def forward(
        self,
        promoter: torch.Tensor,
        halflife: torch.Tensor,
        encode_signal: torch.Tensor | None = None,
        seq_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """返回 (logits, attn_weights)。"""
        x = promoter.permute(0, 2, 1)

        xa = F.max_pool1d(F.relu(self.bn_a(self.conv_a(x))), 8)
        xb = F.max_pool1d(F.relu(self.bn_b(self.conv_b(x))), 8)
        xc = F.max_pool1d(F.relu(self.bn_c(self.conv_c(x))), 8)
        min_len = min(xa.size(2), xb.size(2), xc.size(2))
        x = torch.cat(
            [xa[:, :, :min_len], xb[:, :, :min_len], xc[:, :, :min_len]], dim=1
        )
        x = F.max_pool1d(F.relu(self.bn_fuse(self.conv_fuse(x))), 8)

        x = F.adaptive_avg_pool1d(x, 64)
        x = x.permute(0, 2, 1)
        x = x + self.pos_embed

        x, attn_weights = self.transformer(x)
        x = x.mean(dim=1)

        h = F.relu(self.hl_fc2(F.relu(self.hl_fc1(halflife))))

        parts = [x, h]

        if self.epi_branch is not None:
            epi_out = self.epi_branch(encode_signal, seq_features)
            parts.append(epi_out)

        combined = torch.cat(parts, dim=1)
        combined = self.dropout(F.relu(self.fc1(combined)))
        logits = self.fc2(combined)

        return logits, attn_weights

    def forward_logits_only(self, promoter: torch.Tensor) -> torch.Tensor:
        """Captum 归因接口：仅接受 promoter 单 tensor。"""
        assert self._fixed_halflife is not None, "请先调用 set_fixed_inputs()"
        batch_size = promoter.size(0)
        hl = self._fixed_halflife
        if hl.size(0) != batch_size:
            hl = hl.expand(batch_size, -1)

        enc = self._fixed_encode
        if enc is not None and enc.size(0) != batch_size:
            enc = enc.expand(batch_size, -1, -1)

        sf = self._fixed_seq
        if sf is not None and sf.size(0) != batch_size:
            sf = sf.expand(batch_size, -1)

        logits, _ = self.forward(promoter, hl, encode_signal=enc, seq_features=sf)
        return logits

    def set_fixed_inputs(
        self,
        halflife: torch.Tensor,
        encode: torch.Tensor | None = None,
        seq_feat: torch.Tensor | None = None,
    ) -> None:
        """固定辅助输入用于归因分析。"""
        if halflife.dim() == 1:
            halflife = halflife.unsqueeze(0)
        self._fixed_halflife = halflife[:1].detach()
        self._fixed_encode = encode[:1].detach() if encode is not None else None
        self._fixed_seq = seq_feat[:1].detach() if seq_feat is not None else None

    def load_from_base(self, state_dict: dict) -> None:
        """从原始 GeneExpressTransformerV4 权重加载。"""
        xai_state = {}
        for key, value in state_dict.items():
            xai_state[key] = value
        missing, unexpected = self.load_state_dict(xai_state, strict=False)
        if missing:
            self.load_state_dict(state_dict, strict=True)


if __name__ == "__main__":
    model = GeneExpressTransformerV4XAI(encode_channels=3, seq_feat_dim=588)
    total = sum(p.numel() for p in model.parameters())
    print(f"V4 XAI 模型参数量: {total:,}")

    promoter = torch.randn(2, 20000, 4)
    halflife = torch.randn(2, 8)
    encode = torch.randn(2, 3, 200)
    seq = torch.randn(2, 588)

    model.set_fixed_inputs(halflife, encode, seq)
    logits_only = model.forward_logits_only(promoter)
    print(f"forward_logits_only: {logits_only.shape}")

    logits, attn = model(promoter, halflife, encode, seq)
    print(f"forward: logits {logits.shape}, attn {len(attn)} layers, shape {attn[0].shape}")
    print("验证通过")
