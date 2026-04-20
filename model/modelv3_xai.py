"""GeneExpressTransformer XAI 变体：暴露注意力权重，支持 Captum 归因分析。

继承 modelv3.GeneExpressTransformer，替换 Transformer 组件以暴露注意力矩阵。
提供 forward_logits_only() 方法供 Captum 使用（单 tensor 输入）。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.modelv3 import GeneExpressTransformer


class TransformerEncoderLayerWithAttn(nn.TransformerEncoderLayer):
    """重写 forward 以捕获注意力权重，绕过 PyTorch fastpath。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attn_weights: torch.Tensor | None = None

    def forward(self, src: torch.Tensor, **kwargs) -> torch.Tensor:
        """手动执行 norm -> attn -> norm -> ff，捕获注意力矩阵。"""
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
    """逐层收集注意力矩阵的 Transformer Encoder。"""

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


class GeneExpressTransformerXAI(GeneExpressTransformer):
    """XAI 变体：forward 返回 (logits, attn_weights)。

    注意力矩阵维度：(B, nhead, seq_len, seq_len) = (B, 4, 64, 64)。
    64 token 映射到 20000bp（每 token ~312bp）。
    """

    def __init__(self, num_classes: int = 2):
        super().__init__(num_classes=num_classes)

        # 替换 Transformer 组件
        xai_layer = TransformerEncoderLayerWithAttn(
            d_model=48,
            nhead=4,
            dim_feedforward=96,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        # 复制原始权重到 XAI 层
        orig_layer = self.transformer.layers[0]
        xai_layer.load_state_dict(orig_layer.state_dict())

        self.transformer = TransformerEncoderWithAttn(xai_layer)

        # 固定 halflife（归因分析时使用）
        self._fixed_halflife: torch.Tensor | None = None

    def forward(
        self, promoter: torch.Tensor, halflife: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """返回 (logits, attn_weights_list)。

        Args:
            promoter: (B, 20000, 4)
            halflife: (B, 8)

        Returns:
            logits: (B, 2)
            attn_weights: list of (B, 4, 64, 64)
        """
        x = promoter.permute(0, 2, 1)

        # 使用函数式池化避免 module 复用（DeepLift 要求）
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

        x = torch.cat([x, h], dim=1)
        x = self.dropout(F.relu(self.fc1(x)))
        logits = self.fc2(x)

        return logits, attn_weights

    def forward_logits_only(self, promoter: torch.Tensor) -> torch.Tensor:
        """Captum 归因接口：仅接受 promoter 单 tensor 输入。

        Args:
            promoter: (B, 20000, 4)

        Returns:
            logits: (B, 2)
        """
        assert self._fixed_halflife is not None, "请先调用 set_fixed_halflife()"
        batch_size = promoter.size(0)
        hl = self._fixed_halflife
        # Captum 内部可能改变 batch size，动态扩展 halflife
        if hl.size(0) != batch_size:
            hl = hl.expand(batch_size, -1)
        logits, _ = self.forward(promoter, hl)
        return logits

    def set_fixed_halflife(self, halflife: torch.Tensor) -> None:
        """固定 halflife 用于归因分析。

        Args:
            halflife: (B, 8) 或 (8,)
        """
        if halflife.dim() == 1:
            halflife = halflife.unsqueeze(0)
        # 存储为 (1, 8) 以便 expand 到任意 batch size
        self._fixed_halflife = halflife[:1].detach()

    def load_from_base(self, state_dict: dict) -> None:
        """从原始 GeneExpressTransformer 权重加载。

        自动将 transformer.layers.0.* 映射到 XAI 变体。
        """
        xai_state = {}
        for key, value in state_dict.items():
            if key.startswith("transformer."):
                new_key = key.replace("transformer.layers.0.", "transformer.layers.0.")
                xai_state[new_key] = value
            else:
                xai_state[key] = value

        # 处理可能的键名不匹配
        missing, unexpected = self.load_state_dict(xai_state, strict=False)
        if missing:
            # 尝试直接加载
            self.load_state_dict(state_dict, strict=True)


def convert_to_xai_model(
    base_model: GeneExpressTransformer,
) -> GeneExpressTransformerXAI:
    """将已训练的 base model 转换为 XAI 模型。

    Args:
        base_model: 已加载权重的 GeneExpressTransformer

    Returns:
        权重已复制的 GeneExpressTransformerXAI
    """
    xai_model = GeneExpressTransformerXAI(num_classes=2)
    state_dict = base_model.state_dict()
    xai_model.load_from_base(state_dict)
    return xai_model
