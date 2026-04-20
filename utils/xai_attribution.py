"""XAI 归因计算：DeepLIFT + IntegratedGradients 碱基级重要性。

使用 Captum 库对 promoter DNA 序列进行 per-base 归因分析。
输出 (N, 20000) 碱基级重要性分数。
"""

import torch
import numpy as np


class PromoterOnlyWrapper(torch.nn.Module):
    """包装 XAI 模型，固定 halflife 输入，仅对 promoter 归因。"""

    def __init__(self, model, halflife: torch.Tensor):
        super().__init__()
        self.model = model
        self.halflife = halflife.detach()

    def forward(self, promoter: torch.Tensor) -> torch.Tensor:
        return self.model.forward_logits_only(promoter)


def compute_deeplift(
    model,
    promoter: torch.Tensor,
    halflife: torch.Tensor,
    target_class: int = 1,
    device: torch.device | None = None,
) -> np.ndarray:
    """DeepLIFT 归因计算。

    Baseline: 全零（语义：无碱基信息）。
    使用 halflife 均值作为固定参考，归因纯粹反映 promoter 贡献。

    Args:
        model: GeneExpressTransformerXAI 实例
        promoter: (N, 20000, 4) one-hot DNA
        halflife: (N, 8) 标准化特征
        target_class: 归因目标类（1=高表达）
        device: 计算设备

    Returns:
        attributions: (N, 20000) 碱基级重要性分数
    """
    from captum.attr import DeepLift

    if device is None:
        device = next(model.parameters()).device

    model.eval()
    # 使用 halflife 均值作为固定参考，归因纯粹反映 promoter 贡献
    mean_hl = halflife.mean(dim=0, keepdim=True).to(device)
    model.set_fixed_halflife(mean_hl)
    wrapper = PromoterOnlyWrapper(model, mean_hl)
    wrapper.to(device)
    wrapper.eval()

    dl = DeepLift(wrapper)
    promoter_dev = promoter.to(device)
    baseline = torch.zeros_like(promoter_dev)

    attr = dl.attribute(promoter_dev, baselines=baseline, target=target_class)

    # 取实际碱基通道的归因值
    base_attr = (attr * promoter_dev).sum(dim=-1)

    return base_attr.detach().cpu().numpy()


def compute_integrated_gradients(
    model,
    promoter: torch.Tensor,
    halflife: torch.Tensor,
    target_class: int = 1,
    n_steps: int = 50,
    device: torch.device | None = None,
) -> np.ndarray:
    """Integrated Gradients 归因计算（交叉验证用）。

    Args:
        model: GeneExpressTransformerXAI 实例
        promoter: (N, 20000, 4)
        halflife: (N, 8)
        target_class: 归因目标类
        n_steps: 积分步数
        device: 计算设备

    Returns:
        attributions: (N, 20000) 碱基级重要性分数
    """
    from captum.attr import IntegratedGradients

    if device is None:
        device = next(model.parameters()).device

    model.eval()
    mean_hl = halflife.mean(dim=0, keepdim=True).to(device)
    model.set_fixed_halflife(mean_hl)
    wrapper = PromoterOnlyWrapper(model, mean_hl)
    wrapper.to(device)
    wrapper.eval()

    ig = IntegratedGradients(wrapper)
    promoter_dev = promoter.to(device)
    baseline = torch.zeros_like(promoter_dev)

    attr = ig.attribute(
        promoter_dev, baselines=baseline, target=target_class, n_steps=n_steps
    )

    base_attr = (attr * promoter_dev).sum(dim=-1)

    return base_attr.detach().cpu().numpy()


def compute_deeplift_batched(
    model,
    promoter: torch.Tensor,
    halflife: torch.Tensor,
    target_class: int = 1,
    batch_size: int = 16,
    device: torch.device | None = None,
) -> np.ndarray:
    """分批 DeepLIFT 计算（避免 OOM）。

    Args:
        model: GeneExpressTransformerXAI 实例
        promoter: (N, 20000, 4)
        halflife: (N, 8)
        target_class: 归因目标类
        batch_size: 每批样本数
        device: 计算设备

    Returns:
        attributions: (N, 20000) 碱基级重要性分数
    """
    n_samples = promoter.size(0)
    all_attr = []

    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        batch_p = promoter[start:end]
        batch_h = halflife[start:end]

        attr = compute_deeplift(model, batch_p, batch_h, target_class, device)
        all_attr.append(attr)

    return np.concatenate(all_attr, axis=0)


def extract_conv_filters(
    model, top_k: int = 10
) -> dict[str, np.ndarray]:
    """提取 CNN 第一层卷积核权重 -> PWM motif。

    Args:
        model: GeneExpressTransformerXAI 实例
        top_k: 返回激活最强的前 k 个滤波器

    Returns:
        dict: {
            'conv_a': (top_k, 4, 8),   # k=8 的短 motif
            'conv_b': (top_k, 4, 16),  # k=16 的中等 motif
            'conv_c': (top_k, 4, 32),  # k=32 的长 motif
        }
    """
    result = {}
    for name in ["conv_a", "conv_b", "conv_c"]:
        weight = getattr(model, name).weight.data.cpu().numpy()  # (out, 4, k)
        # 按滤波器 L2 范数排序
        norms = np.linalg.norm(weight.reshape(weight.shape[0], -1), axis=1)
        top_idx = np.argsort(norms)[-top_k:][::-1]
        result[name] = weight[top_idx]
    return result


def identify_key_regions(
    attribution: np.ndarray,
    percentile: float = 95,
    min_length: int = 6,
) -> list[dict]:
    """从归因分数中识别连续高重要性区域。

    Args:
        attribution: (20000,) 单样本碱基级重要性分数
        percentile: 高重要性阈值百分位
        min_length: 最短连续区域长度

    Returns:
        list of {
            'start': int, 'end': int, 'length': int,
            'mean_score': float, 'max_score': float
        }
    """
    threshold = np.percentile(attribution, percentile)
    above = attribution >= threshold

    regions = []
    start = None
    for i in range(len(above)):
        if above[i] and start is None:
            start = i
        elif not above[i] and start is not None:
            if i - start >= min_length:
                region_attr = attribution[start:i]
                regions.append({
                    "start": int(start),
                    "end": int(i),
                    "length": int(i - start),
                    "mean_score": float(region_attr.mean()),
                    "max_score": float(region_attr.max()),
                })
            start = None

    # 处理尾部
    if start is not None and len(above) - start >= min_length:
        region_attr = attribution[start:]
        regions.append({
            "start": int(start),
            "end": int(len(above)),
            "length": int(len(above) - start),
            "mean_score": float(region_attr.mean()),
            "max_score": float(region_attr.max()),
        })

    # 按平均分数降序排列
    regions.sort(key=lambda r: r["mean_score"], reverse=True)
    return regions
