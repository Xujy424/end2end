import torch


class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask


class ProbMask():
    def __init__(self, B, H, L, index, scores, device="cpu"):
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[torch.arange(B)[:, None, None],
                    torch.arange(H)[None, :, None],
                    index, :].to(device)
        self._mask = indicator.view(scores.shape).to(device)

    @property
    def mask(self):
        return self._mask


class PriceTradeCausalMask():
    """
    自定义因果掩码：用于前向任务，约束价格特征和交易特征的关注范围。
    假设 price_indices 和 trade_indices 已知，这里简化为：
    - 所有特征在时间上遵循因果，但价格特征不能关注同一时刻的价格特征。
    实际中需根据特征索引精确控制。
    """
    def __init__(self, B, L, device="cpu", price_only=False):
        """
        price_only: 若为True，生成仅价格特征可用的掩码（预测价格时使用）；
                    若为False，生成所有特征可用的标准因果掩码（用于其他层）。
        """
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            # 标准上三角掩码（禁止未来）
            base_mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)
            if price_only:
                # 额外禁止当前时刻的价格特征关注自身（对角线）
                # 这里简化处理：将对角线也设为True（禁止），即价格不能看自己
                # 注意：实际应根据特征分离，此处仅示意
                diag_mask = torch.eye(L, dtype=torch.bool).to(device).unsqueeze(0).unsqueeze(0)
                self._mask = base_mask | diag_mask
            else:
                self._mask = base_mask

    @property
    def mask(self):
        return self._mask




class FullMask():
    """全注意力掩码（所有位置可见）"""
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.zeros(mask_shape, dtype=torch.bool).to(device)  # False 表示不掩码

    @property
    def mask(self):
        return self._mask

class PaddingMask:
    """将 [B, S] 有效位掩码转换为注意力掩码。"""

    def __init__(self, valid_mask):
        if valid_mask.dim() != 2:
            raise ValueError("valid_mask 的形状必须是 [B, S]")
        # 输入 True/1 表示有效；注意力掩码 True 表示屏蔽。
        self._mask = ~valid_mask.bool()[:, None, None, :]

    @property
    def mask(self):
        return self._mask


def combine_masks(*masks):
    """合并多个可广播的布尔掩码；None 会被忽略。"""
    result = None
    for mask in masks:
        if mask is None:
            continue
        mask = mask.mask if hasattr(mask, "mask") else mask
        mask = mask.bool()
        result = mask if result is None else (result | mask)
    return result
