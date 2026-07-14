import torch.nn as nn
import torch as th
import torch.nn.functional as F


class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, preds, labels):
        return th.sqrt(self.mse(preds, labels))


class MonotonicLogisticLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, preds, labels):
        # preds: [B,N], labels: [B,N]
        p_diff = preds.unsqueeze(-1) - preds.unsqueeze(-2)
        y_diff = labels.unsqueeze(-1) - labels.unsqueeze(-2)
        loss = th.log(1 + th.exp(-th.tanh(p_diff) * th.tanh(y_diff)))
        return loss.mean()


class GraphProximityLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, o, adj):
        sim = o @ o.T  # [N, N]
        sim_sig = th.sigmoid(sim)  # [N, N]
        pos_loss = -th.log(sim_sig + 1e-8)  # 加1e-8防log(0)
        neg_loss = -th.log(1 - sim_sig + 1e-8)
        loss = (pos_loss * adj + neg_loss * (1 - adj)).mean()
        return loss


class ICLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, preds, labels):
        pred_mean = th.mean(preds)
        target_mean = th.mean(labels)
        cov = th.mean((preds - pred_mean) * (labels - target_mean))
        pred_std = th.std(preds)
        target_std = th.std(labels)
        ic = cov / (pred_std * target_std)
        return 1 - ic


class PairwiseRankLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, preds, labels):
        valid_stock = th.isfinite(preds) & th.isfinite(labels)  # [N]
        p_diff = preds.unsqueeze(0) - preds.unsqueeze(1)        # [N, N]
        y_diff = labels.unsqueeze(0) - labels.unsqueeze(1)      # [N, N]
        y_sign = th.sign(y_diff)
        pair_valid = (
            valid_stock.unsqueeze(0)
            & valid_stock.unsqueeze(1)
            & (y_sign != 0)
        )
        loss = F.softplus(-y_sign * p_diff)
        if pair_valid.sum() == 0:
            return preds.sum() * 0.0
        return loss[pair_valid].mean()


LOSS_DICT = {
    'mse': nn.MSELoss(reduction='mean'),
    'mll': MonotonicLogisticLoss(),
    'rmse': RMSELoss(),
    'gpl': GraphProximityLoss(),
    'ic': ICLoss(),
    'pairwiserank': PairwiseRankLoss(),
}