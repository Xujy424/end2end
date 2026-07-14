import torch as th
import numpy as np
import torch.nn as nn
from pathlib import Path

from timm.optim import Muon as MuonBase
from lion_pytorch import Lion as LionBase
from sophia_opt import SophiaG as SophiaBase


class Muon(th.optim.Optimizer):
    """
    Muon 优化器封装。
    自动将 >=2D 参数交给 Muon，其余参数（1D 偏置等）交给 AdamW。
    接口与 torch.optim.Optimizer 完全一致。
    """
    def __init__(self, params, lr=0.02, momentum=0.95, weight_decay=0.01,
                 adamw_lr=1e-3, adamw_betas=(0.95, 0.99), adamw_weight_decay=0.1):
        muon_params, adamw_params = [], []
        for p in params:
            if not p.requires_grad:
                continue
            if p.dim() >= 2:
                muon_params.append(p)
            else:
                adamw_params.append(p)
        param_groups = [
            {'params': muon_params, 'lr': lr, 'momentum': momentum, 'weight_decay': weight_decay},
            {'params': adamw_params, 'lr': adamw_lr, 'betas': adamw_betas, 'weight_decay': adamw_weight_decay},
        ]
        self._optim = MuonBase(param_groups)

    def step(self, closure=None):
        self._optim.step(closure)

    def zero_grad(self, set_to_none=True):
        self._optim.zero_grad(set_to_none)


class Lion(LionBase):
    """直接继承，无需额外改动"""
    pass


class Sophia(SophiaBase):
    """
    继承 SophiaG，step 方法需传入 bs (batch size)。
    同时提供 hessian 更新计数器（可选），但实际更新需在训练循环中调用 update_hessian()。
    """
    def __init__(self, params, **kwargs):
        super().__init__(params, **kwargs)
        self._step_count = 0

    def step(self, closure=None, bs=None):
        if bs is None:
            raise ValueError("Sophia requires `bs` (batch size) in step().")
        super().step(closure=closure, bs=bs)
        self._step_count += 1



OPTIMIZER_DICT = {
    'adam': th.optim.Adam,
    'adamw': th.optim.AdamW,
    'muon': th.optim.Muon,
}


SCHEDULER_DICT = {
    'linearlr': th.optim.lr_scheduler.LinearLR,
    'steplr': th.optim.lr_scheduler.StepLR,               # 阶梯衰减     StepLR(optimizer, step_size=30, gamma=0.1)   每30个epoch学习率乘以0.1
    'multi_steplr': th.optim.lr_scheduler.MultiStepLR,    # 多阶段衰减   MultiStepLR(optimizer, milestones=[50, 100, 150], gamma=0.1)   在第50、100、150个epoch衰减
    'cosine': th.optim.lr_scheduler.CosineAnnealingLR,    # 余弦退火   CosineAnnealingLR(optimizer, T_max=200, eta_min=1e-6)    余弦衰减，200个epoch内从lr降到eta_min
    'reduce_lr_on_plateau': th.optim.lr_scheduler.ReduceLROnPlateau,
    'sequentiallr': th.optim.lr_scheduler.SequentialLR
}




class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, best_loss=np.inf):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = best_loss
        self.early_stop = False
        self.delta = delta

    def __call__(self, val_loss, model, path):
        if val_loss < self.best_loss - self.delta:
            self.save_checkpoint(val_loss, model, path)
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, val_loss, model, path):
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        if self.verbose:
            print(f'Validation loss decreased ({self.best_loss:.6f} --> {val_loss:.6f}).  Saving model ...')
        unwrap_model = model.module if isinstance(model, nn.DataParallel) else model
        unwrap_model = unwrap_model._orig_mod if hasattr(unwrap_model, '_orig_mod') else unwrap_model
        th.save(unwrap_model.state_dict(), path)



























'''
ReduceLROnPlateau(
    optimizer, 
    mode='min',     # 监控指标（min:损失下降, max:准确率上升）
    factor=0.1,     # 衰减系数
    patience=10,    # 容忍多少个epoch不改善
    verbose=True    # 打印衰减信息
)

# 每个epoch后：
val_loss = ...
scheduler.step(val_loss)  # 根据验证损失决定是否衰减
'''