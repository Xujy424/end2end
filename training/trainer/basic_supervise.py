import torch as th
import torch.nn as nn
from networkx.algorithms.distance_measures import periphery
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from pathlib import Path
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import math
import sys
import gc
from collections.abc import Mapping, Sequence

from training.optimizer import EarlyStopping, OPTIMIZER_DICT, SCHEDULER_DICT
from training.loss import LOSS_DICT
from dataset import *
from training.metrics import rankIC, IC, calc_group_ret
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# 原有显存优化配置
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True, max_split_size_mb:128' # 限制显存分片，减少碎片
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # 关闭全量同步，避免卡死
os.environ['TORCH_USE_CUDA_DSA'] = '1'
th.backends.cudnn.benchmark = False
th.backends.cudnn.deterministic = True
# 确认多卡可用，获取GPU列表
TARGET_MAIN_GPU = 4
EXCLUDE_GPUS = [0,1,2,3,6,7]
ALL_AVAILABLE_GPUS = list(range(th.cuda.device_count()))
VALID_GPUS = [gpu_id for gpu_id in ALL_AVAILABLE_GPUS if gpu_id not in EXCLUDE_GPUS]
DEVICE_LIST = [th.device(f'cuda:{TARGET_MAIN_GPU}')] + \
              [th.device(f'cuda:{gpu_id}') for gpu_id in VALID_GPUS if gpu_id != TARGET_MAIN_GPU]
MAIN_DEVICE = DEVICE_LIST[0]  # 主GPU（DP默认使用cuda:0作为主卡，聚合结果）



class BasicSuperviseTrainer:
    name = 'BasicSelfSupervise'

    def __init__(self, args, model, special_loss=None):
        #th.autograd.set_detect_anomaly(True)
        self.args = args
        self.device = th.device(self.args.training.device if th.cuda.is_available() else 'cpu')
        self.set_seed(self.args.training.seed)

        self.batch_size = self.args.training.batch_size  # 批大小（最大化GPU利用率）
        self.accumulation_steps = self.args.optimizer.accumulation_steps
        self.use_grad_accumulation = self.accumulation_steps is not None and self.accumulation_steps > 1

        self.model = model(**self.args.model.params).to(self.device).float()
        if th.cuda.device_count() > 1 and self.args.training.multi_gpu:
            self.model = nn.DataParallel(
                self.model,
                device_ids=self.args.training.available_gpu,
                output_device=self.args.training.main_gpu
            )

        self.set_optimizer()

        self.set_savepath()

        self.early_stopping = EarlyStopping(
            patience=self.args.training.early_stop_patience,
            verbose=True,
            delta=self.args.training.early_stop_delta
        )

        self.train_start = self.args.training.period.train_start
        self.train_end = self.args.training.period.train_end
        self.valid_start = self.args.training.period.valid_start
        self.valid_end = self.args.training.period.valid_end
        self.test_start = self.args.training.period.test_start
        self.test_end = self.args.training.period.test_end

        self.scaler = th.cuda.amp.GradScaler() if self.args.training.amp else None

        self.loss = special_loss if special_loss else LOSS_DICT[self.args.model.loss.name]

    def set_savepath(self):
        """设置模型/结果保存路径"""
        self.perf_dir = Path(self.args.training.perf_path).expanduser() / self.args.model.name / 'basic'
        self.perf_dir.mkdir(parents=True, exist_ok=True)
        self.model_path = self.perf_dir / f'best_model.pth'

    def set_seed(self, seed=42):
        """设置随机种子"""
        random.seed(seed)
        np.random.seed(seed)
        th.manual_seed(seed)
        th.cuda.manual_seed(seed)
        th.cuda.manual_seed_all(seed)
        th.backends.cudnn.deterministic = self.args.training.deterministic
        # 开启benchmark（最大化GPU计算效率，非确定性）
        th.backends.cudnn.benchmark = not self.args.training.deterministic
        os.environ['PYTHONHASHSEED'] = str(seed)
        # 优化：设置CUDA内存分配策略，减少碎片
        if th.cuda.is_available():
            th.cuda.set_per_process_memory_fraction(0.95)
            th.backends.cuda.matmul.allow_tf32 = True  # 开启TF32，提升矩阵运算速度
            th.backends.cudnn.allow_tf32 = True

    def set_optimizer(self):
        """设置优化器/调度器"""
        # DP兼容修正：获取原始模型参数（剥离DataParallel）
        model_module = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        trainable_params = [p for p in model_module.parameters() if p.requires_grad]
        
        # 初始化基础优化器（原有逻辑不变）
        self.optimizer = OPTIMIZER_DICT[self.args.optimizer.name](
            params=trainable_params,
            **self.args.optimizer.optim_params
        )

        self.scheduler = None
        if self.args.optimizer.if_lr_decay:
            # 1. 构建原有衰减调度器（ReduceLROnPlateau）
            self.scheduler = SCHEDULER_DICT[self.args.optimizer.scheduler](
                self.optimizer,
                **self.args.optimizer.sched_params
            )
            self.is_plateau_scheduler = isinstance(self.scheduler, th.optim.lr_scheduler.ReduceLROnPlateau)
            # 2. 构建Warmup调度器：LinearLR 线性从start_lr到target_lr
            if self.args.optimizer.warmup.enabled:
                 self.warmup_scheduler = SCHEDULER_DICT[self.args.optimizer.warmup.name](
                    optimizer=self.optimizer,
                    start_factor=self.args.optimizer.warmup.start_lr / self.args.optimizer.optim_params.lr,  # 起始因子=起始LR/目标LR
                    total_iters=self.args.optimizer.warmup.epoch,  # Warmup总轮数
                )
                 
    def step_scheduler(self, metric=None):
        """
        通用的学习率调度器step方法: 自动适配所有类型的调度器
        args:
            metric: 验证指标, 仅ReduceLROnPlateau需要
        """
        if self.args.optimizer.warmup.enabled and self.current_epoch<=self.args.optimizer.warmup.epoch:
            self.warmup_scheduler.step() 
        else:
            self.scheduler.step(metric) if self.is_plateau_scheduler else self.scheduler.step()

    def get_dataloader(self, start_date, end_date, shuffle=False, batch_size=None, num_workers=6, pin_memory=True, drop_last=True, persistnet_workers=True, prefetch_factor=32):
        """获取DataLoader"""
        self.args.training.dataset.params.shared_param_dict.start_date = start_date
        self.args.training.dataset.params.shared_param_dict.end_date = end_date
        dataset = DATASET_DICT[self.args.training.dataset.name](
            # start_date=start_date,
            # end_date=end_date,
            **self.args.training.dataset.params
        )
        return DataLoader(
            dataset,
            batch_size=batch_size if batch_size else self.batch_size,
            shuffle=shuffle,
            num_workers=num_workers, #os.cpu_count() or 8,  # 多进程加载数据
            pin_memory=pin_memory,  # 固定内存，加速GPU传输
            drop_last=drop_last,  # 丢弃最后不完整批次（避免维度错误）
            persistent_workers=persistnet_workers,  # 保持worker进程，加速迭代
            prefetch_factor=prefetch_factor,  # 预加载2个batch，CPU/GPU并行
            collate_fn=multi_collate_fn,  # 自定义collate_fn，dict形式cat batch
            # pin_memory_device = self.args.training.device  # 直接固定到目标GPU
        )
    
    def push_to_gpu(self, obj, device, non_blocking=True):
        """递归地将所有 Tensor 移动到指定设备，保持原结构不变。"""
        if isinstance(obj, th.Tensor):
            return obj.to(device, non_blocking=non_blocking)
        elif isinstance(obj, Mapping):
            return {k: self.push_to_gpu(v, device, non_blocking) if k not in ['date_idx', 'tick_idxs'] else v for k, v in obj.items()}
        elif isinstance(obj, Sequence) and not isinstance(obj, str):
            return type(obj)(self.push_to_gpu(v, device, non_blocking) for v in obj) # 处理 list / tuple
        else:
            return obj

    def train_step(self, batch):
        """单批次训练（混合精度）。仅前向与 backward，优化步骤由 train() 统一控制。"""
        # x, y, _,_ = batch
        # x = tuple(seq.to(self.device, non_blocking=True) for seq in x) if isinstance(x, list) else x.to(self.device, non_blocking=True)
        # y = y.to(self.device, non_blocking=True).squeeze(0)
        batch = self.push_to_gpu(batch, self.device, non_blocking=True)
        x = batch['feats']
        y = batch['label']

        if self.scaler:
            with autocast():
                y_ = self.model(x)
                loss = self.loss(y_, y)
        else:
            y_ = self.model(x)
            loss = self.loss(y_, y)
            if th.isnan(loss):
                y_ = self.model(x)

        if isinstance(self.model, nn.DataParallel):
            loss = loss.mean()

        raw_loss = loss
        if self.use_grad_accumulation:
            loss = loss / self.accumulation_steps

        if self.scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        return raw_loss.item()
    
    def optimizer_step(self):
        """执行一次优化器更新，并支持混合精度梯度裁剪。"""
        model_module = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        if self.scaler:
            if self.args.optimizer.if_grad_norm:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(model_module.parameters(), self.args.optimizer.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            if self.args.optimizer.if_grad_norm:
                nn.utils.clip_grad_norm_(model_module.parameters(), self.args.optimizer.max_grad_norm)
            self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

    def train(self, save_loss=False):
        """主训练逻辑"""
        self.set_seed(self.args.training.seed)
        
        epoch_train_losses = []
        epoch_valid_losses = []

        train_loader = self.get_dataloader(self.train_start, self.train_end, shuffle=True)
        valid_loader = self.get_dataloader(self.valid_start, self.valid_end, shuffle=False)

        for epoch in range(self.args.training.num_epoch):
            print(f'===================== Epoch {epoch + 1} ======================')
            self.current_epoch = epoch + 1

            self.model.train()
            train_loss, train_step = 0.0, 0
            self.optimizer.zero_grad(set_to_none=True)

            for batch_idx, batch in enumerate(tqdm(train_loader, desc=f'Train Epoch {epoch + 1}')):
                loss = self.train_step(batch)
                train_loss += loss
                train_step += 1

                if (batch_idx + 1) % self.accumulation_steps==0 or batch_idx==len(train_loader)-1:
                    self.optimizer_step()

            avg_train_loss = train_loss / train_step
            epoch_train_losses.append(avg_train_loss)
            print(f'Epoch {epoch + 1} Train Loss: {avg_train_loss:.6f}')

            avg_valid_loss = self.evaluate(valid_loader)
            epoch_valid_losses.append(avg_valid_loss)
            print(f'Epoch {epoch + 1} Valid Loss: {avg_valid_loss:.6f}')

            if self.scheduler:
                self.step_scheduler(avg_valid_loss)

            self.early_stopping(avg_valid_loss, self.model, str(self.model_path))
            if self.early_stopping.early_stop:
                print(f'Early stopping at epoch {epoch + 1}')
                break

            if self.device.type == 'cuda':
                th.cuda.empty_cache()
                gc.collect()
       
        if save_loss: self.save_loss_curve(epoch_train_losses, epoch_valid_losses)
        print('Training complete!')

    @th.no_grad()
    def eval_step(self, batch):
        """单批次评估（无梯度）"""
        # x, y, _,_ = batch
        # x = tuple(seq.to(self.device, non_blocking=True) for seq in x) if isinstance(x, list) else x.to(self.device,non_blocking=True)
        # y = y.to(self.device, non_blocking=True).squeeze(0)
        batch = self.push_to_gpu(batch, self.device, non_blocking=True)
        x = batch['feats']
        y = batch['label']

        # 评估时关闭混合精度，避免精度损失
        y_ = self.model(x)
        loss = self.loss(y_, y)

        if isinstance(self.model, nn.DataParallel):
            loss = loss.mean()
        return loss.item()
    
    @th.no_grad()
    def evaluate(self, eval_loader):
        self.model.eval()
        eval_loss, eval_step = 0.0, 0
        for batch in tqdm(eval_loader, desc=f'Valid'):
            loss = self.eval_step(batch)
            eval_loss += loss
            eval_step += 1
        return eval_loss / eval_step


    @th.no_grad()
    def inference(self, best_model_path=None, date_range=None, perf_dir=None, save=False):
        model_module = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        if best_model_path:
            print(f'Loading best model from {best_model_path}...')
            model_module.load_state_dict(th.load(best_model_path, map_location=self.device))
        else:
            print(f'Loading best model from {self.model_path}...')
            model_module.load_state_dict(th.load(self.model_path, map_location=self.device))
        model_module.eval()

        if date_range:
            infer_loader = self.get_dataloader(date_range[0], date_range[1], shuffle=False, batch_size=self.args.training.batch_size)
        else:
            infer_loader = self.get_dataloader(self.test_start, self.test_end, shuffle=False, batch_size=self.args.training.batch_size)

        infer_dataset = infer_loader.dataset
        infer_dates = infer_dataset.dates
        infer_ticks = infer_dataset.ticks
        pred_df = np.full((len(infer_dates),len(infer_ticks)), np.nan)
        label_df = np.full((len(infer_dates), len(infer_ticks)), np.nan)

        with th.no_grad():
            for batch in tqdm(infer_loader, desc=f'Inference'):
                # x,y,d,t = batch
                # x = tuple(seq.to(self.device, non_blocking=True) for seq in x) if isinstance(x, list) else x.to(self.device, non_blocking=True)
                batch = self.push_to_gpu(batch, self.device, non_blocking=True)
                x = batch['feats']
                y = batch['label']
                d = batch['date_idx']
                t = batch['tick_idxs']

                y_ = model_module(x)
                y_ = y_.detach().cpu().numpy() if not isinstance(y_, tuple) else y_[0].detach().cpu().numpy()

                pred_df[d, t] = y_
                label_df[d, t] = y.cpu().numpy()

        pred_df = pd.DataFrame(pred_df, index=infer_dates, columns=infer_ticks).loc[infer_dates[infer_dataset.valid_date_mask]]
        label_df = pd.DataFrame(label_df, index=infer_dates, columns=infer_ticks).loc[infer_dates[infer_dataset.valid_date_mask]]
        date_tag = f"{pred_df.index[0].replace('-','')}_{pred_df.index[-1].replace('-','')}"
        
        if save:
            perf_dir = perf_dir if perf_dir else self.perf_dir
            pred_df.to_csv(f'{perf_dir}/alpha_{date_tag}.csv')
            label_df.to_csv(f'{perf_dir}/label_{date_tag}.csv')

        return infer_dataset, pred_df, label_df


    def save_loss_curve(self, train_losses, valid_losses=None):
            #保存损失曲线图
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss', marker='o')
            plt.plot(range(1, len(valid_losses) + 1), valid_losses, label='Valid Loss', marker='s')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training (& Validation) Loss Curve')
            plt.legend()
            plt.grid(True)
            plt.savefig(self.perf_dir / 'loss_curve.png', dpi=300, bbox_inches='tight')
            plt.close()
            # 保存损失数据
            loss_df = pd.DataFrame({
                'epoch': range(1, len(train_losses) + 1),
                'train_loss': train_losses,
                'valid_loss': valid_losses
            })
            loss_df.to_csv(self.perf_dir / 'loss_history.csv', index=False)
    
    def plot_cumsumIC(self, pred_df, label_df, name):
        pred = pred_df.values
        label = label_df.values
        rankics, ics = rankIC(pred, label), IC(pred, label)
        plt.figure(figsize=(10, 6))
        plt.plot(pd.to_datetime(pred_df.index), np.cumsum(rankics), label='test_rankics')
        plt.plot(pd.to_datetime(pred_df.index), np.cumsum(ics), label='test_ics')
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        plt.xticks(rotation=30, ha="right")
        plt.legend()
        plt.title('Cumulative Information Coefficient')
        save_path = self.perf_dir / f'{name}_cumsumIC.png'
        plt.savefig(save_path)
        plt.show()
    
    def plot_group_ret(self, pred_df, label_df, name):
        pred = pred_df.values
        label = label_df.values
        rankics, ics = rankIC(pred, label), IC(pred, label)
        print(np.nanmean(rankics), np.nanmean(ics))
        calc_group_ret(pred_df, label_df)
        plt.title(f'mean_RankIC:{np.mean(rankIC(pred, label)):.3%},  mean_IC:{np.mean(IC(pred, label)):.3%}')
        plt.savefig(self.perf_dir / f'{name}_GroupRet.png')
        plt.show()








