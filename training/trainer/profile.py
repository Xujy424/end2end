import torch as th
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import  autocast
from pathlib import Path
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import math

from training.optimizer import EarlyStopping, OPTIMIZER_DICT, SCHEDULER_DICT
from training.loss import LOSS_DICT
#from dataloader import DATASET_DICT, collate_fn


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

    def __init__(self, args, model):
        #th.autograd.set_detect_anomaly(True)
        self.args = args
        self.device = th.device(self.args.training.device if th.cuda.is_available() else 'cpu')
        self.set_seed(self.args.training.seed)

        # 1. 初始化数据加载相关
        self.data_path = self.args.paths.data_path
        self.label = self.args.model.target
        self.batch_size = self.args.training.batch_size  # 批大小（最大化GPU利用率）

        # 2. 初始化模型
        self.model = model(**self.args.model.params).to(self.device).float()
        # 多GPU支持（最大化计算效率）
        if th.cuda.device_count() > 1 and self.args.training.multi_gpu:
            self.model = nn.DataParallel(
                self.model,
                device_ids=self.args.training.available_gpu,
                output_device=self.args.training.main_gpu
            )

        # 3. 优化器/调度器
        self.set_optimizer()

        # 4. 路径设置
        self.set_savepath()

        # 5. 早停
        self.early_stopping = EarlyStopping(
            patience=self.args.training.early_stop_patience,
            verbose=True,
            delta=self.args.training.early_stop_delta
        )

        # 6. 时间范围处理
        self.train_start = self.args.training.period.train_start
        self.train_end = self.args.training.period.train_end
        self.valid_start = self.args.training.period.valid_start
        self.valid_end = self.args.training.period.valid_end
        self.test_start = self.args.training.period.test_start
        self.test_end = self.args.training.period.test_end

        # 7. 混合精度训练（提升GPU计算效率）
        self.scaler = th.cuda.amp.GradScaler() if self.args.training.amp else None

        # 8. 损失设置
        self.loss = LOSS_DICT[self.args.model.loss.name]

        # 9. Profiler 开关
        self.enable_profiler = getattr(self.args.training, 'enable_profiler', False)
        self.profiler_wait = 1
        self.profiler_warmup = 1
        self.profiler_active = 2
        self.profiler_repeat = 1

    def set_savepath(self):
        """设置模型/结果保存路径"""
        self.perf_dir = Path(self.args.paths.perf_path).expanduser() / self.args.model.name / 'basic'
        self.perf_dir.mkdir(parents=True, exist_ok=True)
        self.model_path = self.perf_dir / f'best_model.pth'

    def set_optimizer(self):
        """设置优化器/调度器"""
        # DP兼容修正：获取原始模型参数（剥离DataParallel）
        model_module = self.model.module if isinstance(self.model, nn.DataParallel) else self.model

        # 初始化基础优化器（原有逻辑不变）
        self.optimizer = OPTIMIZER_DICT[self.args.optimizer.name](
            params=model_module.parameters(),
            **self.args.optimizer.optim_params
        )

        self.scheduler = None
        if self.args.optimizer.if_lr_decay:
            # 1. 构建原有衰减调度器（ReduceLROnPlateau）
            self.scheduler = SCHEDULER_DICT[self.args.optimizer.scheduler](
                self.optimizer,
                **self.args.optimizer.sched_param
            )
            self.is_plateau_scheduler = isinstance(self.scheduler, th.optim.lr_scheduler.ReduceLROnPlateau)
            # 2. 构建Warmup调度器：LinearLR 线性从start_lr到target_lr
            if self.args.optimizer.warmup.enabled:
                 self.warmup_scheduler = SCHEDULER_DICT[self.args.optimizer.warmup.name](
                    optimizer=self.optimizer,
                    start_factor=self.args.optimizer.warmup.start_lr / self.args.optimizer.optim_params.lr,  # 起始因子=起始LR/目标LR
                    total_iters=self.args.optimizer.warmup.epoch,  # Warmup总轮数
                )

    def get_dataloader(self, start_date, end_date, shuffle=True, batch_size=None):
        """获取DataLoader（优化GPU利用率）"""
        dataset = DATASET_DICT[self.args.training.dataset.name](
            data_path=self.data_path,
            fields=self.args.model.fields,
            label=self.args.model.target,
            start_date=start_date,
            end_date=end_date,
            **self.args.training.dataset.params
        )
        return DataLoader(
            dataset,
            batch_size=batch_size if batch_size else self.batch_size,
            shuffle=shuffle,
            num_workers=6, #os.cpu_count() or 8,  # 多进程加载数据
            pin_memory=True,  # 固定内存，加速GPU传输
            drop_last=True,  # 丢弃最后不完整批次（避免维度错误）
            persistent_workers=True,  # 保持worker进程，加速迭代
            prefetch_factor=32,  # 预加载2个batch，CPU/GPU并行
            collate_fn=collate_fn,  # 自定义collate_fn，减少开销
            # pin_memory_device = self.args.training.device  # 直接固定到目标GPU
        )

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

    def train_step(self, batch):
        """单批次训练（混合精度）"""
        x, y = batch # 指纹模型无显式label，损失在forward中计算
        x = x.to(self.device, non_blocking=True)  # 非阻塞传输（提升效率）
        if th.isnan(x).sum()!=0:
            print(x)
        y = y.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)  # 优化显存

        # 混合精度前向
        if self.scaler:
            with autocast():
                y_ = self.model(x)
                if th.isnan(y_).sum() != 0:
                    print(y_)
                loss = self.loss(y_, y)
        else:
            y_ = self.model(x)
            loss = self.loss(y_, y)

        # 多卡时，DataParallel会返回各卡损失的张量，需聚合为标量
        if isinstance(self.model, nn.DataParallel):
            loss = loss.mean()  # 多卡损失取平均（或sum，根据业务）

        # 混合精度反向
        if self.scaler:
            self.scaler.scale(loss).backward()
            # 梯度裁剪（防止梯度爆炸）
            if self.args.optimizer.if_grad_norm:
                self.scaler.unscale_(self.optimizer)
                model_module = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
                nn.utils.clip_grad_norm_(model_module.parameters(), self.args.optimizer.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            if self.args.optimizer.if_grad_norm:
                model_module = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
                nn.utils.clip_grad_norm_(model_module.parameters(),self.args.optimizer.max_grad_norm)
            self.optimizer.step()

        return loss.item()

    @th.no_grad()
    def eval_step(self, batch):
        """单批次评估（无梯度）"""
        x, y = batch
        x = x.to(self.device, non_blocking=True)
        if th.isnan(x).sum()!=0:
            print(x)
        y = y.to(self.device, non_blocking=True)

        # 评估时关闭混合精度，避免精度损失
        y_ = self.model(x)
        if th.isnan(y_).sum()!=0:
            print(y_)
        loss = self.loss(y_, y)
        return loss.item()

    def step_scheduler(self, metric=None):
        """
        通用的学习率调度器step方法：自动适配所有类型的调度器
        Args:
            metric: 验证指标（仅ReduceLROnPlateau需要）
        """
        if self.args.optimizer.warmup.enabled and self.current_epoch<=self.args.optimizer.warmup.epoch:
            self.warmup_scheduler.step()  # Warmup阶段仅调LinearLR
        else:
            # 非Warmup阶段：自动判断是否传metric（Plateau需要，其他不需要）
            self.scheduler.step(metric) if self.is_plateau_scheduler else self.scheduler.step()

    def run_profiler(self, train_loader):
        print('\n[Profiler] 开始模型性能/参数/内存统计...')
        dummy_batch = next(iter(train_loader))
        activities = [ProfilerActivity.CPU]
        if th.cuda.is_available():
            activities.append(ProfilerActivity.CUDA)
        with profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
            with_flops=True,
            with_modules=True,
            schedule=th.profiler.schedule(
                wait=self.profiler_wait,
                warmup=self.profiler_warmup,
                active=self.profiler_active,
                repeat=self.profiler_repeat
            ),
            on_trace_ready=th.profiler.tensorboard_trace_handler(self.perf_dir)
        ) as prof:
            for _ in range(self.profiler_wait + self.profiler_warmup + self.profiler_active):
                self.train_step(dummy_batch)
                prof.step()
        # 打印统计结果
        print("\n===== 🔥 模型参数 & 性能 Profile 结果 =====")
        print(prof.key_averages().table(
            sort_by="cuda_memory_usage" if th.cuda.is_available() else "cpu_memory_usage", row_limit=15))
        prof.export_chrome_trace(self.perf_dir / "profiler_trace.json")
        print(f"[Profiler] 详细追踪文件已保存到: {self.perf_dir}/profiler_trace.json")
        print("===== Profiler 完成 =====\n")

    def train(self):
        """主训练逻辑"""
        epoch_train_losses = []
        epoch_valid_losses = []

        # 1. 预加载全量训练数据集（CPU侧memmap懒加载，不占内存）
        train_loader = self.get_dataloader(self.train_start, self.train_end, shuffle=True)
        valid_loader = self.get_dataloader(self.valid_start, self.valid_end, shuffle=False)  # 预加载验证集

        # 启动 Profiler
        if self.enable_profiler:
            self.run_profiler(train_loader)

        # 训练循环
        for epoch in range(self.args.training.num_epoch):
            print(f'\n===================== Epoch {epoch + 1} ======================')
            self.current_epoch = epoch + 1

            # 训练阶段：纯DataLoader迭代，逐batch推GPU
            self.model.train()
            train_loss, train_steps = 0.0, 0

            count = 0
            for batch in tqdm(train_loader, desc=f'Train Epoch {epoch + 1}'):
                # 每个batch仅推当前数据到GPU，显存只存1个batch
                loss = self.train_step(batch)
                if not loss:
                    print('Train Loss is NaN')
                train_loss += 0.0 if math.isnan(loss) else loss
                train_steps += 1

            # 计算epoch损失
            avg_train_loss = train_loss / train_steps if train_steps > 0 else 0
            epoch_train_losses.append(avg_train_loss)
            print(f'Epoch {epoch + 1} Train Loss: {avg_train_loss:.6f}')

            # 验证阶段（同样纯DataLoader，无block）
            avg_valid_loss = self.evaluate(valid_loader)
            epoch_valid_losses.append(avg_valid_loss)
            print(f'Epoch {epoch + 1} Valid Loss: {avg_valid_loss:.6f}')

            # 学习率调度
            if self.scheduler:
                self.step_scheduler(avg_valid_loss)

            # 早停
            self.early_stopping(avg_valid_loss, self.model, str(self.model_path))
            if self.early_stopping.early_stop:
                print(f'Early stopping at epoch {epoch + 1}')
                break

            if self.device.type == 'cuda':
                th.cuda.empty_cache()

        # loss curve结果保存
        self.save_loss_curve(epoch_train_losses, epoch_valid_losses)
        print('\nTraining completed!')

    @th.no_grad()
    def evaluate(self, eval_loader):
        self.model.eval()
        eval_loss, eval_steps = 0.0, 0
        for batch in tqdm(eval_loader, desc=f'Valid'):
            loss = self.eval_step(batch)
            eval_loss += loss
            eval_steps += 1
        return eval_loss / eval_steps if eval_steps > 0 else 0

    def save_loss_curve(self, train_losses, valid_losses=None):
        """保存损失曲线"""
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

    @th.no_grad()
    def inference(self, best_model_path=None, date_range=None):
        if best_model_path:
            print(f'Loading best model from {best_model_path}...')
            model_module = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
            model_module.load_state_dict(th.load(best_model_path, map_location=self.device))
        self.model.eval()

        if date_range:
            infer_loader = self.get_dataloader(date_range[0], date_range[1], shuffle=False, batch_size=1)
        else:
            infer_loader = self.get_dataloader(self.test_start, self.test_end, shuffle=False, batch_size=1)

        infer_dataset = infer_loader.dataset
        infer_dates = infer_dataset.dates
        infer_ticks = infer_dataset.ticks
        pred_df = np.full((len(infer_dates),len(infer_ticks)), np.nan)
        label_df = np.full((len(infer_dates), len(infer_ticks)), np.nan)

        with th.no_grad():
            for ix, batch in enumerate(tqdm(infer_loader, desc=f'Inference')):
                x,y = batch
                x = x.to(self.device, non_blocking=True)
                y_ = self.model(x).detach().cpu().numpy()

                date_idx, tick_indices = infer_dataset.data_map[ix]
                pred_df[date_idx, tick_indices] = y_.flatten()
                label_df[date_idx, tick_indices] = y.flatten()

        pred_df = pd.DataFrame(pred_df, index=infer_dates, columns=infer_ticks).loc[infer_dates[infer_dataset.valid_date_mask]]
        label_df = pd.DataFrame(label_df, index=infer_dates, columns=infer_ticks).loc[infer_dates[infer_dataset.valid_date_mask]]

        pred_df.to_csv(f'{self.perf_dir}/alpha_{infer_dataset.start_date}_{infer_dataset.end_date}.csv')
        label_df.to_csv(f'{self.perf_dir}/label_{infer_dataset.start_date}_{infer_dataset.end_date}.csv')

        return infer_dataset, pred_df, label_df



if __name__ == '__main__':

    from torchinfo import summary

    from models.RNNs.gru import GRU_Model
    gru = GRU_Model(input_size=32, hidden_size=128, num_layers=2, dropout=0.01)
    # gru.load_state_dict(th.load("/home/xujiayi/PycharmProjects/Models/rzh/old_framework/result/gru/basic/best_model.pth", map_location='cpu'))

    from models.Transformers.fingerprint import Fingerprint_Model, Fingerprint_Args
    fingerprint = Fingerprint_Model(Fingerprint_Args())
    #fingerprint.load_state_dict(th.load("/home/xujiayi/PycharmProjects/Models/End2End/result/fingerprint/basic/best_model.pth", map_location='cpu'))

    from torch.profiler import profile, record_function, ProfilerActivity



    x = th.randn(1, 237, 32).cuda()

    def count_params(model):
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"总参数量: {total:,}")
        print(f"可训练参数量: {trainable:,}")
        print(f"模型大小: {total * 4 / 1024 / 1024:.2f} MB (FP32)")
        return total, trainable



    print("=== Model 1: GRU ===")

    count_params(gru)
    summary(gru, input_size=(1, 237, 32))

    with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            profile_memory=True,
            record_shapes=True
    ) as prof:
        with record_function("model_infer"):
            gru(x)  # 随便喂一组数据
    print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=15))



    print("=== Model 2: Fingerprint ===")

    count_params(fingerprint)
    summary(fingerprint, input_size=(1, 237, 32))

    with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            profile_memory=True,
            record_shapes=True
    ) as prof:
        with record_function("model_infer"):
            fingerprint(x)
    print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))