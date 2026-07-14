import torch as th
import torch.nn as nn
from pathlib import Path
import os
import pandas as pd
import bisect
import numpy as np

from training.trainer import BasicSelfSuperviseTrainer
from training.optimizer import EarlyStopping, OPTIMIZER_DICT, SCHEDULER_DICT
from training.loss import LOSS_DICT
from dataset import *


# ========== 改进后的 RollingTrainer（支持多冻结层+兼容model.module） ==========
class RollingSelfSuperviseTrainer(BasicSelfSuperviseTrainer):
    name = 'RollingSelfSupervise'

    def __init__(self, args, model, rolling_windows, freeze_layer_names=None):
        """
        滚动训练器：支持冻结多个指定层 + 兼容model.module
        Args:
            args: 配置参数
            model: 模型类
            rolling_windows: 滚动窗口列表，例：[(train_win, valid_win, test_win), ...]
            freeze_layer_names: 要冻结的层名称列表（默认['proj']）
        """
        super().__init__(args, model)
        self.rolling_windows = rolling_windows  # 滚动窗口列表
        self.freeze_layer_names = freeze_layer_names  # 多冻结层列表
        self.history_best_model_path = None  # 上一轮最优模型路径

    def set_savepath(self):
        """重写路径：按滚动窗口保存，避免覆盖"""
        self.base_perf_dir = Path(self.args.paths.perf_path) / self.name / 'rolling'
        self.base_perf_dir.mkdir(parents=True, exist_ok=True)
        # 模型路径按窗口动态生成，初始化时暂不指定
        self.model_path = None

    def freeze_specified_layers(self):
        """
        工业级实现：冻结指定的多个层，兼容model.module
        - 容错：层不存在时抛出明确错误
        - 可视化：打印冻结的层名称+参数数量
        """
        if not self.freeze_layer_names:
            print("冻结层列表为空，跳过冻结逻辑，全量参数训练")
            return

        # 复用父类封装的方法，获取原始模型（兼容DataParallel）
        model_module = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        frozen_param_count = 0

        # 遍历所有要冻结的层
        for layer_name in self.freeze_layer_names:
            # 工业级容错：检查层是否存在
            if not hasattr(model_module, layer_name):
                raise ValueError(f"模型中未找到要冻结的层：{layer_name}（模型层列表：{dir(model_module)}）")
            # 获取目标层并冻结参数
            target_layer = getattr(model_module, layer_name)
            for param in target_layer.parameters():
                param.requires_grad = False
                frozen_param_count += 1

        print(f"已成功冻结 {len(self.freeze_layer_names)} 个层：{self.freeze_layer_names}")
        print(f"冻结参数总数：{frozen_param_count}")

    def load_history_model(self):
        """加载上一轮最优模型（兼容model.module）"""
        if self.history_best_model_path and os.path.exists(self.history_best_model_path):
            print(f"加载上一轮最优模型：{self.history_best_model_path}")
            # 复用父类封装的方法，获取原始模型
            model_module = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
            model_module.load_state_dict(th.load(self.history_best_model_path, map_location=self.device))

    def train_rolling(self):
        """核心：滚动训练主逻辑（兼容多冻结层+model.module）"""
        for idx, (train_window, valid_window, test_window) in enumerate(self.rolling_windows):
            print(f"\n********************* 滚动窗口 {idx + 1}/{len(self.rolling_windows)} *********************")
            print(f"训练窗口：{train_window[0]} ~ {train_window[1]}")
            print(f"验证窗口：{valid_window[0]} ~ {valid_window[1]}")
            print(f"测试窗口：{test_window[0]} ~ {test_window[1]}")

            # 1. 更新当前窗口的时间范围
            self.train_start, self.train_end = train_window
            self.valid_start, self.valid_end = valid_window
            self.test_start, self.test_end = test_window

            # 2. 设置当前窗口的保存路径（按窗口隔离）
            window_name = f"window{idx+1}_{train_window[0].replace('-', '')}_{train_window[1].replace('-', '')}"
            self.perf_dir = self.base_perf_dir / window_name
            self.perf_dir.mkdir(parents=True, exist_ok=True)
            self.model_path = self.perf_dir / 'best_model.pth'

            # 3. 加载上一轮最优模型（兼容model.module）
            self.load_history_model()

            # 4. 冻结指定的多个层（兼容model.module）
            if idx >= 1:
                self.freeze_specified_layers()
            else:
                print("第一个窗口：不冻结任何层，全量训练")

            # 5. 重新初始化优化器（仅优化未冻结的参数）
            self.set_optimizer()

            # 6. 重置早停器（每个窗口独立早停）
            self.early_stopping = EarlyStopping(
                patience=self.args.training.early_stop_patience,
                verbose=True,
                delta=self.args.training.early_stop_delta
            )

            # 7. 训练当前窗口（复用父类train方法）
            super().train()

            # 8. 记录当前窗口最优模型为下一轮历史模型
            self.history_best_model_path = self.model_path

            # # 9. 推理当前窗口
            infer_loader, pred_df = self.inference(best_model_path=self.model_path, date_range=test_window)

            # 10. 显存清理
            th.cuda.empty_cache()

        print("\n所有滚动窗口训练完成！")

    def get_merged_result(self):
        pred_dfs = []
        for idx, (train_win, test_win) in enumerate(self.rolling_windows):
            train_str = train_win[0].replace('-', '')
            train_end_str = train_win[1].replace('-', '')
            window_dir = self.base_perf_dir / f"window{idx+1}_{train_str}_{train_end_str}"
            alpha_files = list(window_dir.glob("alpha_*.csv"))
            pred_df = pd.read_csv(alpha_files[0], index_col=0)
            pred_dfs.append(pred_df)
        pred_df = pd.concat(pred_dfs).sort_index()

        final_s = pred_df.index[0].replace('-', '')
        final_e = pred_df.index[-1].replace('-', '')
        pred_df.to_csv(self.base_perf_dir / f"alpha_total_{final_s}_{final_e}.csv")
        return pred_df



# Mark!
def get_rolling_windows(start_dt, end_dt, train_len=5, test_len=1, rolling_gap=1):
    train_start = pd.to_datetime(start_dt)                                 # 查找第一个大于等于value的索引
    windows = []
    while True:
        test_start = train_start + pd.DateOffset(years=train_len)
        train_end = test_start - pd.Timedelta(days=1)
        test_end = test_start + pd.DateOffset(years=test_len) - pd.Timedelta(days=1)
        if test_end>pd.to_datetime(end_dt):
            break
        windows.append(
            (((train_start).strftime('%Y-%m-%d'), (train_end).strftime('%Y-%m-%d')),
             ((test_start).strftime('%Y-%m-%d'), (test_end).strftime('%Y-%m-%d')))
        )
        train_start += pd.DateOffset(years=rolling_gap)   # months,years
    splitratio = f'{train_len}y{test_len}y'
    return windows, splitratio



# ===================== 使用示例 =====================
if __name__ == '__main__':

    from models.Transformers.fingerprint__ import FingerprintModel, FingerprintArgs

    # 定义滚动窗口
    rolling_windows, splitratio = get_rolling_windows('2022-01-01', '2024-12-31')

    # 初始化滚动训练器（冻结proj+encoder两个层）
    trainer = RollingSelfSuperviseTrainer(
        FingerprintArgs,
        FingerprintModel,
        rolling_windows=rolling_windows,
        freeze_layer_names=['proj']  # 冻结多个层
    )

    # 启动滚动训练
    trainer.train_rolling()

    infer_loader, fingerprints = trainer.inference(best_model_path=trainer.history_best_model_path, date_range=('20250101','20251231'))

    dates = infer_loader.dates
    date_idx = infer_loader.valid_date_idx
    ticks = infer_loader.ticks
    tick_idx = infer_loader.valid_tick_idx
    for idx in sorted(list(set(date_idx))):
        date = dates[idx]
        mask = date_idx == idx
        tick = ticks[tick_idx[mask]]
        value = fingerprints[mask]
        df = pd.DataFrame(value, index=tick)
        df.to_parquet(f'result/alpha/fingerprint/rolling/{date}.parquet')



