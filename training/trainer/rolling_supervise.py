from training.trainer.basic_supervise import BasicSuperviseTrainer
from training.optimizer import *
from training.metrics import *

from pathlib import Path
import os
import torch as th
import shutil
import pandas as pd
import matplotlib.pyplot as plt



class RollingSuperviseTrainer(BasicSuperviseTrainer):
    name = 'RollingSelfSupervise'

    def __init__(self, args, model, rolling_windows, special_loss=None, freeze_layer_names=None):
        """
        滚动训练器：支持冻结多个指定层 + 兼容model.module
        Args:
            args: 配置参数
            model: 模型类
            rolling_windows: 滚动窗口列表，例：[(train_win, valid_win, test_win), ...]
            freeze_layer_names: 要冻结的层名称列表（默认['proj']）
        """
        super().__init__(args, model, special_loss)
        self.rolling_windows = rolling_windows  # 滚动窗口列表
        self.freeze_layer_names = freeze_layer_names  # 多冻结层列表
        self.history_best_model_path = None  # 上一轮最优模型路径
        self.current_best_loss = np.inf

        self.set_savepath()

    def set_savepath(self):
        """重写路径：按滚动窗口保存，避免覆盖"""
        self.perf_dir = Path(self.args.training.perf_path).expanduser() / self.args.model.name / 'rolling'
        self.perf_dir.mkdir(parents=True, exist_ok=True)
        # 模型路径按窗口动态生成，初始化时暂不指定
        self.model_path = str(self.perf_dir / 'best_model.pth')

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
        if self.model_path and os.path.exists(self.model_path):
            print(f"加载上一轮最优模型：{self.model_path}")
            model_module = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
            model_module.load_state_dict(th.load(self.model_path, map_location=self.device))

    def train(self):
        """核心：滚动训练主逻辑（兼容多冻结层+model.module）"""
        # super().set_seed(self.args.training.seed)  # 每个窗口重置随机种子，确保可复现
        test_preds, test_labels, test_dates = [], [], []

        for idx, (train_window, valid_window, test_window) in enumerate(self.rolling_windows):
            print(f"\n********************* 滚动窗口 {idx + 1}/{len(self.rolling_windows)} *********************")
            print(f"训练窗口：{train_window[0]} ~ {train_window[1]}")
            print(f"验证窗口：{valid_window[0]} ~ {valid_window[1]}")
            print(f"测试窗口：{test_window[0]} ~ {test_window[1]}")

            # 1. 更新当前窗口的时间范围
            self.train_start, self.train_end = train_window
            self.valid_start, self.valid_end = valid_window
            self.test_start, self.test_end = test_window
            
            if idx >= 1:
                self.load_history_model()

                self.freeze_specified_layers()

                valid_loader = self.get_dataloader(self.valid_start, self.valid_end, shuffle=False)
                self.current_best_loss = self.evaluate(valid_loader)
                print('过去最优模型在当前验证集上的损失：{:.6f}'.format(self.current_best_loss))
            else:
                print("第一个窗口：不冻结任何层，全量训练")

            self.early_stopping = EarlyStopping(
                patience=self.args.training.early_stop_patience,
                verbose=True,
                delta=self.args.training.early_stop_delta,
                best_loss=self.current_best_loss
            )

            # 5. 重新初始化优化器（仅优化未冻结的参数）
            self.set_optimizer()

            # 7. 训练当前窗口（复用父类train方法）
            super().train(save_loss=False)

            # 9. 推理当前窗口（使用已保存的历史最优模型或内存模型）
            infer_dataset, pred_df, label_df = self.inference(best_model_path=self.model_path, date_range=test_window, save=False)
            test_preds.append(pred_df.values)
            test_labels.append(label_df.values)
            test_dates.extend(pred_df.index)
        
        pred = np.concatenate(test_preds, axis=0)
        label = np.concatenate(test_labels, axis=0)
        dates = np.array(test_dates)
        self.pred_df = pd.DataFrame(pred, index=dates, columns=infer_dataset.ticks)
        self.label_df = pd.DataFrame(label, index=dates, columns=infer_dataset.ticks)

        date_tag = f"{dates[0].replace('-','')}_{dates[-1].replace('-','')}"
        self.pred_df.to_csv(f'{self.perf_dir}/alpha_merge_{date_tag}.csv')
        self.label_df.to_csv(f'{self.perf_dir}/label_merge_{date_tag}.csv')

        print("\n所有滚动窗口训练完成！")
        return self.pred_df, self.label_df
    
    def inference(self, best_model_path=None, date_range=None, perf_dir=None, save=False):
        best_model_path = best_model_path if best_model_path else self.model_path
        perf_dir = perf_dir if perf_dir else self.perf_dir
        date_range = date_range if date_range else (self.rolling_windows[0][2][0], self.rolling_windows[-1][2][1])
        return super().inference(best_model_path=best_model_path, date_range=date_range, perf_dir=perf_dir, save=save)







