import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import pandas as pd
from omegaconf import OmegaConf
import argparse

from models import *
from training.trainer import *
from training.metrics import*




def get_rolling_windows(start_dt, end_dt, train_len=8, valid_len=2, test_len=1, rolling_gap=1):
    train_start = pd.to_datetime(start_dt)                                 # 查找第一个大于等于value的索引
    windows = []
    while True:
        valid_start = train_start + pd.DateOffset(years=train_len)
        train_end = valid_start - pd.Timedelta(days=1)
        test_start = valid_start + pd.DateOffset(years=valid_len)
        valid_end = test_start - pd.Timedelta(days=1)
        test_end = test_start + pd.DateOffset(years=test_len) - pd.Timedelta(days=1)
        if test_end>pd.to_datetime(end_dt):
            break
        windows.append(
            (((train_start).strftime('%Y-%m-%d'), (train_end).strftime('%Y-%m-%d')),
             ((valid_start).strftime('%Y-%m-%d'), (valid_end).strftime('%Y-%m-%d')),
             ((test_start).strftime('%Y-%m-%d'), (test_end).strftime('%Y-%m-%d')))
        )
        train_start += pd.DateOffset(years=rolling_gap)   # months,years
    splitratio = f'{train_len}y{valid_len}y{test_len}y'
    return windows, splitratio



def run(trainer_str, model_str, window_params=None):

    ArgsClass, model = MODEL_DICT[model_str]
    args = ArgsClass()
    CLI_CONFIG = OmegaConf.from_cli()
    args.rewrite(CLI_CONFIG)

    type1 = trainer_str.split('_')[0]
    if type1 == 'rolling':
        windows, _ = get_rolling_windows(**window_params)
        trainer = TRAINER_DICT[trainer_str](args,model,windows)
        trainer.train_rolling()
        pred_df, label_df = trainer.get_merged_result()

    elif type1 == 'basic':
        trainer = TRAINER_DICT[trainer_str](args, model)
        trainer.train()
        _, pred_df, label_df = trainer.inference()

    # 以下为metric方法，可以后续修改
    pred = pred_df.values
    label = label_df.values
    rankics, ics = rankIC(pred, label), IC(pred, label)
    #alpha = cal_alpha(pred, label)
    #sharpe = cal_sharpe(alpha)
    #maxdrawdown = cal_maxdrawdown(alpha)

    plt.figure(figsize=(10, 6))
    plt.plot(pred_df.index, np.cumsum(rankics), label='test_rankics')
    plt.plot(pred_df.index, np.cumsum(ics), label='test_ics')
    plt.legend()
    plt.title('Cumulative Information Coefficient')
    plt.savefig(trainer.perf_dir / 'cumsumIC.png')
    plt.show()




# 创建解析器，只解析 --trainer 和 --model
parser = argparse.ArgumentParser()
parser.add_argument('--trainer', type=str, default='basic_supervise')
parser.add_argument('--model', type=str, default='gru')
parser.add_argument('--start_dt', type=str, default='2015-01-01')
parser.add_argument('--end_dt', type=str, default='2025-12-31')
parser.add_argument('--train_len', type=int, default=5)
parser.add_argument('--valid_len', type=int, default=2)
parser.add_argument('--test_len', type=int, default=1)
parser.add_argument('--rolling_gap', type=int, default=1)
# 使用 parse_known_args 忽略其他未知参数（这些参数会留给内部的 OmegaConf）
main_args, unknown = parser.parse_known_args()



if __name__ == '__main__':

    # 如果是滚动训练，先设置滚动区间
    window_params = None
    if 'rolling' in main_args.trainer:
        window_params = {
            'start_dt': main_args.start_dt,
            'end_dt': main_args.end_dt,
            'train_len': main_args.train_len,
            'valid_len': main_args.valid_len,
            'test_len': main_args.test_len,
            'rolling_gap': main_args.rolling_gap,
        }
    print(main_args)
    run(main_args.trainer, main_args.model, window_params=window_params)

    label = pd.read_csv('result/deltalag/rolling/label_total_20220104_20251231.csv', index_col=0)
    pred = pd.read_csv('result/deltalag/rolling/alpha_total_20220104_20251231.csv', index_col=0)

    calc_group_ret(pred, label)
    plt.show()

    rankics, ics = rankIC(pred, label), IC(pred, label)
    print(np.nanmean(rankics), np.nanmean(ics))

    plt.figure(figsize=(10, 6))
    plt.plot(pred.index, np.cumsum(rankics), label='test_rankics')
    plt.plot(pred.index, np.cumsum(ics), label='test_ics')
    plt.legend()
    plt.title('Cumulative Information Coefficient')
    plt.show()
