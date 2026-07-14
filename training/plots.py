from training.metrics import rankIC, IC, calc_group_ret
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd




def plot_cumsumIC(pred_df, label_df, name, perf_dir):
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
    save_path = perf_dir / f'{name}_cumsumIC.png'
    plt.savefig(save_path)
    plt.show()


def plot_group_ret(pred_df, label_df, name, perf_dir):
    pred = pred_df.values
    label = label_df.values
    rankics, ics = rankIC(pred, label), IC(pred, label)
    print(np.nanmean(rankics), np.nanmean(ics))
    calc_group_ret(pred_df, label_df)
    plt.title(f'mean_RankIC:{np.mean(rankIC(pred, label)):.3%},  mean_IC:{np.mean(IC(pred, label)):.3%}')
    plt.savefig(perf_dir / f'{name}_GroupRet.png')
    plt.show()


def plot_pred(pred_df, label_df):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.hist(pred_df.values.flatten(), bins=50, alpha=0.7, color='blue')
    plt.title('RankIC Distribution')
    plt.xlabel('RankIC')
    plt.ylabel('Frequency')

    plt.subplot(1, 2, 2)
    plt.hist(label_df.values.flatten(), bins=50, alpha=0.7, color='orange')
    plt.title('IC Distribution')
    plt.xlabel('IC')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()


def plot_ic_distribution(pred_df, label_df):
    pred = pred_df.values
    label = label_df.values
    rankics, ics = rankIC(pred, label), IC(pred, label)
    print(f"平均RankIC: {np.nanmean(rankics):.4f}, 平均IC: {np.nanmean(ics):.4f}")

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.hist(rankics, bins=50, alpha=0.7, color='blue')
    plt.title('RankIC Distribution')
    plt.xlabel('RankIC')
    plt.ylabel('Frequency')

    plt.subplot(1, 2, 2)
    plt.hist(ics, bins=50, alpha=0.7, color='orange')
    plt.title('IC Distribution')
    plt.xlabel('IC')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()