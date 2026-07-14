
if __name__ == '__main__':

    data_path = '/data/xujiayi/mmap/fingerprint'

    PRICE_FEATURES = [
        'high', 'low', 'close', 'ppos'
    ]
    TRADE_FEATURES = [  # 0,4,7,11,12,13,16,20,21,22,27
        'volume', 'amount',
        'passive_count', 'passive_amount', 'passive_volume',
        'active_buy_count', 'active_buy_amount', 'active_buy_volume',
        'active_buy_floatamount_super_count', 'active_buy_floatamount_large_count', 'active_buy_floatamount_mid_count',
        'active_buy_floatamount_super_volume', 'active_buy_floatamount_large_volume',
        'active_buy_floatamount_mid_volume',
        'active_sell_count', 'active_sell_amount', 'active_sell_volume',
        'active_sell_floatamount_super_count', 'active_sell_floatamount_large_count',
        'active_sell_floatamount_mid_count',
        'active_sell_floatamount_super_volume', 'active_sell_floatamount_large_volume',
        'active_sell_floatamount_mid_volume',
        'cj_count', 'avg_cj_amount',
        'cancel_count', 'cancel_amount', 'cancel_volume'
    ]
    fields = PRICE_FEATURES + TRADE_FEATURES


    # data_path = "/data/xujiayi/end2end/"
    # fields = ['close_adj_senorm20_seczscore', 'open_adj_senorm20_seczscore', 'low_adj_senorm20_seczscore',
    #           'high_adj_senorm20_seczscore', 'volume_adj_senorm20_seczscore', 'amount_senorm20_seczscore',
    #           'turnover_senorm20_seczscore']

    dataset = BatchDataset(data_path, fields, None, '2022-01-01', '2022-11-31', freq='intraday')
    loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=6, #os.cpu_count() or 8,  # 多进程加载数据
            pin_memory=True,  # 固定内存，加速GPU传输
            drop_last=True,  # 丢弃最后不完整批次（避免维度错误）
            persistent_workers=True,  # 保持worker进程，加速迭代
            prefetch_factor=128,  # 预加载2个batch，CPU/GPU并行
            collate_fn=collate_fn,  # 自定义collate_fn，减少开销
            # pin_memory_device = self.args.training.device  # 直接固定到目标GPU
    )
    for batch in loader:
        print(batch[0].shape)
        print(batch[1].shape)