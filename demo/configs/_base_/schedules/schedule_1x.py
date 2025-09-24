# 1xの学習スケジュール設定
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=12, val_interval=1)  # エポックベースの学習ループ、最大12エポック、1エポックごとに検証
val_cfg = dict(type='ValLoop')  # 検証ループの設定
test_cfg = dict(type='TestLoop')  # テストループの設定

# 学習率スケジューラの設定
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),  # 最初の500イテレーションで線形に学習率を増加
    dict(
        type='MultiStepLR',
        begin=0,
        end=12,
        by_epoch=True,
        milestones=[8, 11],
        gamma=0.1)  # 8, 11エポックで学習率を0.1倍に減少
]

# オプティマイザの設定
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001))  # SGDオプティマイザ、学習率0.02、モメンタム0.9、重み減衰0.0001

# 学習率の自動スケーリング設定
#   - `enable`がTrueの場合、バッチサイズに応じて学習率を自動調整
#   - `base_batch_size`は(8GPU) x (2サンプル/GPU)を想定
auto_scale_lr = dict(enable=False, base_batch_size=16)  # デフォルトでは自動スケーリング無効、基準バッチサイズ16
auto_scale_lr = dict(enable=False, base_batch_size=16)
