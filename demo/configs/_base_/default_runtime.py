default_scope = 'mmdet'  # デフォルトのスコープ名を'mmdet'に設定

default_hooks = dict(  # 実行時に使用する各種フックの設定
    timer=dict(type='IterTimerHook'),  # イテレーションごとのタイマー計測フック
    logger=dict(type='LoggerHook', interval=50),  # ログ出力フック（50イテレーションごと）
    param_scheduler=dict(type='ParamSchedulerHook'),  # パラメータスケジューラフック
    checkpoint=dict(type='CheckpointHook', interval=1),  # チェックポイント保存フック（1エポックごと）
    sampler_seed=dict(type='DistSamplerSeedHook'),  # 分散サンプラーのシード設定フック
    visualization=dict(type='DetVisualizationHook'))  # 可視化フック

env_cfg = dict(  # 実行環境の設定
    cudnn_benchmark=False,  # cuDNNのベンチマークモードを無効化
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),  # マルチプロセス設定（fork開始、OpenCVスレッド数0）
    dist_cfg=dict(backend='nccl'),  # 分散学習のバックエンドをNCCLに設定
)

vis_backends = [dict(type='LocalVisBackend')]  # 可視化バックエンドの設定（ローカル）
visualizer = dict(  # 可視化器の設定
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)  # ログ処理の設定（50イテレーションのウィンドウ、エポック単位）

log_level = 'INFO'  # ログレベルをINFOに設定
load_from = None  # 事前学習済みモデルのパス（未指定）
resume = False  # 学習の再開フラグ（False: 新規学習）
