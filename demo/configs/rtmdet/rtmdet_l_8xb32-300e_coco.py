_base_ = [
    # ベースとなる設定ファイルのリスト
    '../_base_/default_runtime.py',  # 実行時のデフォルト設定
    '../_base_/schedules/schedule_1x.py',  # 学習スケジュール設定
    '../_base_/datasets/coco_detection.py',  # COCOデータセット設定
    './rtmdet_tta.py'  # TTA（Test Time Augmentation）設定
]
model = dict(
    # モデル全体の設定
    type='RTMDet',  # モデルタイプ
    data_preprocessor=dict(
        # データ前処理の設定
        type='DetDataPreprocessor',  # 前処理クラス
        mean=[103.53, 116.28, 123.675],  # 画像の平均値
        std=[57.375, 57.12, 58.395],  # 標準偏差
        bgr_to_rgb=False,  # BGRからRGBへの変換有無
        batch_augments=None),  # バッチ単位のAugment
    backbone=dict(
        # バックボーン（特徴抽出部）の設定
        type='CSPNeXt',  # バックボーンタイプ
        arch='P5',  # アーキテクチャ
        expand_ratio=0.5,  # チャンネル拡張率
        deepen_factor=1,  # 深さ係数
        widen_factor=1,  # 幅係数
        channel_attention=True,  # チャンネルアテンション有効化
        norm_cfg=dict(type='SyncBN'),  # 正規化設定
        act_cfg=dict(type='SiLU', inplace=True)),  # 活性化関数設定
    neck=dict(
        # ネック（特徴融合部）の設定
        type='CSPNeXtPAFPN',  # ネックタイプ
        in_channels=[256, 512, 1024],  # 入力チャンネル数
        out_channels=256,  # 出力チャンネル数
        num_csp_blocks=3,  # CSPブロック数
        expand_ratio=0.5,  # 拡張率
        norm_cfg=dict(type='SyncBN'),  # 正規化設定
        act_cfg=dict(type='SiLU', inplace=True)),  # 活性化関数設定
    bbox_head=dict(
        # バウンディングボックスヘッドの設定
        type='RTMDetSepBNHead',  # ヘッドタイプ
        num_classes=80,  # クラス数
        in_channels=256,  # 入力チャンネル数
        stacked_convs=2,  # 畳み込み層数
        feat_channels=256,  # 特徴チャンネル数
        anchor_generator=dict(
            # アンカー生成器の設定
            type='MlvlPointGenerator', offset=0, strides=[8, 16, 32]),
        bbox_coder=dict(type='DistancePointBBoxCoder'),  # バウンディングボックスコーダ
        loss_cls=dict(
            # クラス分類損失の設定
            type='QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0),  # バウンディングボックス損失
        with_objectness=False,  # オブジェクトネス使用有無
        exp_on_reg=True,  # 回帰にexp使用
        share_conv=True,  # 畳み込み共有
        pred_kernel_size=1,  # 畳み込みカーネルサイズ
        norm_cfg=dict(type='SyncBN'),  # 正規化設定
        act_cfg=dict(type='SiLU', inplace=True)),  # 活性化関数設定
    train_cfg=dict(
        # 学習時の設定
        assigner=dict(type='DynamicSoftLabelAssigner', topk=13),  # アサイナー設定
        allowed_border=-1,  # 境界許容値
        pos_weight=-1,  # 正例重み
        debug=False),  # デバッグモード
    test_cfg=dict(
        # テスト時の設定
        nms_pre=30000,  # NMS前の最大数
        min_bbox_size=0,  # 最小バウンディングボックスサイズ
        score_thr=0.001,  # スコア閾値
        nms=dict(type='nms', iou_threshold=0.65),  # NMS設定
        max_per_img=300),  # 画像ごとの最大検出数
)

train_pipeline = [
    # 学習時のデータ前処理パイプライン
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),  # 画像読み込み
    dict(type='LoadAnnotations', with_bbox=True),  # アノテーション読み込み
    dict(type='CachedMosaic', img_scale=(640, 640), pad_val=114.0),  # Mosaic Augmentation
    dict(
        type='RandomResize',
        scale=(1280, 1280),
        ratio_range=(0.1, 2.0),
        keep_ratio=True),  # ランダムリサイズ
    dict(type='RandomCrop', crop_size=(640, 640)),  # ランダムクロップ
    dict(type='YOLOXHSVRandomAug'),  # HSV Augmentation
    dict(type='RandomFlip', prob=0.5),  # ランダムフリップ
    dict(type='Pad', size=(640, 640), pad_val=dict(img=(114, 114, 114))),  # パディング
    dict(
        type='CachedMixUp',
        img_scale=(640, 640),
        ratio_range=(1.0, 1.0),
        max_cached_images=20,
        pad_val=(114, 114, 114)),  # MixUp Augmentation
    dict(type='PackDetInputs')  # 入力パッキング
]

train_pipeline_stage2 = [
    # 学習後半のパイプライン（Mosaic/MixUpなし）
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),  # 画像読み込み
    dict(type='LoadAnnotations', with_bbox=True),  # アノテーション読み込み
    dict(
        type='RandomResize',
        scale=(640, 640),
        ratio_range=(0.1, 2.0),
        keep_ratio=True),  # ランダムリサイズ
    dict(type='RandomCrop', crop_size=(640, 640)),  # ランダムクロップ
    dict(type='YOLOXHSVRandomAug'),  # HSV Augmentation
    dict(type='RandomFlip', prob=0.5),  # ランダムフリップ
    dict(type='Pad', size=(640, 640), pad_val=dict(img=(114, 114, 114))),  # パディング
    dict(type='PackDetInputs')  # 入力パッキング
]

test_pipeline = [
    # テスト時のデータ前処理パイプライン
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),  # 画像読み込み
    dict(type='Resize', scale=(640, 640), keep_ratio=True),  # リサイズ
    dict(type='Pad', size=(640, 640), pad_val=dict(img=(114, 114, 114))),  # パディング
    dict(type='LoadAnnotations', with_bbox=True),  # アノテーション読み込み
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))  # メタ情報パッキング
]

train_dataloader = dict(
    # 学習用データローダー設定
    batch_size=32,  # バッチサイズ
    num_workers=10,  # ワーカースレッド数
    batch_sampler=None,  # バッチサンプラー
    pin_memory=True,  # ピンメモリ使用
    dataset=dict(pipeline=train_pipeline))  # 使用パイプライン
val_dataloader = dict(
    # 検証用データローダー設定
    batch_size=5, num_workers=10, dataset=dict(pipeline=test_pipeline))
test_dataloader = val_dataloader  # テスト用データローダーは検証と同じ

max_epochs = 300  # 最大エポック数
stage2_num_epochs = 20  # ステージ2のエポック数
base_lr = 0.004  # 基本学習率
interval = 10  # チェックポイント間隔

train_cfg = dict(
    # 学習設定
    max_epochs=max_epochs,  # 最大エポック数
    val_interval=interval,  # 検証間隔
    dynamic_intervals=[(max_epochs - stage2_num_epochs, 1)])  # 動的検証間隔

val_evaluator = dict(proposal_nums=(100, 1, 10))  # 検証評価指標
test_evaluator = val_evaluator  # テスト評価指標

# optimizer
optim_wrapper = dict(
    # オプティマイザ設定
    _delete_=True,  # 既存設定削除
    type='OptimWrapper',  # ラッパータイプ
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.05),  # AdamWオプティマイザ
    paramwise_cfg=dict(
        norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True))  # パラメータごとの設定

# learning rate
param_scheduler = [
    # 学習率スケジューラ設定
    dict(
        type='LinearLR',
        start_factor=1.0e-5,
        by_epoch=False,
        begin=0,
        end=1000),  # 線形ウォームアップ
    dict(
        # use cosine lr from 150 to 300 epoch
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.05,
        begin=max_epochs // 2,
        end=max_epochs,
        T_max=max_epochs // 2,
        by_epoch=True,
        convert_to_iter_based=True),  # コサインアニーリング
]

# hooks
default_hooks = dict(
    # デフォルトフック設定
    checkpoint=dict(
        interval=interval,  # チェックポイント間隔
        max_keep_ckpts=3  # only keep latest 3 checkpoints
    ))
custom_hooks = [
    # カスタムフック設定
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        priority=49),  # EMAフック
    dict(
        type='PipelineSwitchHook',
        switch_epoch=max_epochs - stage2_num_epochs,
        switch_pipeline=train_pipeline_stage2)  # パイプライン切替フック
]
