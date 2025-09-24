_base_ = './rtmdet_l_8xb32-300e_coco.py'  # ベースとなる設定ファイルを指定
checkpoint = 'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-s_imagenet_600e.pth'  # noqa  # 学習済みモデルのチェックポイントURL
model = dict(  # モデルの設定
    backbone=dict(  # バックボーンの設定
        deepen_factor=0.33,  # ネットワークの深さ係数
        widen_factor=0.5,  # ネットワークの幅係数
        init_cfg=dict(  # 初期化設定
            type='Pretrained', prefix='backbone.', checkpoint=checkpoint)),  # 事前学習済み重みを利用
    neck=dict(in_channels=[128, 256, 512], out_channels=128, num_csp_blocks=1),  # ネック部分の設定
    bbox_head=dict(in_channels=128, feat_channels=128, exp_on_reg=False))  # バウンディングボックスヘッドの設定

train_pipeline = [  # 学習時のデータ前処理パイプライン
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),  # 画像ファイルの読み込み
    dict(type='LoadAnnotations', with_bbox=True),  # アノテーション（バウンディングボックス）読み込み
    dict(type='CachedMosaic', img_scale=(640, 640), pad_val=114.0),  # モザイク画像生成
    dict(
        type='RandomResize',
        scale=(1280, 1280),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),  # ランダムリサイズ
    dict(type='RandomCrop', crop_size=(640, 640)),  # ランダムクロップ
    dict(type='YOLOXHSVRandomAug'),  # HSV色空間でのランダム拡張
    dict(type='RandomFlip', prob=0.5),  # ランダム反転
    dict(type='Pad', size=(640, 640), pad_val=dict(img=(114, 114, 114))),  # パディング
    dict(
        type='CachedMixUp',
        img_scale=(640, 640),
        ratio_range=(1.0, 1.0),
        max_cached_images=20,
        pad_val=(114, 114, 114)),  # MixUp拡張
    dict(type='PackDetInputs')  # 入力データのパッキング
]

train_pipeline_stage2 = [  # 学習後半用のデータ前処理パイプライン
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),  # 画像ファイルの読み込み
    dict(type='LoadAnnotations', with_bbox=True),  # アノテーション読み込み
    dict(
        type='RandomResize',
        scale=(640, 640),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),  # ランダムリサイズ
    dict(type='RandomCrop', crop_size=(640, 640)),  # ランダムクロップ
    dict(type='YOLOXHSVRandomAug'),  # HSV色空間でのランダム拡張
    dict(type='RandomFlip', prob=0.5),  # ランダム反転
    dict(type='Pad', size=(640, 640), pad_val=dict(img=(114, 114, 114))),  # パディング
    dict(type='PackDetInputs')  # 入力データのパッキング
]

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))  # データローダーのパイプライン設定

custom_hooks = [  # カスタムフックの設定
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        priority=49),  # EMA（指数移動平均）フック
    dict(
        type='PipelineSwitchHook',
        switch_epoch=280,
        switch_pipeline=train_pipeline_stage2)  # エポック280でパイプラインを切り替えるフック
]
