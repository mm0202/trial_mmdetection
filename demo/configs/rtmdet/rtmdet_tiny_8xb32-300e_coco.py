_base_ = './rtmdet_s_8xb32-300e_coco.py'  # ベースとなる設定ファイルを指定

checkpoint = 'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-tiny_imagenet_600e.pth'  # noqa  # 事前学習済みモデルのURL

model = dict(  # モデルの設定
    backbone=dict(  # バックボーンの設定
        deepen_factor=0.167,  # ネットワークの深さを調整
        widen_factor=0.375,  # ネットワークの幅を調整
        init_cfg=dict(  # 初期化設定
            type='Pretrained', prefix='backbone.', checkpoint=checkpoint)),  # 事前学習済み重みを利用
    neck=dict(in_channels=[96, 192, 384], out_channels=96, num_csp_blocks=1),  # ネック部分の設定
    bbox_head=dict(in_channels=96, feat_channels=96, exp_on_reg=False))  # バウンディングボックスヘッドの設定

train_pipeline = [  # 学習時の前処理パイプライン
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),  # 画像ファイルの読み込み
    dict(type='LoadAnnotations', with_bbox=True),  # アノテーション（バウンディングボックス）の読み込み
    dict(
        type='CachedMosaic',  # モザイク画像生成
        img_scale=(640, 640),  # 画像サイズ
        pad_val=114.0,  # パディング値
        max_cached_images=20,  # キャッシュする画像枚数
        random_pop=False),  # ランダムに画像を取り出すかどうか
    dict(
        type='RandomResize',  # ランダムリサイズ
        scale=(1280, 1280),  # 最大スケール
        ratio_range=(0.5, 2.0),  # 拡大・縮小の範囲
        keep_ratio=True),  # アスペクト比を保持
    dict(type='RandomCrop', crop_size=(640, 640)),  # ランダムクロップ
    dict(type='YOLOXHSVRandomAug'),  # HSV色空間でのランダムオーグメンテーション
    dict(type='RandomFlip', prob=0.5),  # ランダムフリップ（左右反転）
    dict(type='Pad', size=(640, 640), pad_val=dict(img=(114, 114, 114))),  # パディング
    dict(
        type='CachedMixUp',  # MixUp画像生成
        img_scale=(640, 640),  # 画像サイズ
        ratio_range=(1.0, 1.0),  # 比率範囲
        max_cached_images=10,  # キャッシュする画像枚数
        random_pop=False,  # ランダムに画像を取り出すかどうか
        pad_val=(114, 114, 114),  # パディング値
        prob=0.5),  # MixUpの適用確率
    dict(type='PackDetInputs')  # 入力データのパッキング
]

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))  # dataloaderのパイプライン設定
