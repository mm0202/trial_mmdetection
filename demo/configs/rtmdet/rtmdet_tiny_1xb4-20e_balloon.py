  # balloon用の設定ファイル内容（文字列）
# この設定をベースに一部を継承・上書き
_base_ = './rtmdet_tiny_8xb32-300e_coco.py'

data_root = 'data/balloon/' # データセットのルート

train_batch_size_per_gpu = 4  # 1GPUあたりのバッチサイズ
train_num_workers = 2  # データローダのワーカ数

max_epochs = 20  # 最大エポック数
stage2_num_epochs = 1  # ステージ2のエポック数
base_lr = 0.00008  # 基本学習率


metainfo = {  # データセットのメタ情報（クラス・色）
    'classes': ('balloon', ),
    'palette': [
        (220, 20, 60),
    ]
}

train_dataloader = dict(  # 訓練データローダ設定
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        data_prefix=dict(img='train/'),
        ann_file='train.json'))

val_dataloader = dict(  # 検証データローダ設定
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        data_prefix=dict(img='val/'),
        ann_file='val.json'))

test_dataloader = val_dataloader  # テストデータローダは検証と同じ

val_evaluator = dict(ann_file=data_root + 'val.json')  # 検証用アノテーションファイル指定

test_evaluator = val_evaluator  # テスト評価は検証と同じ

model = dict(bbox_head=dict(num_classes=1))  # モデルのクラス数を1に設定

# 学習率スケジューラ設定
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0e-5,
        by_epoch=False,
        begin=0,
        end=10),
    dict(
        # 10～20エポックはコサインlrを使用
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.05,
        begin=max_epochs // 2,
        end=max_epochs,
        T_max=max_epochs // 2,
        by_epoch=True,
        convert_to_iter_based=True),
]

train_pipeline_stage2 = [  # ステージ2用のデータ前処理パイプライン
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='RandomResize',
        scale=(640, 640),
        ratio_range=(0.1, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=(640, 640)),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Pad', size=(640, 640), pad_val=dict(img=(114, 114, 114))),
    dict(type='PackDetInputs')
]

# optimizer設定
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.05),
    paramwise_cfg=dict(
        norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True))

default_hooks = dict(
    checkpoint=dict(
        interval=5,
        max_keep_ckpts=2,  # 最新2つのチェックポイントのみ保持
        save_best='auto'
    ),
    logger=dict(type='LoggerHook', interval=5))

custom_hooks = [
    dict(
        type='PipelineSwitchHook',
        switch_epoch=max_epochs - stage2_num_epochs,
        switch_pipeline=train_pipeline_stage2)
]

# COCO事前学習済み重みをロード
load_from = './checkpoints/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth'

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)  # 学習ループ設定
visualizer = dict(vis_backends=[dict(type='LocalVisBackend'),dict(type='TensorboardVisBackend')])  # 可視化設定
