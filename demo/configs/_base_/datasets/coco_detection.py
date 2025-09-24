# dataset settings  # データセットの設定
dataset_type = 'CocoDataset'  # 使用するデータセットの種類（COCO形式）
data_root = 'data/coco/'  # データセットのルートディレクトリ

# Example to use different file client  # 別のファイルクライアントを使う例
# Method 1: simply set the data root and let the file I/O module  # 方法1: data_rootを設定し、ファイルI/Oモジュールに自動判別させる
# automatically infer from prefix (not support LMDB and Memcache yet)  # プレフィックスから自動判別（LMDBとMemcacheは未対応）

# data_root = 's3://openmmlab/datasets/detection/coco/'  # S3上のデータセットを使う例

# Method 2: Use `backend_args`, `file_client_args` in versions before 3.0.0rc6  # 方法2: backend_argsやfile_client_argsを使う（v3.0.0rc6以前）
# backend_args = dict(  # backend_argsの例
#     backend='petrel',  # バックエンドにpetrelを指定
#     path_mapping=dict({  # パスのマッピング設定
#         './data/': 's3://openmmlab/datasets/detection/',  # ローカル→S3のマッピング
#         'data/': 's3://openmmlab/datasets/detection/'  # ローカル→S3のマッピング
#     }))
backend_args = None  # バックエンド引数は未設定

train_pipeline = [  # 学習時の前処理パイプライン
    dict(type='LoadImageFromFile', backend_args=backend_args),  # 画像ファイルの読み込み
    dict(type='LoadAnnotations', with_bbox=True),  # アノテーション（バウンディングボックス）読み込み
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),  # 画像サイズ変更（比率維持）
    dict(type='RandomFlip', prob=0.5),  # ランダム反転（50%の確率）
    dict(type='PackDetInputs')  # 入力データのパッキング
]
test_pipeline = [  # テスト時の前処理パイプライン
    dict(type='LoadImageFromFile', backend_args=backend_args),  # 画像ファイルの読み込み
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),  # 画像サイズ変更（比率維持）
    # If you don't have a gt annotation, delete the pipeline  # gtアノテーションが無い場合はこの行を削除
    dict(type='LoadAnnotations', with_bbox=True),  # アノテーション（バウンディングボックス）読み込み
    dict(
        type='PackDetInputs',  # 入力データのパッキング
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))  # メタ情報のキー指定
]
train_dataloader = dict(  # 学習用データローダーの設定
    batch_size=2,  # バッチサイズ
    num_workers=2,  # ワーカープロセス数
    persistent_workers=True,  # ワーカーの永続化
    sampler=dict(type='DefaultSampler', shuffle=True),  # サンプラー（シャッフルあり）
    batch_sampler=dict(type='AspectRatioBatchSampler'),  # アスペクト比バッチサンプラー
    dataset=dict(
        type=dataset_type,  # データセットタイプ
        data_root=data_root,  # データセットルート
        ann_file='annotations/instances_train2017.json',  # アノテーションファイル（学習用）
        data_prefix=dict(img='train2017/'),  # 画像ファイルのプレフィックス
        filter_cfg=dict(filter_empty_gt=True, min_size=32),  # 空gt除外・最小サイズ指定
        pipeline=train_pipeline,  # 前処理パイプライン
        backend_args=backend_args))  # バックエンド引数
val_dataloader = dict(  # 検証用データローダーの設定
    batch_size=1,  # バッチサイズ
    num_workers=2,  # ワーカープロセス数
    persistent_workers=True,  # ワーカーの永続化
    drop_last=False,  # 最後のバッチをドロップしない
    sampler=dict(type='DefaultSampler', shuffle=False),  # サンプラー（シャッフルなし）
    dataset=dict(
        type=dataset_type,  # データセットタイプ
        data_root=data_root,  # データセットルート
        ann_file='annotations/instances_val2017.json',  # アノテーションファイル（検証用）
        data_prefix=dict(img='val2017/'),  # 画像ファイルのプレフィックス
        test_mode=True,  # テストモード
        pipeline=test_pipeline,  # 前処理パイプライン
        backend_args=backend_args))  # バックエンド引数
test_dataloader = val_dataloader  # テスト用データローダーは検証用と同じ設定

val_evaluator = dict(  # 検証用評価指標の設定
    type='CocoMetric',  # COCO形式の評価
    ann_file=data_root + 'annotations/instances_val2017.json',  # アノテーションファイル（検証用）
    metric='bbox',  # 評価指標（バウンディングボックス）
    format_only=False,  # フォーマットのみかどうか
    backend_args=backend_args)  # バックエンド引数
test_evaluator = val_evaluator  # テスト用評価指標は検証用と同じ設定

# inference on test dataset and  # テストデータセットで推論し
# format the output results for submission.  # 提出用に結果をフォーマットする例
# test_dataloader = dict(  # テスト用データローダーの例
#     batch_size=1,  # バッチサイズ
#     num_workers=2,  # ワーカープロセス数
#     persistent_workers=True,  # ワーカーの永続化
#     drop_last=False,  # 最後のバッチをドロップしない
#     sampler=dict(type='DefaultSampler', shuffle=False),  # サンプラー（シャッフルなし）
#     dataset=dict(
#         type=dataset_type,  # データセットタイプ
#         data_root=data_root,  # データセットルート
#         ann_file=data_root + 'annotations/image_info_test-dev2017.json',  # アノテーションファイル（テスト用）
#         data_prefix=dict(img='test2017/'),  # 画像ファイルのプレフィックス
#         test_mode=True,  # テストモード
#         pipeline=test_pipeline))  # 前処理パイプライン
# test_evaluator = dict(  # テスト用評価指標の例
#     type='CocoMetric',  # COCO形式の評価
#     metric='bbox',  # 評価指標（バウンディングボックス）
#     format_only=True,  # フォーマットのみ
#     ann_file=data_root + 'annotations/image_info_test-dev2017.json',  # アノテーションファイル（テスト用）
#     outfile_prefix='./work_dirs/coco_detection/test')  # 結果出力先
