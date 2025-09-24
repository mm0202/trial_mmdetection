tta_model = dict(  # TTA（Test Time Augmentation）用のモデル設定を辞書で定義
    type='DetTTAModel',  # モデルタイプを指定（TTA用の検出モデル）
    tta_cfg=dict(  # TTAの設定を辞書で定義
        nms=dict(type='nms', iou_threshold=0.6),  # NMS（非最大抑制）の設定。IoU閾値0.6
        max_per_img=100  # 画像ごとの最大検出数を100に設定
    )
)

img_scales = [(640, 640), (320, 320), (960, 960)]  # TTAで使用する画像サイズのリスト

tta_pipeline = [  # TTA用の前処理パイプラインをリストで定義
    dict(type='LoadImageFromFile', backend_args=None),  # 画像ファイルを読み込む処理
    dict(
        type='TestTimeAug',  # テスト時のAugmentationを適用する処理
        transforms=[
            [
                dict(type='Resize', scale=s, keep_ratio=True)  # 画像サイズごとにリサイズ（アスペクト比維持）
                for s in img_scales  # img_scalesの各サイズに対してリサイズ処理を作成
            ],
            [
                # ``RandomFlip`` must be placed before ``Pad``, otherwise
                # bounding box coordinates after flipping cannot be
                # recovered correctly.
                dict(type='RandomFlip', prob=1.),  # 画像を必ず左右反転する処理
                dict(type='RandomFlip', prob=0.)  # 画像を反転しない処理
            ],
            [
                dict(
                    type='Pad',  # 画像を指定サイズにパディングする処理
                    size=(960, 960),  # パディング後の画像サイズ
                    pad_val=dict(img=(114, 114, 114))  # パディング部分の画素値（RGBで114）
                ),
            ],
            [dict(type='LoadAnnotations', with_bbox=True)],  # アノテーション（バウンディングボックス）を読み込む処理
            [
                dict(
                    type='PackDetInputs',  # モデル入力用にデータをパックする処理
                    meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                               'scale_factor', 'flip', 'flip_direction')  # 必要なメタデータのキーを指定
                )
            ]
        ]
    )
]
