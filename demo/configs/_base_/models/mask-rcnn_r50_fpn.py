# モデル設定
model = dict(  # モデル全体の設定を辞書で定義
    type='MaskRCNN',  # モデルタイプはMask R-CNN
    data_preprocessor=dict(  # データ前処理の設定
        type='DetDataPreprocessor',  # 前処理クラス名
        mean=[123.675, 116.28, 103.53],  # 画像の平均値（正規化用）
        std=[58.395, 57.12, 57.375],  # 画像の標準偏差（正規化用）
        bgr_to_rgb=True,  # BGRからRGBへ変換するか
        pad_mask=True,  # マスク画像をパディングするか
        pad_size_divisor=32),  # パディングサイズの基準
    backbone=dict(  # バックボーン（特徴抽出部）の設定
        type='ResNet',  # ResNetを使用
        depth=50,  # ResNetの深さ（層数）
        num_stages=4,  # ステージ数
        out_indices=(0, 1, 2, 3),  # 出力するステージのインデックス
        frozen_stages=1,  # 固定するステージ数
        norm_cfg=dict(type='BN', requires_grad=True),  # バッチ正規化の設定
        norm_eval=True,  # 評価時に正規化層を固定するか
        style='pytorch',  # 実装スタイル
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),  # 事前学習済みモデルの指定
    neck=dict(  # ネック（特徴融合部）の設定
        type='FPN',  # FPN（Feature Pyramid Network）を使用
        in_channels=[256, 512, 1024, 2048],  # 入力チャンネル数
        out_channels=256,  # 出力チャンネル数
        num_outs=5),  # 出力数
    rpn_head=dict(  # RPN（Region Proposal Network）ヘッドの設定
        type='RPNHead',  # RPNヘッドクラス名
        in_channels=256,  # 入力チャンネル数
        feat_channels=256,  # 特徴チャンネル数
        anchor_generator=dict(  # アンカー生成器の設定
            type='AnchorGenerator',  # アンカー生成クラス名
            scales=[8],  # アンカーのスケール
            ratios=[0.5, 1.0, 2.0],  # アンカーのアスペクト比
            strides=[4, 8, 16, 32, 64]),  # アンカーのストライド
        bbox_coder=dict(  # バウンディングボックスのエンコーダ設定
            type='DeltaXYWHBBoxCoder',  # DeltaXYWH方式
            target_means=[.0, .0, .0, .0],  # ターゲット平均
            target_stds=[1.0, 1.0, 1.0, 1.0]),  # ターゲット標準偏差
        loss_cls=dict(  # 分類損失の設定
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),  # シグモイド付きクロスエントロピー
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),  # バウンディングボックス損失（L1損失）
    roi_head=dict(  # RoIヘッドの設定
        type='StandardRoIHead',  # 標準RoIヘッド
        bbox_roi_extractor=dict(  # バウンディングボックス用RoI抽出器
            type='SingleRoIExtractor',  # 単一RoI抽出器
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),  # RoIAlignレイヤー
            out_channels=256,  # 出力チャンネル数
            featmap_strides=[4, 8, 16, 32]),  # 特徴マップのストライド
        bbox_head=dict(  # バウンディングボックスヘッド
            type='Shared2FCBBoxHead',  # 2層FCヘッド
            in_channels=256,  # 入力チャンネル数
            fc_out_channels=1024,  # FC層の出力チャンネル数
            roi_feat_size=7,  # RoI特徴サイズ
            num_classes=80,  # クラス数
            bbox_coder=dict(  # バウンディングボックスエンコーダ
                type='DeltaXYWHBBoxCoder',  # DeltaXYWH方式
                target_means=[0., 0., 0., 0.],  # ターゲット平均
                target_stds=[0.1, 0.1, 0.2, 0.2]),  # ターゲット標準偏差
            reg_class_agnostic=False,  # クラス非依存回帰かどうか
            loss_cls=dict(  # 分類損失
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),  # クロスエントロピー損失
            loss_bbox=dict(type='L1Loss', loss_weight=1.0)),  # バウンディングボックス損失
        mask_roi_extractor=dict(  # マスク用RoI抽出器
            type='SingleRoIExtractor',  # 単一RoI抽出器
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),  # RoIAlignレイヤー
            out_channels=256,  # 出力チャンネル数
            featmap_strides=[4, 8, 16, 32]),  # 特徴マップのストライド
        mask_head=dict(  # マスクヘッド
            type='FCNMaskHead',  # FCNベースのマスクヘッド
            num_convs=4,  # 畳み込み層数
            in_channels=256,  # 入力チャンネル数
            conv_out_channels=256,  # 畳み込み出力チャンネル数
            num_classes=80,  # クラス数
            loss_mask=dict(  # マスク損失
                type='CrossEntropyLoss', use_mask=True, loss_weight=1.0))),  # マスク用クロスエントロピー損失
    # モデルの学習・テスト設定
    train_cfg=dict(  # 学習時の設定
        rpn=dict(  # RPNの学習設定
            assigner=dict(  # アサイナー（割り当て）の設定
                type='MaxIoUAssigner',  # 最大IoUアサイナー
                pos_iou_thr=0.7,  # 正例IoU閾値
                neg_iou_thr=0.3,  # 負例IoU閾値
                min_pos_iou=0.3,  # 最小正例IoU
                match_low_quality=True,  # 低品質マッチも許容
                ignore_iof_thr=-1),  # 無視するIoF閾値
            sampler=dict(  # サンプラーの設定
                type='RandomSampler',  # ランダムサンプラー
                num=256,  # サンプル数
                pos_fraction=0.5,  # 正例割合
                neg_pos_ub=-1,  # 負例上限
                add_gt_as_proposals=False),  # GTを提案に追加するか
            allowed_border=-1,  # 許容境界
            pos_weight=-1,  # 正例重み
            debug=False),  # デバッグモード
        rpn_proposal=dict(  # RPN提案の設定
            nms_pre=2000,  # NMS前の最大数
            max_per_img=1000,  # 画像ごとの最大提案数
            nms=dict(type='nms', iou_threshold=0.7),  # NMSの設定
            min_bbox_size=0),  # 最小バウンディングボックスサイズ
        rcnn=dict(  # RCNNの学習設定
            assigner=dict(  # アサイナー設定
                type='MaxIoUAssigner',  # 最大IoUアサイナー
                pos_iou_thr=0.5,  # 正例IoU閾値
                neg_iou_thr=0.5,  # 負例IoU閾値
                min_pos_iou=0.5,  # 最小正例IoU
                match_low_quality=True,  # 低品質マッチも許容
                ignore_iof_thr=-1),  # 無視するIoF閾値
            sampler=dict(  # サンプラー設定
                type='RandomSampler',  # ランダムサンプラー
                num=512,  # サンプル数
                pos_fraction=0.25,  # 正例割合
                neg_pos_ub=-1,  # 負例上限
                add_gt_as_proposals=True),  # GTを提案に追加するか
            mask_size=28,  # マスクサイズ
            pos_weight=-1,  # 正例重み
            debug=False)),  # デバッグモード
    test_cfg=dict(  # テスト時の設定
        rpn=dict(  # RPNのテスト設定
            nms_pre=1000,  # NMS前の最大数
            max_per_img=1000,  # 画像ごとの最大提案数
            nms=dict(type='nms', iou_threshold=0.7),  # NMSの設定
            min_bbox_size=0),  # 最小バウンディングボックスサイズ
        rcnn=dict(  # RCNNのテスト設定
            score_thr=0.05,  # スコア閾値
            nms=dict(type='nms', iou_threshold=0.5),  # NMSの設定
            max_per_img=100,  # 画像ごとの最大検出数
            mask_thr_binary=0.5)))  # マスクの2値化閾値
