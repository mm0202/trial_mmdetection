# Copyright (c) OpenMMLab. All rights reserved.
import argparse  # コマンドライン引数を解析するためのモジュールをインポート
import os  # OS関連の機能を利用するためのモジュールをインポート
import os.path as osp  # パス操作を簡単にするためのモジュールをインポート

from mmengine.config import Config, DictAction  # MMEngineの設定管理用クラスと辞書型引数アクションをインポート
from mmengine.registry import RUNNERS  # Runnerのレジストリをインポート
from mmengine.runner import Runner  # Runnerクラスをインポート

from mmdet.utils import setup_cache_size_limit_of_dynamo  # Dynamoのキャッシュサイズ制限設定関数をインポート


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')  # 引数パーサを作成し説明を追加
    parser.add_argument('config', help='train config file path')  # 設定ファイルパスの引数を追加
    parser.add_argument('--work-dir', help='the dir to save logs and models')  # ログやモデル保存先ディレクトリ指定
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic-mixed-precision training')  # AMP（自動混合精度）トレーニングの有効化
    parser.add_argument(
        '--auto-scale-lr',
        action='store_true',
        help='enable automatically scaling LR.')  # 学習率自動スケーリングの有効化
    parser.add_argument(
        '--resume',
        nargs='?',
        type=str,
        const='auto',
        help='If specify checkpoint path, resume from it, while if not '
        'specify, try to auto resume from the latest checkpoint '
        'in the work directory.')  # チェックポイントからの再開設定
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')  # 設定ファイルの一部をコマンドラインから上書きするためのオプション
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')  # 分散学習のランチャー指定
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)  # 分散学習時のローカルランク指定
    args = parser.parse_args()  # 引数を解析
    if 'LOCAL_RANK' not in os.environ:  # 環境変数にLOCAL_RANKがなければ
        os.environ['LOCAL_RANK'] = str(args.local_rank)  # 引数からLOCAL_RANKを環境変数に設定

    return args  # 解析結果を返す


def main():
    args = parse_args()  # コマンドライン引数を取得

    # Reduce the number of repeated compilations and improve
    # training speed.
    setup_cache_size_limit_of_dynamo()  # Dynamoのキャッシュサイズ制限を設定し、再コンパイル回数を減らす

    # load config
    cfg = Config.fromfile(args.config)  # 設定ファイルを読み込む
    cfg.launcher = args.launcher  # ランチャー設定をcfgに反映
    if args.cfg_options is not None:  # コマンドラインで設定上書きがあれば
        cfg.merge_from_dict(args.cfg_options)  # 設定をマージ

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:  # work_dirが指定されていれば
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir  # cfgのwork_dirを上書き
    elif cfg.get('work_dir', None) is None:  # cfgにwork_dirがなければ
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])  # 設定ファイル名からwork_dirを生成

    # enable automatic-mixed-precision training
    if args.amp is True:  # AMPが有効なら
        cfg.optim_wrapper.type = 'AmpOptimWrapper'  # OptimWrapperのタイプをAMP用に変更
        cfg.optim_wrapper.loss_scale = 'dynamic'  # ロススケールを動的に設定

    # enable automatically scaling LR
    if args.auto_scale_lr:  # 学習率自動スケーリングが有効なら
        if 'auto_scale_lr' in cfg and \
                'enable' in cfg.auto_scale_lr and \
                'base_batch_size' in cfg.auto_scale_lr:  # 必要な設定が揃っていれば
            cfg.auto_scale_lr.enable = True  # 自動スケーリングを有効化
        else:
            raise RuntimeError('Can not find "auto_scale_lr" or '
                               '"auto_scale_lr.enable" or '
                               '"auto_scale_lr.base_batch_size" in your'
                               ' configuration file.')  # 設定不足ならエラー

    # resume is determined in this priority: resume from > auto_resume
    if args.resume == 'auto':  # resumeがautoなら
        cfg.resume = True  # 自動再開を有効化
        cfg.load_from = None  # チェックポイントパスは指定しない
    elif args.resume is not None:  # resumeが指定されていれば
        cfg.resume = True  # 再開を有効化
        cfg.load_from = args.resume  # 指定されたチェックポイントから再開

    # build the runner from config
    if 'runner_type' not in cfg:  # runner_typeが設定されていなければ
        # build the default runner
        runner = Runner.from_cfg(cfg)  # デフォルトのRunnerを構築
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)  # レジストリからカスタムRunnerを構築

    # start training
    runner.train()  # トレーニングを開始


if __name__ == '__main__':  # スクリプトが直接実行された場合
    main()  # main関数を呼び出す
