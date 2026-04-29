# ============================================================
# Train script for MMSegmentation
# Model: MambaVision (backbone) + UPerNet (decoder)
# Dataset: ADE20K (default, configurable via config file)
#
# Usage:
#   python train.py path/to/config.py
#   python train.py path/to/config.py --amp
# ============================================================

import argparse
import logging
import os
import os.path as osp

from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmengine.runner import Runner

# MMSeg runner registry
from mmseg.registry import RUNNERS

# ------------------------------------------------------------
# IMPORTANT:
# This import is REQUIRED to register the custom backbone:
#   MM_mamba_vision
# ------------------------------------------------------------
import semantic_segmentation.mmseg_mambavision_backbone

def parse_args():
    parser = argparse.ArgumentParser(
        description='Train a semantic segmentation model (MambaVision + MMSeg)'
    )

    # --------------------------------------------------------
    # Config file (REQUIRED)
    # Example:
    #   configs/mamba/mamba_tiny_upernet_ade20k.py
    # --------------------------------------------------------
    parser.add_argument(
        'config',
        help='Path to training config file'
    )

    # Optional arguments
    parser.add_argument(
        '--work-dir',
        help='Directory to save logs and checkpoints'
    )

    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume training from latest checkpoint'
    )

    parser.add_argument(
        '--amp',
        action='store_true',
        help='Enable Automatic Mixed Precision (AMP)'
    )

    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='Override config options from command line'
    )

    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='Job launcher'
    )

    parser.add_argument(
        '--local_rank',
        '--local-rank',
        type=int,
        default=0
    )

    args = parser.parse_args()

    # Required for distributed training
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()

    # --------------------------------------------------------
    # Load config
    # --------------------------------------------------------
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # --------------------------------------------------------
    # Set work directory
    # Priority: CLI > config file > default
    # --------------------------------------------------------
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join(
            './work_dirs',
            osp.splitext(osp.basename(args.config))[0]
        )

    # --------------------------------------------------------
    # Enable AMP if requested
    # --------------------------------------------------------
    if args.amp:
        optim_wrapper_type = cfg.optim_wrapper.type
        if optim_wrapper_type == 'AmpOptimWrapper':
            print_log(
                'AMP is already enabled in config.',
                logger='current',
                level=logging.WARNING
            )
        else:
            assert optim_wrapper_type == 'OptimWrapper', \
                '`--amp` requires OptimWrapper in config.'
            cfg.optim_wrapper.type = 'AmpOptimWrapper'
            cfg.optim_wrapper.loss_scale = 'dynamic'

    # Resume training if specified
    cfg.resume = args.resume

    # --------------------------------------------------------
    # Build runner and start training
    # --------------------------------------------------------
    if 'runner_type' not in cfg:
        runner = Runner.from_cfg(cfg)
    else:
        runner = RUNNERS.build(cfg)

    runner.train()


if __name__ == '__main__':
    main()
