# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import json

import torch
from mmengine import Config
from mmengine.registry import init_default_scope

from mmpretrain import get_model

try:
    from mmengine.analysis import get_model_complexity_info
except ImportError:
    raise ImportError('Please upgrade mmengine >= 0.6.0')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Get a editor complexity',
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('config', help='train config file path')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    print(json.dumps(cfg.model.to_dict(), indent=4))

    model = get_model(args.config)

    param_num = sum([param.numel() for param in model.parameters()])
    print(f'Number of parameters: {param_num:,}')


if __name__ == '__main__':
    main()
