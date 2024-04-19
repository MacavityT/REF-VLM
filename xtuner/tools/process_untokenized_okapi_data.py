# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import warnings

from mmengine import Config

from xtuner.registry import BUILDER
# import debugpy
# debugpy.connect(('127.0.0.1', 5577))

# ignore FutureWarning in hf datasets
warnings.simplefilter(action='ignore', category=FutureWarning)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='config file name or path.')
    args = parser.parse_args()
    return args


def build_okapi_dataset(config):
    dataset = BUILDER.build(config.okapi_dataset)
    return dataset


if __name__ == '__main__':
    args = parse_args()
    cfg = Config.fromfile(args.config)

    okapi_dataset = build_okapi_dataset(cfg)
    okapi_dataset.save_offline_dataset()

