# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import numpy as np
from mmengine.config import Config
from xtuner.dataset import *
from xtuner.registry import BUILDER, DATASETS


# import debugpy
# debugpy.connect(('127.0.0.1', 5577))


def parse_args():
    parser = argparse.ArgumentParser(description='Log processed dataset.')
    parser.add_argument('config', help='config file name or path.')
    # chose which kind of dataset style to show
    parser.add_argument(
        '--show',
        default='text',
        choices=['text', 'masked_text', 'input_ids', 'labels', 'all'],
        help='which kind of dataset style to show')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    print(cfg.train_dataloader.dataset)
    tokenizer = BUILDER.build(cfg.tokenizer)
    if cfg.get('framework', 'mmengine').lower() == 'huggingface':
        train_dataset = BUILDER.build(cfg.train_dataset)
    else:
        train_dataset = BUILDER.build(cfg.train_dataloader.dataset)

    example = train_dataset[0]

    if args.show == 'text' or args.show == 'all':
        print('#' * 20 + '   text   ' + '#' * 20)

        example_with_pad = example['input_ids']
        example_with_pad = np.where(example_with_pad != -100,example_with_pad,tokenizer.pad_token_id)
        example_with_pad = np.where(example_with_pad != -200,example_with_pad,tokenizer.pad_token_id)
        print(tokenizer.decode(example_with_pad))
    if args.show == 'masked_text' or args.show == 'all':
        print('#' * 20 + '   text(masked)   ' + '#' * 20)
        masked_text = ' '.join(
            ['[-100]' for i in example['labels'] if i == -100])
        unmasked_text = tokenizer.decode(
            [i for i in example['labels'] if i != -100])
        print(masked_text + ' ' + unmasked_text)
    if args.show == 'input_ids' or args.show == 'all':
        print('#' * 20 + '   input_ids   ' + '#' * 20)
        print(example['input_ids'])
    if args.show == 'labels' or args.show == 'all':
        print('#' * 20 + '   labels   ' + '#' * 20)
        print(example['labels'])

    print(example)

if __name__ == '__main__':
    main()
