# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
from typing import Dict, Any
from functools import partial
from tqdm import trange
from PIL import Image

import torch
from datasets import Dataset as HFDataset
from datasets import DatasetDict
from mmengine import print_log
from mmengine.config import Config, ConfigDict
from torch.utils.data import Dataset

from xtuner.registry import BUILDER, DATASETS, FUNCTIONS
from .huggingface import process_hf_dataset
from .utils import expand2square, all_expand2square, imfrombytes

class OkapiDataset(Dataset):

    def __init__(self,
                 dataset=None,
                 image_processor=None,
                 tokenizer=None,
                 max_dataset_length=None,
                 dataset_map_fn=None,
                 template_map_fn=None,
                 max_length=2048,
                 pad_image_to_square=False):
        super().__init__()

        self.max_dataset_length = max_dataset_length
        self.dataset_map_fn = dataset_map_fn
        self.template_map_fn = template_map_fn
        self.max_length = max_length
        self.pad_image_to_square = pad_image_to_square

        if isinstance(image_processor, dict) or isinstance(
                image_processor, Config) or isinstance(image_processor,
                                                       ConfigDict):
            self.image_processor = BUILDER.build(image_processor)
        else:
            self.image_processor = image_processor

        if isinstance(tokenizer, dict) or isinstance(
                tokenizer, Config) or isinstance(tokenizer, ConfigDict):
            self.tokenizer = BUILDER.build(tokenizer)
        else:
            self.tokenizer = tokenizer

        self.dataset = []
        if isinstance(dataset, dict):
            dataset_build_fn = {
                ds_name: partial(DATASETS.build, 
                image_processor = self.image_processor,
                tokenizer = self.tokenizer,
                pad_image_to_square=self.pad_image_to_square,
                **ds_args) 
                for ds_name, ds_args in dataset.keys()
            }       
            self.dataset = [fn() for fn in dataset_build_fn]
        elif isinstance(dataset, list):
            self.dataset = dataset
        else:
            self.dataset = [dataset]

        self.data = self.dataset_process()


    @property
    def modality_length(self):
        length_list = []
        for data_dict in self.data:
            cur_len = len(data_dict['input_ids'])
            if data_dict.get('image', None) is None:
                cur_len = -cur_len
            length_list.append(cur_len)
        return length_list


    def __len__(self):
        return len(self.data)

    def image_process(self, image):
        # load image
        image = imfrombytes(image, flag='color', channel_order='rgb') # array
        image = Image.fromarray(image) # PIL.Image
        # expand2square
        if self.pad_image_to_square:
            image = expand2square(
                image,
                tuple(
                    int(x * 255) for x in self.image_processor.image_mean))
            
        image = self.image_processor.preprocess(
            image, return_tensors='pt')['pixel_values'][0]
        return image

    def dataset_process(self):
        data_dict = {}
        for ds in self.dataset:
            '''
            item = {
                'image': {
                    'path': '/path/to/image', # str
                    'width': 512, # int
                    'height': 512, # int 
                },
                'target': {
                    # xmin, ymin, xmax, ymax
                    'boxes': [
                        [10, 10, 256, 265],  # dog1
                        [24, 18, 378, 768],  # dog2
                        [100, 310, 670, 653],  # man
                        [278, 320, 809, 673],  # rope
                    ],
                },
                "conversations": [
                    {
                        'from': 'human',
                        'value': 'What is the relation between the two dogs <boxes> and the man <boxes> in the image <image> ?',
                        'boxes_seq': [[0, 1], [2], ],
                    },
                    {
                        'from': 'gpt',
                        'value': 'a rope <boxes> is connecting the left dog <boxes> with the man <boxes>. '
                                    'So the man <boxes> is walking the dog <boxes>.'
                                'And the man <boxes> has no relationship with the right dog <boxes>',
                        'boxes_seq': [[3], [0], [2], [2], [0], [2], [1]],
                    }
                ]
            }
            '''
            ds_data = []
            for item in ds:
                if 'width' not in item['image'].keys() or \
                    'height' not in item['image'].keys():
                    image_path = item['image']['path']
                    image = imfrombytes(image_path, flag='unchanged') # array
                    item['height'] = image.shape[0]
                    item['width'] = image.shape[1]
                ds_data.append(item)
            data_dict[type(ds).__name__] = HFDataset.from_list(ds_data)
            
        gathered_data = DatasetDict(data_dict)
        return process_hf_dataset(
                    dataset=gathered_data,
                    tokenizer=self.tokenizer,
                    max_length=self.max_length,
                    dataset_map_fn=self.dataset_map_fn,
                    template_map_fn=self.template_map_fn,
                    split=None,
                    max_dataset_length=self.max_dataset_length,
                    remove_unused_columns=False,
                    pack_to_max_length=False,
                    with_image_token=True)


    def __getitem__(self, index):
        data_dict = self.data[index]

        # image
        if data_dict.get('image', None) is not None:
            image_path = data_dict['image']
            image = self.image_process(image_path)
            data_dict['pixel_values'] = image

            if image_path.split('.')[-1] == '.npy':
                data_dict['tensor_image'] = True

        else:
            if hasattr(self.image_processor, 'crop_size'):
                crop_size = self.image_processor.crop_size
            else:
                crop_size = self.image_processor.size
            data_dict['pixel_values'] = torch.zeros(3, crop_size['height'],
                                                    crop_size['width'])
        return data_dict

