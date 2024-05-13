# Copyright (c) OpenMMLab. All rights reserved.
import os
import json
import jsonlines
import numpy as np

from typing import Dict, Any, List
from functools import partial
from tqdm import tqdm, trange
from PIL import Image

import torch
from datasets import Dataset as HFDataset
from mmengine import print_log
from mmengine.config import Config, ConfigDict
from mmengine.utils.misc import get_object_from_string

from torch.utils.data import Dataset
from torch.utils.data import ConcatDataset as TorchConcatDataset
from xtuner.utils import IGNORE_INDEX
from xtuner.registry import BUILDER, DATASETS, MAP_FUNC
from xtuner.dataset.single_dataset import OfflineDataset
from .huggingface import process_hf_dataset, encode_fn
from .utils import (
    imfrombytes,
    expand2square,
    bbox2mask,
    point2mask,
    boxes_xyxy_expand2square,
    points_xy_expand2square
)
from xtuner.utils.constants import SPECIAL_TOKENS

REFORM_DATASET = [
    'SubSet',
    'InterleaveDateset',
    'ConcatDatasetWithShuffle'
]

class OkapiDataset(Dataset):

    def __init__(self,
                 dataset=None,
                 image_processor=None,
                 tokenizer=None,
                 max_dataset_length=None,
                 dataset_map_fn=None,
                 template_map_fn=None,
                 max_length=2048,
                 shard_process_max_length=5e4,
                 pad_image_to_square=False,
                 save_offline_dataset=False,
                 pretokenize=True):
        super().__init__()

        self.max_dataset_length = max_dataset_length
        self.dataset_map_fn = dataset_map_fn
        self.template_map_fn = template_map_fn
        self.max_length = max_length
        self.shard_process_max_length=int(shard_process_max_length)
        self.pad_image_to_square = pad_image_to_square
        self.pretokenize = pretokenize
        self.init_visual_tokenizer(image_processor, tokenizer)

        # Build datasets
        print_log("Okapi Datasets Building ...")
        self.dataset = self.build_dataset(dataset)
        print_log("Okapi Datasets Build Success.")

        if self.pretokenize:
            #taiyan TODO: modify save offline dataset process to multi-thread real-time, avoiding use 'process_hf_dataset'
            if not save_offline_dataset:
                print_log("Okapi Datasets PreTokenize Processing ...")
                self.data_list = self.dataset_process()
                self.data = TorchConcatDataset(self.data_list)
                print_log("Okapi Datasets PreTokenize Process Success.")
            else:
                print_log("Datasets build success, call function 'save_offline_dataset' to save.")
        else:
            if isinstance(dataset_map_fn, str):
                map_fn_obj = MAP_FUNC.get(dataset_map_fn) or \
                    get_object_from_string(dataset_map_fn)
                if map_fn_obj is not None:
                    self.dataset_map_fn = map_fn_obj
                else:
                    raise TypeError('dataset_map_fn must be a function or a '
                                    "registered function's string in MAP_FUNC, "
                                    f"but got a string of '{dataset_map_fn}'")

            if isinstance(template_map_fn, dict) or \
                isinstance(template_map_fn, Config) or \
                isinstance(template_map_fn, ConfigDict):
                self.template_map_fn = BUILDER.build(template_map_fn)
            self.data = TorchConcatDataset(self.dataset)
            print_log("'pretokenize' is set to False, getitem and map_fn real-time.")


    @property
    def modality_length(self):
        # only work when self.pretokenize = True
        length_list = []
        for data_dict in self.data:
            cur_len = len(data_dict['input_ids'])
            if data_dict.get('image', None) is None:
                cur_len = -cur_len
            length_list.append(cur_len)
        return length_list


    def __len__(self):
        return len(self.data)

    def init_visual_tokenizer(self, image_processor, tokenizer):
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

        self.tokenizer.add_tokens(SPECIAL_TOKENS, special_tokens=True)

    def build_dataset(self, dataset):
        if isinstance(dataset, list):
            dataset_build_fn = []
            for ds_args in dataset:
                dataset_build_fn.append(partial(
                    DATASETS.build,
                    ds_args
                ))
            dataset = [fn() for fn in dataset_build_fn]
        else:
            dataset = [dataset]
        return dataset

    def single_dataset_process(self, index, dataset, return_shards=False):
        all_shards = []
        shard_ds_data = []
        shard_idx = 1
        for i in tqdm(range(len(dataset)), desc=f'Processing Dataset {index}'):
            item = dataset[i]
            if 'width' not in item['image'].keys() or \
                'height' not in item['image'].keys():
                image_path = item['image']['path']
                image = imfrombytes(image_path, flag='unchanged') # array
                item['image']['height'] = image.shape[0]
                item['image']['width'] = image.shape[1]
            
            if 'target' in item.keys():
                self.target_process(item['target'],
                                    width=item['image']['width'],
                                    height=item['image']['height'])

            if i < (shard_idx * self.shard_process_max_length):
                shard_ds_data.append(item)
            else:
                all_shards.append(shard_ds_data)
                shard_ds_data = [item]
                shard_idx += 1            
        all_shards.append(shard_ds_data) # append the last shard (or the only shard)

        # torch.distributed.barrier()

        if return_shards: # for offline saving process
            return all_shards

        # gather all shards
        all_shards_hf = []
        for shard in all_shards:
            shard_hf = process_hf_dataset(
                dataset=HFDataset.from_list(shard),
                tokenizer=self.tokenizer,
                max_length=self.max_length,
                dataset_map_fn=self.dataset_map_fn,
                template_map_fn=self.template_map_fn,
                split=None,
                max_dataset_length=self.max_dataset_length,
                remove_unused_columns=False,
                pack_to_max_length=False,
                with_image_token=True
                )
            all_shards_hf.append(shard_hf)

        ds_data_hf = TorchConcatDataset(all_shards_hf)
        return ds_data_hf

    def save_offline_dataset(self):
        def _write_json(path, content):
            with open(path, 'w') as f:
                f.write(json.dumps(content))
        
        for ds_idx, ds in enumerate(self.dataset):
            assert type(ds).__name__ not in REFORM_DATASET, \
                f"Dataset {ds_idx} with type of {type(ds).__name__}, offline save process not supported!"
            assert ds.offline_processed_text_folder is not None, \
                f"Dataset {ds_idx} offline text folder is None."
            if os.path.exists(ds.offline_processed_text_folder) and \
                (not ds.enforce_online):
                print(f"Skipped Dataset {ds_idx}, offline text folder existed.")
                continue

            os.makedirs(ds.offline_processed_text_folder, exist_ok=True)
            all_shards = self.single_dataset_process(ds_idx, ds, return_shards=True)
            total_length = 0
            for shard in all_shards:
                length = len(shard)
                total_length += length
            print_log(f'Dataset {ds_idx}: shards num {len(all_shards)}, data item num {total_length}')
            
            filename_count = 0
            for shard_idx, shard in enumerate(all_shards):
                # checking current files, if exists then skip 'process_hf_dataset' function
                skip_flag = False
                for check_idx in range(shard_idx*self.shard_process_max_length, \
                                       (shard_idx+1)*self.shard_process_max_length):
                    check_file = os.path.join(
                        ds.offline_processed_text_folder, 
                        f"offline_text_file_{check_idx}.json"
                    )
                    if not os.path.isfile(check_file):
                        break
                    # if all check file exists, then set flag
                    skip_flag = True
                    
                if skip_flag:
                    filename_count = (shard_idx+1)*self.shard_process_max_length
                    print_log(f"Shard {shard_idx} offline files exists, skipped.")
                    continue

                shard_hf = process_hf_dataset(
                    dataset=HFDataset.from_list(shard),
                    tokenizer=self.tokenizer,
                    max_length=self.max_length,
                    dataset_map_fn=self.dataset_map_fn,
                    template_map_fn=self.template_map_fn,
                    split=None,
                    max_dataset_length=self.max_dataset_length,
                    remove_unused_columns=False,
                    pack_to_max_length=False,
                    with_image_token=True
                    )

                for item in tqdm(shard_hf, desc=f'Writing shard {shard_idx}'):
                    filename = f"offline_text_file_{filename_count}.json"
                    save_path = os.path.join(
                        ds.offline_processed_text_folder, 
                        filename
                    )
                    assert not os.path.exists(save_path), \
                    f"{save_path} existed."
                    _write_json(save_path, item)
                    filename_count += 1

    def dataset_process(self):
        data_list = []
        for idx, ds in enumerate(self.dataset):
            if type(ds).__name__ in REFORM_DATASET and (not ds.enforce_online):
                ds_data_hf = ds
                print_log(f"Dataset {idx} with type of {type(ds).__name__} offline prepared, please make sure all datasets in it with offline.")
            elif (type(ds).__name__ not in REFORM_DATASET) and isinstance(ds.text_data, OfflineDataset):
                ds_data_hf = ds
                print_log(f"Dataset {idx} offline prepared.")
            else:
                ds_data_hf = self.single_dataset_process(idx, ds)
            data_list.append(ds_data_hf)

        return data_list

    def target_process(self, target, width, height):
        if self.pad_image_to_square:
            if 'boxes' in target.keys():
                target['boxes'] = boxes_xyxy_expand2square(target['boxes'], width=width, height=height)
            if 'points' in target.keys():
                target['points'] = points_xy_expand2square(target['points'], width=width, height=height)

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
    
    def visual_prompts_process(self, visual_prompts: List[torch.FloatTensor]):
        #taiyan TODO: convert visual prompts to masks(tensor)
        if 'box':
            res = bbox2mask()
        elif 'point':
            res = point2mask()
        
        return visual_prompts

    def __getitem__(self, index):
        data_dict = self.data[index]

        if not self.pretokenize:
            # add keys: 'conversation', 'input_ids', 'labels'
            data_dict.update(self.dataset_map_fn(data_dict))
            data_dict.update(self.template_map_fn(data_dict))
            data_dict.update(
                encode_fn(
                    example=data_dict,
                    tokenizer=self.tokenizer,
                    max_length=self.max_length,
                    input_ids_with_output=True,
                    with_image_token=True
                )
            )

        # image
        if data_dict.get('image', None) is not None:
            image_info = data_dict['image']
            image_path = image_info['path']
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

        if data_dict.get('visual_prompts', None) is not None:
            assert data_dict.get('image', None) is not None, \
                'visual prompts set, but no image input.'
            visual_prompts = self.visual_prompts_process(data_dict['visual_prompts'])
            data_dict['visual_prompts'] = visual_prompts
        return data_dict

