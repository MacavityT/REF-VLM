# Copyright (c) OpenMMLab. All rights reserved.
import os
import cv2
import shutil
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
from xtuner.utils import IGNORE_INDEX, VISUAL_PROMPT_INDEX
from xtuner.registry import BUILDER, DATASETS, MAP_FUNC
from xtuner.dataset.single_dataset import OfflineDataset
from .huggingface import process_hf_dataset, encode_fn
from .utils import (
    imfrombytes,
    expand2square,
    bbox2mask,
    point2mask,
    boxes_xyxy_expand2square,
    points_xy_expand2square,
    masks_expand2square,
    mask_transform,
    norm_box_xyxy, 
    norm_point_xyxy,
    de_norm_box_xyxy,
    visualize_mask,
    visualize_mask_single,
    visualize_box_single,
    visualize_box,
    visualize_point
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
                data_list = self.dataset_process()
                self.data = TorchConcatDataset(data_list)
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
            elif isinstance(dataset_map_fn, dict) or \
                isinstance(dataset_map_fn, Config) or \
                isinstance(dataset_map_fn, ConfigDict):
                self.dataset_map_fn = partial(
                    dataset_map_fn['function'], 
                    **dataset_map_fn['args']
                )

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

            item['ori_height'] = item['image']['height']
            item['ori_width'] = item['image']['width']
            if 'target' in item.keys():
                self.target_process(item['target'],
                                    width=item['ori_width'],
                                    height=item['ori_height'])

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
            if 'masks' in target.keys():
                target['masks'] = masks_expand2square(target['masks'])
            width = max(width, height)
            height = width
        
        # normalize or transform all targets
        if 'boxes' in target.keys():
            normalized_boxes = []
            for box in target['boxes']:
                normalized_boxes.append(
                    norm_box_xyxy(box, w=width, h=height)
                )
            target['boxes'] = normalized_boxes

        if 'points' in target.keys():
            normalized_points = []
            for point in target['points']:
                normalized_points.append(
                    norm_point_xyxy(point, w=width, h=height)
                )
            target['points'] = normalized_points

        if 'masks' in target.keys():
            transformed_masks = []
            for mask in target['masks']:
                transformed_masks.append(
                    mask_transform(mask, self.image_processor)
                )
            target['masks'] = transformed_masks

    def image_process(self, image):
        # load image
        image_path = image
        try:
            image = imfrombytes(image, flag='color', channel_order='rgb') # array
        except:
            image_path = ''
            image = np.zeros((336,336,3)).astype(np.uint8)
            # ori_width = 0
            # ori_height = 0
        image = Image.fromarray(image) # PIL.Image
        ori_width = image.size[0]
        ori_height = image.size[1]
        # expand2square
        if self.pad_image_to_square:
            image = expand2square(
                image,
                tuple(int(x * 255) for x in self.image_processor.image_mean)
            )
            # image.save('square_image.jpg')
        if ori_width == 1 and ori_height == 1:
            print_log(f"Warning: Image path {image_path} is invalid! Please check the image path.")
            image = image.resize((336,336))
            image_path = ''
            ori_width = 336
            ori_height = 336
        image = self.image_processor.preprocess(
            image, return_tensors='pt')['pixel_values'][0]

        
        return dict(
            pixel_values = image,
            ori_width = ori_width,
            ori_height = ori_height,
            image_path = image_path,
        )
    
    def visual_prompts_process(self, visual_prompts, ori_width, ori_height, max_num):
        if self.pad_image_to_square:
            ori_width = max(ori_width, ori_height)
            ori_height = ori_width

        converted_vpt = []
        for vpt_one_turn in visual_prompts:
            if vpt_one_turn is None: continue
            for vpt in vpt_one_turn:
                if vpt['type'] == 'box':
                    box = de_norm_box_xyxy(vpt['value'], w=ori_width, h=ori_height)
                    mask = bbox2mask(box, width=ori_height, height=ori_height)
                elif vpt['type'] == 'point':
                    assert vpt['value'].size == 4 or vpt['value'].size == 2
                    if vpt['value'].size == 4:
                        assert all(vpt[2:4] < 0)
                    elif vpt['value'].size == 2:
                        new_vpt = np.ones(4) * IGNORE_INDEX
                        new_vpt[:2] = vpt['value']
                        vpt['value'] = new_vpt
                    else:
                        raise ValueError("Points target value error!")
                    point = de_norm_box_xyxy(vpt['value'], w=ori_width, h=ori_height)
                    mask = point2mask(point, radius=10, width=ori_height, height=ori_height)
                elif vpt['type'] == 'mask':
                    # scribble or mask
                    mask = vpt['value']
                transformed_mask = mask_transform(mask, self.image_processor)
                converted_vpt.append(transformed_mask)

                if (len(converted_vpt)) == max_num: 
                    return converted_vpt
        return converted_vpt

    def __getitem__(self, index):
        data_dict = self.data[index]

        # image
        if data_dict.get('image', None) is not None:
            image_info = data_dict['image']
            image_path = image_info['path']
            image_meta = self.image_process(image_path)
            data_dict['pixel_values'] = image_meta['pixel_values']
            if 'width' in data_dict['image'].keys() and 'height' in data_dict['image'].keys():
                data_dict['ori_width'] = data_dict['image']['width']
                data_dict['ori_height'] = data_dict['image']['height']
            else:
                data_dict['ori_width'] = image_meta['ori_width']
                data_dict['ori_height'] = image_meta['ori_height']
            data_dict['image_path'] = image_meta['image_path']
        else:
            if hasattr(self.image_processor, 'crop_size'):
                crop_size = self.image_processor.crop_size
            else:
                crop_size = self.image_processor.size
            data_dict['pixel_values'] = torch.zeros(3, crop_size['height'],
                                                    crop_size['width'])
            data_dict['ori_height'] = 0
            data_dict['ori_width'] = 0
            data_dict['image_path'] = ''
        
        if 'input_ids' not in data_dict.keys():
            if 'target' in data_dict.keys():
                if 'image' in data_dict.keys():
                    self.target_process(
                        data_dict['target'], 
                        width=data_dict['ori_width'],
                        height=data_dict['ori_height']
                    )
                else:
                    from xtuner.model.utils import save_wrong_data
                    save_wrong_data("wrong_okapi",data_dict)
                    del data_dict['target']
            # 'visual_prompts', 'decode_labels', 'conversation'
            data_dict.update(self.dataset_map_fn(data_dict))
            data_dict.update(self.template_map_fn(data_dict))
            # 'input_ids', 'labels'
            data_dict.update(
                encode_fn(
                    example=data_dict,
                    tokenizer=self.tokenizer,
                    max_length=self.max_length,
                    input_ids_with_output=True,
                    with_image_token=True,
                    visual_prompts=data_dict.get('visual_prompts', None)
                )
            )

        if data_dict.get('visual_prompts', None) is not None:
            assert data_dict.get('image', None) is not None, \
                'visual prompts set, but no image input.'
            max_num = data_dict['input_ids'].count(VISUAL_PROMPT_INDEX)
            if max_num > 0:
                visual_prompts = self.visual_prompts_process(
                    data_dict['visual_prompts'],
                    ori_width=data_dict['ori_width'],
                    ori_height=data_dict['ori_height'],
                    max_num = max_num
                )
                data_dict['visual_prompts'] = visual_prompts
            elif max_num == 0:
                data_dict.pop('visual_prompts',None)
            else:
                raise f"max num:{max_num} is lower than 0"

        #region debug
        # ori_path = 'vis_origin.jpg'
        # shutil.copy(data_dict['image_path'], ori_path)

        # image = data_dict['pixel_values'].numpy().transpose((1, 2, 0))
        # image = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        # image = image.astype(np.uint8)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # res_path = 'vis_normed.jpg'
        # cv2.imwrite(res_path, image)

        # if 'visual_prompts' in data_dict.keys():
        #     vpts = data_dict['visual_prompts']
        #     for i,vpt in enumerate(vpts):
        #         vis_vpt = visualize_mask_single(image, vpt, alpha=1.0, beta=1.0)
        #         save_path = f'vis_vpt_{i}.jpg'
        #         cv2.imwrite(save_path, vis_vpt)

        # if 'masks' in data_dict['target'].keys():
        #     masks = data_dict['target']['masks']
        #     for j,mask in enumerate(masks):
        #         vis_mask = visualize_mask_single(image, mask, alpha=1.0, beta=1.0)
        #         save_path = f'vis_mask_{j}.jpg'
        #         cv2.imwrite(save_path, vis_mask)
        
        # if 'boxes' in data_dict['target'].keys():
        #     boxes = data_dict['target']['boxes']
        #     width = image.shape[0]
        #     height = image.shape[1]
        #     for k,box in enumerate(boxes):
        #         denorm_box = de_norm_box_xyxy(box,width,height)
        #         vis_box = visualize_box_single(image.copy(), denorm_box)
        #         save_path = f'vis_box_{k}.jpg'
        #         cv2.imwrite(save_path, vis_box)
        #endregion

        return data_dict

