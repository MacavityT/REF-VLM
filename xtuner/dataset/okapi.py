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
    get_pixel_mask,
    bbox2mask,
    point2mask,
    boxes_xyxy_expand2square,
    points_xy_expand2square,
    keypoints_xyc_expand2square,
    masks_expand2square,
    mask_transform,
    norm_box_xyxy, 
    norm_point_xyxy,
    de_norm_box_xyxy,
    de_norm_keypoint,
    visualize_keypoints,
    box_xyxy_to_xywh,
    de_norm_keypoint_square2origin,
    de_norm_box_xyxy_square2origin,
    denorm_box_xywh_square2origin,
    box_xywh_to_xyxy,
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
                 image_tower_processor=None,
                 tokenizer=None,
                 max_dataset_length=None,
                 dataset_map_fn=None,
                 template_map_fn=None,
                 max_length=2048,
                 pad_image_to_square=False,
                 mode='train'):
        super().__init__()

        self.max_dataset_length = max_dataset_length
        self.dataset_map_fn = dataset_map_fn
        self.template_map_fn = template_map_fn
        self.max_length = max_length
        self.pad_image_to_square = pad_image_to_square
        self.data = None
        self.mode = mode
        self.init_visual_tokenizer(
            image_processor, 
            tokenizer, 
            image_tower_processor=image_tower_processor
        )

        if mode == 'train' or mode == 'test':
            # Build datasets
            print_log("Okapi Datasets Building ...")
            self.dataset = self.build_dataset(dataset)
            print_log("Okapi Datasets Build Success.")
            self.data = TorchConcatDataset(self.dataset)
        else:
            print_log("Mode = inference, Okapi Datasets is empty ...")
            self.data = []

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
    
    def add_dataset(self,dataset):
        assert self.mode == 'inference', 'Wrong mode, only inference mode could add additional dataset.'
        '''for inference mode, add dataset'''
        if isinstance(dataset,Dataset):
            self.data = dataset
        elif isinstance(dataset,list):
            self.dataset = self.build_dataset(dataset)
            self.data = TorchConcatDataset(self.dataset)
        else:
            raise NotImplementedError

    def __len__(self):
        return len(self.data)

    def init_visual_tokenizer(self, image_processor, tokenizer, image_tower_processor=None):
        if isinstance(image_processor, dict) or isinstance(
                image_processor, Config) or isinstance(image_processor,
                                                       ConfigDict):
            self.image_processor = BUILDER.build(image_processor)
        else:
            self.image_processor = image_processor

        if isinstance(image_tower_processor, dict) or isinstance(
                image_tower_processor, Config) or isinstance(image_tower_processor,
                                                       ConfigDict):
            self.image_tower_processor = BUILDER.build(image_tower_processor)
        else:
            self.image_tower_processor = image_tower_processor

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

    def target_process(self, target, width, height):
        if self.pad_image_to_square:
            if 'boxes' in target.keys():
                target['boxes'] = boxes_xyxy_expand2square(target['boxes'], width=width, height=height)
            if 'points' in target.keys():
                target['points'] = points_xy_expand2square(target['points'], width=width, height=height)
            if 'masks' in target.keys():
                target['masks'] = masks_expand2square(target['masks'])
            if 'keypoints' in target.keys():
                target['keypoints'] = keypoints_xyc_expand2square(target['keypoints'], width=width, height=height)
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

        if 'keypoints' in target.keys():
            normalized_all_kpts = []
            for keypoints in target['keypoints']:
                normalized_kpts = []
                for keypoint in keypoints:
                    point_class = keypoint[-1]
                    norm_x, norm_y = norm_point_xyxy(keypoint[:-1], w=width, h=height)
                    normalized_kpts.append(
                        [norm_x, norm_y, point_class]
                    )
                normalized_all_kpts.append(normalized_kpts)
            target['keypoints'] = normalized_all_kpts

    def image_process(self, image):
        # load image
        image_path = image
        if isinstance(image_path,str):
            try:
                image = imfrombytes(image, flag='color', channel_order='rgb') # array
            except:
                print_log(f"Warning: Image path {image_path} is invalid! Please check the image path.")
                image_path = ''
                image = np.zeros((336,336,3)).astype(np.uint8)
        elif isinstance(image_path, np.ndarray):
            image_path = ''
        image = Image.fromarray(image) # PIL.Image
        ori_width = image.size[0]
        ori_height = image.size[1]
        # expand2square
        if self.pad_image_to_square:
            image = expand2square(
                image,
                tuple(int(x * 255) for x in self.image_processor.image_mean)
            )
        if ori_width == 1 and ori_height == 1:
            print_log(f"Warning: Image path {image_path} is invalid! Please check the image path.")
            image = image.resize((336, 336))
            image_path = ''
            ori_width = 336
            ori_height = 336

        image_tower = None
        if self.image_tower_processor is not None:
            image_tower = self.image_tower_processor.preprocess(
                image, return_tensors='pt')['pixel_values'][0]
        image = self.image_processor.preprocess(
            image, return_tensors='pt')['pixel_values'][0]

        return dict(
            pixel_values = image,
            pixel_values_tower = image_tower,
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
                    assert (vpt['value'][2:] >= vpt['value'][:2]).all(), \
                        f"boxes label must be in [x1, y1, x2, y2] (corner) format, but got {vpt['value']}"
                    box = de_norm_box_xyxy(vpt['value'], w=ori_width, h=ori_height)
                    mask = bbox2mask(box, width=ori_height, height=ori_height)
                elif vpt['type'] == 'point':
                    assert vpt['value'].size == 4 or vpt['value'].size == 2
                    if vpt['value'].size == 4:
                        assert all(vpt['value'][2:] < 0)
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
                else:
                    raise ValueError(f"Unsupport vpt type: {vpt['type']}")
                transformed_mask = mask_transform(mask, self.image_processor)
                converted_vpt.append(transformed_mask)

                if (len(converted_vpt)) == max_num: 
                    return converted_vpt
        return converted_vpt

    def decode_labels_process(self, decode_labels):
        converted_labels = dict()
        for label_one_turn in decode_labels:
            if label_one_turn is None: continue
            for label in label_one_turn:
                if label['type'] == 'box':
                    assert (label['value'][2:] >= label['value'][:2]).all(), \
                        f"boxes label must be in [x1, y1, x2, y2] (corner) format, but got {label['value']}"
                    unit_label = box_xyxy_to_xywh(label['value'])
                elif label['type'] == 'point':
                    assert label['value'].size == 4 or label['value'].size == 2
                    if label['value'].size == 4:
                        assert all(label['value'][2:] < 0)
                        unit_label = label['value']
                        unit_label[2:] = 0
                    elif label['value'].size == 2:
                        unit_label = np.zeros(4)
                        unit_label[:2] = label['value']
                    else:
                        raise ValueError("Points target value error!")
                elif label['type'] == 'mask':
                    unit_label = label['value']
                elif label['type'] == 'keypoint':
                    unit_label = label['value']
                else:
                    raise ValueError(f"Unsupport label type: {label['type']}")
                
                if label['type'] not in converted_labels.keys():
                    converted_labels[label['type']] = [unit_label]
                else:
                    converted_labels[label['type']].append(unit_label)
        return converted_labels

    def decode_seqs_process(self, decode_seqs):
        converted_seqs = []
        for seq_one_turn in decode_seqs:
            if seq_one_turn is None: continue
            for seq in seq_one_turn:
                converted_seqs.append(seq)
        return converted_seqs

    def __getitem__(self, index):
        assert self.data is not None, 'Please add valid dataset first!'
        data_dict = self.data[index]

        try:
            # image
            if data_dict.get('image', None) is not None:
                image_info = data_dict['image']
                if 'path' in image_info.keys():
                    image = image_info['path']
                elif 'value' in image_info.keys():
                    image = image_info['value']
                image_meta = self.image_process(image)
                data_dict.update(image_meta)
            else:
                if hasattr(self.image_processor, 'crop_size'):
                    crop_size = self.image_processor.crop_size
                else:
                    crop_size = self.image_processor.size
                data_dict['pixel_values'] = torch.zeros(3, crop_size['height'],
                                                        crop_size['width'])
                if self.image_tower_processor is None:
                    data_dict['pixel_values_tower'] = None
                else:
                    if hasattr(self.image_tower_processor, 'crop_size'):
                        crop_size = self.image_tower_processor.crop_size
                    else:
                        crop_size = self.image_tower_processor.size
                    data_dict['pixel_values_tower'] = torch.zeros(3, crop_size['height'],
                                                            crop_size['width'])
                data_dict['ori_height'] = crop_size['height']
                data_dict['ori_width'] = crop_size['width']
                data_dict['image_path'] = ''

            data_dict['pixel_masks'] = get_pixel_mask(
                data_dict['pixel_values'],
                data_dict['ori_width'],
                data_dict['ori_height']
            )
            
            if 'input_ids' not in data_dict.keys():
                if 'target' in data_dict.keys():
                    self.target_process(
                        data_dict['target'], 
                        width=data_dict['ori_width'],
                        height=data_dict['ori_height']
                    )
                # 'visual_prompts', 'decode_labels', 'decode_units', 'conversation'
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
                        max_num = max_num # input content might be cut off
                    )
                    data_dict['visual_prompts'] = visual_prompts
                elif max_num == 0:
                    data_dict.pop('visual_prompts', None)
                else:
                    raise f"max num:{max_num} is lower than 0"

            # decode units
            if data_dict.get('decode_units', None) is not None:
                assert data_dict.get('image', None) is not None, \
                    'decode units set, but no image input.'
                first_unit = data_dict['decode_units'][0]
                assert all(unit == first_unit \
                    for unit in data_dict['decode_units'])
                data_dict['decode_units'] = first_unit

            # decode seqs
            if data_dict.get('decode_seqs', None) is not None:
                
                decode_seqs = self.decode_seqs_process(
                    data_dict['decode_seqs']
                )
                data_dict['decode_seqs'] = decode_seqs

            # decode labels
            if data_dict.get('decode_labels', None) is not None:
                decode_labels = self.decode_labels_process(
                    data_dict['decode_labels']
                )
                data_dict['decode_labels'] = decode_labels
        except Exception as e:
            from xtuner.model.utils import save_wrong_data
            print(e)
            save_wrong_data(f"wrong_data_dict", data_dict)
            raise ValueError('Error in get data process')

        # #region debug
        # ori_path = 'vis_origin.jpg'
        # shutil.copy(data_dict['image_path'], ori_path)

        # image = data_dict['pixel_values'].numpy().transpose((1, 2, 0))
        # image = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        # image = image.astype(np.uint8)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # res_path = 'vis_normed.jpg'
        # cv2.imwrite(res_path, image)

        # if 'pixel_masks' in data_dict.keys():
        #     pixel_mask = data_dict['pixel_masks']
        #     vis_pixel_mask = visualize_mask_single(image, pixel_mask.cpu().numpy(), alpha=1.0, beta=1.0)
        #     save_path = f'pixel_mask.jpg'
        #     cv2.imwrite(save_path, vis_pixel_mask)

        # if 'decode_labels' in data_dict.keys():
        #     if 'mask' in data_dict['decode_labels'].keys():
        #         masks = data_dict['decode_labels']['mask']
        #         for j,mask in enumerate(masks):
        #             vis_mask = visualize_mask_single(image, mask, alpha=1.0, beta=1.0)
        #             save_path = f'decode_label_mask_{j}.jpg'
        #             cv2.imwrite(save_path, vis_mask)
        #     if 'box' in data_dict['decode_labels'].keys():
        #         boxes = data_dict['decode_labels']['box']
        #         width = image.shape[0]
        #         height = image.shape[1]
        #         for k,box in enumerate(boxes):
        #             box = box_xywh_to_xyxy(box)
        #             denorm_box = de_norm_box_xyxy(box,width,height)
        #             vis_box = visualize_box_single(image.copy(), denorm_box)
        #             save_path = f'decode_labels_box_{k}.jpg'
        #             cv2.imwrite(save_path, vis_box)

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


        # if 'keypoints' in data_dict['target'].keys():
        #     keypoints = data_dict['target']['keypoints']
        #     image = Image.open(ori_path)
        #     width_origin = image.width
        #     height_origin = image.height
        #     image = np.array(image)
        #     for p,keypoint in enumerate(keypoints):
        #         skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]] 
        #         keypoint = de_norm_keypoint_square2origin(np.array(keypoint),width_origin,height_origin)
        #         keypoint = np.array(keypoint)
        #         visualize_keypoints(image=image,keypoints=keypoint,skeleton=skeleton,index=p)

        # #endregion
        return data_dict
