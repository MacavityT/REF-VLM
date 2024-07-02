# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Sequence

import torch
from xtuner.utils import DEFAULT_PAD_TOKEN_INDEX, IGNORE_INDEX
from .default_collate_fn import default_collate_fn

def okapi_collate_fn(instances: Sequence[Dict],
                       pad_index: int = DEFAULT_PAD_TOKEN_INDEX,
                       return_hf_format: bool = False,
                       use_varlen_attn: bool = False):
    
    collate_results = default_collate_fn(
        instances, 
        pad_index, 
        return_hf_format, 
        use_varlen_attn)
    if return_hf_format:
        data_dict = collate_results
    else:
        data_dict = collate_results['data']
    
    has_vpt = any(inst.get('visual_prompts') is not None for inst in instances)
    has_decode_label = any(inst.get('decode_labels') is not None for inst in instances)

    if has_vpt:
        visual_prompts = []
        for example in instances:
            if example.get('visual_prompts', None):
                visual_prompts.append(example['visual_prompts'])
            else:
                visual_prompts.append(None)
        data_dict['visual_prompts']  = visual_prompts

    if has_decode_label:
        decode_labels = []
        for example in instances:
            if example.get('decode_labels', None):
                decode_labels.append(example['decode_labels'])
            else:
                decode_labels.append(None)
        data_dict['decode_labels'] = decode_labels

    image_path = [example['image_path'] for example in instances]
    ori_height = [example['ori_height'] for example in instances]
    ori_width = [example['ori_width'] for example in instances]
    data_dict['image_path'] = image_path
    data_dict['ori_height'] = ori_height
    data_dict['ori_width'] = ori_width

    if return_hf_format:
        return data_dict
    else:
        return {'data': data_dict, 'data_samples': None}
