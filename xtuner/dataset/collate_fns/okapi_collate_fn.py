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
    
    # example['conversations'] is the origin data from single dataset
    # example['conversation'] is the data processed with map_fn
    conversations = []
    for example in instances:
        conversations.append(example.get('conversation', None))
    data_dict['conversations']  = conversations

    dynamic_keys = [
        'decode_units', 'visual_prompts',
        'decode_labels', 'decode_seqs', 
        'pixel_values_tower'
    ]
    for key in dynamic_keys:
        exist = any(inst.get(key) is not None for inst in instances)
        if not exist: continue
        contents = []
        for example in instances:
            if example.get(key) is not None:
                contents.append(example[key])
            else:
                contents.append(None)
        data_dict[key] = contents    

    image_info_keys = ['image_path', 'ori_height', 'ori_width', 'pixel_masks']
    for key in image_info_keys:
        data_dict[key] = [example[key] for example in instances]

    if 'pixel_values_tower' in data_dict.keys():
        data_dict['pixel_values_tower'] = torch.stack(data_dict['pixel_values_tower'])

    if return_hf_format:
        return data_dict
    else:
        return {'data': data_dict, 'data_samples': None}
