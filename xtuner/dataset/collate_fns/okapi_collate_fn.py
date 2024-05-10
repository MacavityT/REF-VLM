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

    if not has_vpt:
        return data_dict
    
    visual_prompts = []
    for example in instances:
        if example.get('visual_prompts'):
            assert example.get('pixel_values') is not None, \
                'visual prompts set, but no image input.'
            
            # List[List[Tensor]]
            visual_prompts.append(example['visual_prompts'])
        else:
            visual_prompts.append(None)

    data_dict['visual_prompts']  = visual_prompts

    if return_hf_format:
        return data_dict
    else:
        return {'data': data_dict, 'data_samples': None}
