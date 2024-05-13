# Copyright (c) OpenMMLab. All rights reserved.
import re
from typing import List, Dict, Any, Tuple, Union
from xtuner.utils.constants import (
    BOXES_PLACEHOLDER, 
    POINTS_PLACEHOLDER, 
    MASKS_PLACEHOLDER,
    DEFAULT_IMAGE_TOKEN
    )
from xtuner.dataset.utils import norm_box_xyxy, norm_point_xyxy, de_norm_box_xyxy
from xtuner.utils import IGNORE_INDEX
from .okapi_map_fn import map_obj

# ret = {
#     'image': image,
#     'target': {'boxes': item['boxes']},  # 'seg' /
#     'conversations': [
#         {
#             'from': 'system',
#             'value':[{'task':{'task_name':'gcg_detection','element':['phrase','sentence'],'use_unit':True},'unit':['box']}]
#         },
#         {
#             'from': 'human',
#             'value': question,
#         },
#         {
#             'from': 'gpt',
#             'value': caption,
#             'boxes_seq': item['boxes_seq'],
#         }
#     ]
# }

def conversation_map_fn(example):
    messages = example['conversations']
    # assert len(messages) == 0.5*
    input = ''
    conversation = []
    while messages and messages[0]['from'] == 'gpt':
        # Skip the first one if it is from gpt
        messages = messages[1:]
    for msg in messages:
        if msg['from'] == 'human':
            if DEFAULT_IMAGE_TOKEN in msg['value']:
                msg['value'] = msg['value'].replace(DEFAULT_IMAGE_TOKEN,
                                                    '').strip()
                msg['value'] = DEFAULT_IMAGE_TOKEN + '\n' + msg['value']
                msg['value'] = msg['value'].strip()
            input += msg['value']

        elif msg['from'] == 'gpt':
            conversation.append({'input': input, 'output': msg['value']})
            input = ''
        else:
            raise NotImplementedError
    return {'conversation': conversation}


def okapi_map_fn_stage2(example):
    res = example
    #TODO: 修改 llava map fn， 加 assert check length/2
    res = conversation_map_fn(example)
    return res