# Copyright (c) OpenMMLab. All rights reserved.
import re
from typing import List, Dict, Any, Tuple, Union
from xtuner.utils.constants import (
    DEFAULT_IMAGE_TOKEN,
    VISUAL_PROMPT_PLACEHOLDER,
    VISUAL_REFERENCE_TOKEN,
    BOT_TOKEN, EOT_TOKEN, 
    BOV_TOKEN, 
    EOV_TOKEN
    )
from xtuner.dataset.utils import norm_box_xyxy, norm_point_xyxy, de_norm_box_xyxy
from xtuner.utils import IGNORE_INDEX
from .okapi_map_fn import map_obj

# ret = {
#     'image': image,
#     'target': {'boxes': item['boxes']},  # 'seg' /
#     'map_placeholders': dict(
#            input=[BOXES_PLACEHOLDER],
#            output=[BOXES_PLACEHOLDER],
#         )
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


def target_map_fn(example):
    messages = example['conversations']
    map_placeholders = example.get('map_placeholders', None)
    if not map_placeholders:
        return dict()

    if messages[0]['from'] == 'system':
        messages = messages[1:]

    for msg in messages:
        if msg['from'] == 'human':
            input = msg['value']

        elif msg['from'] == 'gpt':
            output = msg['value']

    return dict(
        visual_prompts = '',
        decode_labels = ''
    )

def conversation_map_fn(example):
    messages = example['conversations']
    map_placeholders = example.get('map_placeholders', None)
    input = ''
    systems = None
    conversation = []
    if messages[0]['from'] == 'system':
        systems = messages[0]
        messages = messages[1:]
        assert 0.5 * len(messages) == len(systems['value'])

    sys_idx = 0
    for msg in messages:
        if msg['from'] == 'human':
            if DEFAULT_IMAGE_TOKEN in msg['value']:
                msg['value'] = msg['value'].replace(DEFAULT_IMAGE_TOKEN,
                                                    '').strip()
                msg['value'] = DEFAULT_IMAGE_TOKEN + '\n' + msg['value']
                msg['value'] = msg['value'].strip()
            input += msg['value']

            if map_placeholders:
                input_placeholders = map_placeholders.get('input', [])
                for placeholder in input_placeholders:
                    input = input.replace(placeholder, VISUAL_PROMPT_PLACEHOLDER)

        elif msg['from'] == 'gpt':
            output = msg['value']

            #TODO: vrt 判断，是否需要增加vrt


            if map_placeholders:
                output_placeholders = map_placeholders.get('output', [])
                for placeholder in output_placeholders:
                    #TODO: 判断 placeholder 之前有没有 phrase

                    # 增加括号包裹住 placeholder

                    for chunk in output.split(placeholder):
                        pass

            if systems:
                info = systems['value'][sys_idx]
                task_name = f'- task name: {info['task']['task_name']}\n' 
                if info['task']['use_unit']:
                    info['task']['element'].append('unit')
                    unit = ', '.join(info['unit'])
                    unit = f'- unit: {unit}\n'
                else:
                    unit = ''
                element = ', '.join(info['task']['element'])
                element = f'- answer element: {element}\n'  
                sys = 'Task Command:\n' + task_name + element + unit
                conversation.append({'system': sys, 'input': input, 'output': output})
                sys_idx += 1
            else:
                conversation.append({'input': input, 'output': output})
            input = ''
        else:
            raise NotImplementedError
    return {'conversation': conversation}


def okapi_map_fn_stage2(example):
    messages = example['conversations']
    while messages and messages[0]['from'] == 'gpt':
        # Skip the first one if it is from gpt
        example['conversations'] = example['conversations'][1:]

    res = target_map_fn(example)
    conversation = conversation_map_fn(example)
    res.update(conversation)
    return res