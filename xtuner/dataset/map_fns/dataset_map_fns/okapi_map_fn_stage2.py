# Copyright (c) OpenMMLab. All rights reserved.
import re
import numpy as np
from typing import List, Dict, Any, Tuple, Union
from xtuner.utils.constants import (
    DEFAULT_IMAGE_TOKEN,
    BOXES_PLACEHOLDER,
    MASKS_PLACEHOLDER,
    POINTS_PLACEHOLDER,
    VISUAL_PROMPT_PLACEHOLDER,
    VISUAL_REPRESENTATION_TOKEN,
    VISUAL_REFERENCE_TOKEN,
    BOT_TOKEN, EOT_TOKEN, 
    BOU_TOKEN, EOU_TOKEN,
    BOV_TOKEN, EOV_TOKEN
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

SEQ_MAP = {
    BOXES_PLACEHOLDER: 'boxes_seq',
    MASKS_PLACEHOLDER: 'masks_seq',
    POINTS_PLACEHOLDER: 'points_seq'
}

UNIT_MAP = {
    BOXES_PLACEHOLDER: 'unit',
    MASKS_PLACEHOLDER: 'mask',
    POINTS_PLACEHOLDER: 'point'
}

def flatten_obj(obj_list):
    res = []
    for obj in obj_list:
        if isinstance(obj, list):
            res.extend(flatten_obj(obj))
        else:
            res.append(obj)
    return res


def map_obj(target_value: List[List[float]], target_seq: List[List[int]]) -> List[List[List[float]]]:
    """
    >>> normalized_boxes = [[0.1, 0.1, 0.1, 0.1], [0.2, 0.2, 0.2, 0.2], [0.3, 0.3, 0.3, 0.3]]
    >>> boxes_seq_ = [[3, 1], [2]]
    >>> var = map_obj(normalized_boxes, boxes_seq_)
    >>> assert var == [[[0.3,0.3,0.3,0.3], [0.1,0.1,0.1,0.1]], [0.2,0.2,0.2,0.2]]
    """
    try:
        ret = []
        for targets in target_seq:
            targets_ret = []
            for tgt_idx in targets:
                targets_ret.append(np.array(target_value[tgt_idx]))
            ret.append(targets_ret)
        return ret
    except:
        raise SystemExit(f"error: map obj {target_value} {target_seq}")


def map_units(str, units, placeholder):
    counter = 0

    def _replacement(match):
        nonlocal counter
        unit = units[counter]
        counter += 1
        return unit

    # Use re.sub with a replacement function
    result = re.sub(re.escape(placeholder), _replacement, str)
    return result


def target_map_fn(example):
    target = example['target']
    messages = example['conversations']
    map_placeholders = example.get('map_placeholders', None)
    if not map_placeholders:
        return dict()

    if messages[0]['from'] == 'system':
        messages = messages[1:]

    visual_prompts = {}
    decode_labels = {}
    for msg in messages:
        sentence = msg['value']
        if msg['from'] == 'human':
            visual_prompts.append()
            placeholders = map_placeholders.get('input', [])
        elif msg['from'] == 'gpt':
            decode_labels.append()
            placeholders = map_placeholders.get('output', [])
        else:
            raise NotImplementedError

        tgt_in_msg = dict()
        for placeholder in placeholders:
            pattern = re.compile(placeholder)
            all_find = pattern.findall(sentence)

            tgt_seq = msg.get(SEQ_MAP[placeholder], None)
            tgt_seq = map_obj(target[placeholder], tgt_seq)
            target[placeholder] = tgt_seq

            flat_tgt_seq = flatten_obj(tgt_seq)
            assert len(all_find) == len(flat_tgt_seq), \
                f"placeholder {placeholder} not match. sentence: {sentence}. targets:{flat_tgt_seq}"
            if len(all_find) == 0: continue

            tgt_in_msg[placeholder] = flat_tgt_seq

        if msg['from'] == 'human':
            visual_prompts = tgt_in_msg
        elif msg['from'] == 'gpt':
            decode_labels = tgt_in_msg
        else:
            raise NotImplementedError

    return dict(
        visual_prompts = visual_prompts,
        decode_labels = decode_labels
    )

def conversation_map_fn(example, vrt_len, ref_len):
    messages = example['conversations']

    system_list = []
    if messages[0]['from'] == 'system':
        systems = messages[0]
        messages = messages[1:]
        assert 0.5 * len(messages) == len(systems['value'])

        for info in systems['value']:
            task_name = f"- task name: {info['task']['task_name']}\n"
            if info['task']['use_unit']:
                info['task']['element'].append('unit')
                info['unit'] = [BOU_TOKEN + u + EOU_TOKEN for u in info['unit']]
                unit = ', '.join(info['unit'])
                unit = f'- unit: {unit}\n'
            else:
                unit = ''
            element = ', '.join(info['task']['element'])
            element = f'- answer element: {element}\n'  
            sys = 'Task Command:\n' + task_name + element + unit
            system_list.append(sys)


    map_placeholders = example.get('map_placeholders', None)
    input = ''
    conversation = []
    vrt_exist = False
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

            if map_placeholders:
                output_placeholders = map_placeholders.get('output', [])
                unit_decode = any(output.count(placeholder) > 0 for placeholder in output_placeholders)

                if unit_decode and not vrt_exist:
                    # cot = f"{BOT_TOKEN}Unit needs to be decoded, but the visual representation \\
                    #     has not been generated. Now, let's generate it.{EOT_TOKEN}"
                    cot = f"{BOT_TOKEN}Unit decode (True). VPT prepared (False). Generate VPT (True).{EOT_TOKEN}"
                    vrt = f"{BOV_TOKEN}{VISUAL_REPRESENTATION_TOKEN * vrt_len}{EOV_TOKEN}"
                    
                elif unit_decode and vrt_exist:
                    # cot = f"{BOT_TOKEN}Unit needs to be decoded, and the visual representation \\
                    #     has been generated. Now, let's generate it.{EOT_TOKEN}"
                    cot = f"{BOT_TOKEN}Unit decode (True). VPT prepared (True). Generate VPT (False).{EOT_TOKEN}"
                    vrt = ''
                else:
                    # cot = f"{BOT_TOKEN}Unit needs not to be decoded, skip visual representation process.{EOT_TOKEN}"
                    cot = f"{BOT_TOKEN}Unit decode (False). Generate VPT (False).{EOT_TOKEN}"
                    vrt = ''
                output = cot + vrt + output
                
                for placeholder in output_placeholders:
                    target_seq = msg[SEQ_MAP[placeholder]]
                    
                    all_units = []
                    for tgts in target_seq:
                        units = []
                        for tgt_idx, tgt in enumerate(tgts):
                            unit = BOU_TOKEN + UNIT_MAP[placeholder] + EOU_TOKEN + \
                                VISUAL_REFERENCE_TOKEN * ref_len + f"[{tgt_idx}]"
                            units.append(unit)
                        units[0] = '(' + units[0]
                        units[-1] = units[-1] + ')'
                        all_units.extend(units)

                    output = map_units(output, all_units, placeholder)

            conversation.append({'input': input, 'output': output})
            input = ''
        else:
            raise NotImplementedError
        
        assert len(conversation) == len(system_list)
        res_conversation = []
        for conv, sys in zip(conversation, system_list):
            conv['system'] = sys
            res_conversation.append(conv)

    return {'conversation': res_conversation}


def okapi_map_fn_stage2(example):
    messages = example['conversations']
    while messages and messages[0]['from'] == 'gpt':
        # Skip the first one if it is from gpt
        example['conversations'] = example['conversations'][1:]

    res = target_map_fn(example)
    conversation = conversation_map_fn(example)
    res.update(conversation)
    return res