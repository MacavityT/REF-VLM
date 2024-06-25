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
    BOV_TOKEN, EOV_TOKEN,
    PHRASE_ST_PLACEHOLDER_STAGE2,
    PHRASE_ED_PLACEHOLDER_STAGE2
    )

SEQ_MAP = {
    BOXES_PLACEHOLDER: 'boxes_seq',
    MASKS_PLACEHOLDER: 'masks_seq',
    POINTS_PLACEHOLDER: 'points_seq'
}

TGT_KEY_MAP = {
    BOXES_PLACEHOLDER: 'boxes',
    MASKS_PLACEHOLDER: 'masks',
    POINTS_PLACEHOLDER: 'points'
}

UNIT_MAP = {
    BOXES_PLACEHOLDER: 'box',
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
                if target_value[tgt_idx] is not None:
                    targets_ret.append(np.array(target_value[tgt_idx]))
                else:
                    targets_ret.append(None)
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

def get_placeholders_order(string, placeholders):
    positions = []
    for placeholders in placeholders:
        for match in re.finditer(re.escape(placeholders), string):
            positions.append((match.start(), placeholders))
    
    positions.sort()
    ordered_placeholders = [placeholder for _, placeholder in positions]
    return ordered_placeholders

def target_map_fn(example):
    if 'target' not in example.keys():
        return {}
    
    target = example['target']
    messages = example['conversations']
    map_placeholders = example.get('map_placeholders', None)
    if not map_placeholders:
        return dict()

    if messages[0]['from'] == 'system':
        messages = messages[1:]

    visual_prompts = []
    decode_labels = []
    for msg in messages:
        sentence = msg['value']
        if msg['from'] == 'human':
            placeholders = map_placeholders.get('input', [])
        elif msg['from'] == 'gpt':
            placeholders = map_placeholders.get('output', [])
        else:
            raise NotImplementedError

        tgt_in_msg = dict()
        for placeholder in placeholders:
            pattern = re.compile(placeholder)
            all_find = pattern.findall(sentence)

            tgt_seq = msg.get(SEQ_MAP[placeholder], None)
            if not tgt_seq: continue
            
            tgt_value = target.get(TGT_KEY_MAP[placeholder], None)
            mapped_tgt_seq = map_obj(tgt_value, tgt_seq)
            flat_tgt_seq = flatten_obj(mapped_tgt_seq)
            assert len(all_find) == len(flat_tgt_seq), \
                f"placeholder {placeholder} not match. sentence: {sentence}. num targets:{len(flat_tgt_seq)}"
            if len(all_find) == 0: continue
            tgt_in_msg[placeholder] = flat_tgt_seq

        items = []
        ordered_placeholders = get_placeholders_order(sentence, placeholders)
        for placeholder in ordered_placeholders:
            value = tgt_in_msg[placeholder].pop(0)
            items.append(
                dict(
                    type = UNIT_MAP[placeholder],
                    value = value
                )
            )
        if msg['from'] == 'human':
            visual_prompts.append(items if len(items) > 0 else None)
        elif msg['from'] == 'gpt':
            decode_labels.append(items if len(items) > 0 else None)

    result = dict()
    if len(visual_prompts) > 0 \
        and any(vpt is not None for vpt in visual_prompts):
        result['visual_prompts'] = visual_prompts
    if len('decode_labels') > 0 \
        and any(label is not None for label in decode_labels):
        result['decode_labels'] = decode_labels
    return result

def conversation_map_fn(example, vrt_len=64, ref_len=1):
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
                unit = '- unit: None'
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
                vrt = f"{BOV_TOKEN}{VISUAL_REPRESENTATION_TOKEN * vrt_len}{EOV_TOKEN}\n"
                msg['value'] = DEFAULT_IMAGE_TOKEN + '\n' + vrt + msg['value']
                msg['value'] = msg['value'].strip()
            input += msg['value']

            if map_placeholders:
                input_placeholders = map_placeholders.get('input', [])
                for placeholder in input_placeholders:
                    input = input.replace(placeholder, VISUAL_PROMPT_PLACEHOLDER)

        elif msg['from'] == 'gpt':
            output = msg['value']

            # #region old cot
            # if map_placeholders:
            #     output_placeholders = map_placeholders.get('output', [])
            #     unit_decode = any(output.count(placeholder) > 0 for placeholder in output_placeholders)

            #     if unit_decode and not vrt_exist:
            #         # cot = f"{BOT_TOKEN}Unit needs to be decoded, but the visual representation \\
            #         #     has not been generated. Now, let's generate it.{EOT_TOKEN}"
            #         cot = f"{BOT_TOKEN}Unit decode (True). VRT prepared (False). Generate VRT (True).{EOT_TOKEN}"
            #         vrt = f"{BOV_TOKEN}{VISUAL_REPRESENTATION_TOKEN * vrt_len}{EOV_TOKEN}"
            #         vrt_exist = True
                    
            #     elif unit_decode and vrt_exist:
            #         # cot = f"{BOT_TOKEN}Unit needs to be decoded, and the visual representation \\
            #         #     has been generated. Now, let's generate it.{EOT_TOKEN}"
            #         cot = f"{BOT_TOKEN}Unit decode (True). VRT prepared (True). Generate VRT (False).{EOT_TOKEN}"
            #         vrt = ''
            #     else:
            #         # cot = f"{BOT_TOKEN}Unit needs not to be decoded, skip visual representation process.{EOT_TOKEN}"
            #         cot = f"{BOT_TOKEN}Unit decode (False). Generate VRT (False).{EOT_TOKEN}"
            #         vrt = ''
            #     output = cot + vrt + output
                
            #     for placeholder in output_placeholders:
            #         target_seq = msg.get(SEQ_MAP[placeholder], None)
            #         if not target_seq: continue
                    
            #         units = []
            #         for tgts in target_seq:
            #             for tgt_idx, tgt in enumerate(tgts):
            #                 unit = BOU_TOKEN + UNIT_MAP[placeholder] + EOU_TOKEN + \
            #                     VISUAL_REFERENCE_TOKEN * ref_len + f"[{tgt_idx}]"
            #                 if tgt_idx == 0: unit = '(' + unit
            #                 if tgt_idx == len(tgts) - 1: unit = unit + ')'
            #                 if tgt_idx != 0 and len(tgts) > 1: unit = ', ' + unit
            #                 units.append(unit)
            #         output = map_units(output, units, placeholder)
            # # for map placeholder is None
            # else:
            #     cot = f"{BOT_TOKEN}Unit decode (False). Generate VRT (False).{EOT_TOKEN}"
            #     output = cot + output
            # #endregion

            #region new cot
            if map_placeholders:
                output_placeholders = map_placeholders.get('output', [])
                unit_decode = any(output.count(placeholder) > 0 for placeholder in output_placeholders)
                if unit_decode:
                    p_pattern = PHRASE_ST_PLACEHOLDER_STAGE2 + r'.*?' + PHRASE_ED_PLACEHOLDER_STAGE2
                    p_matches = re.findall(p_pattern, output)
                    splits = re.split(p_pattern, output)[1:]
                    assert len(splits) == len(p_matches)
                    
                    u_counts = []
                    u_names = []
                    for split in splits:
                        counts = [split.count(placeholder) for placeholder in output_placeholders]
                        idx_nonzero = [idx for idx, num in enumerate(counts) if num != 0]
                        assert len(idx_nonzero) == 1
                        idx_placeholder = idx_nonzero[0]
                        u_names.append(output_placeholders[idx_placeholder]) 
                        u_counts.append(counts[idx_placeholder])
                    
                    cot_content = ''
                    for cls_name, unit_name, tgt_num in zip(p_matches, u_names, u_counts):
                        cot_content += f'- Name: {cls_name} Unit: { BOU_TOKEN + UNIT_MAP[unit_name] + EOU_TOKEN} Num: {tgt_num}\n'
                    
                    cot = f"{BOT_TOKEN}\nUnit decode (True). Class name, target unit and number:\n{cot_content}{EOT_TOKEN}\n"

                else:
                    cot = f"{BOT_TOKEN}\nUnit decode (False).\n{EOT_TOKEN}\n"
                
                for placeholder in output_placeholders:
                    target_seq = msg.get(SEQ_MAP[placeholder], None)
                    if not target_seq: continue
                    
                    units = []
                    for tgts in target_seq:
                        for tgt_idx, tgt in enumerate(tgts):
                            unit = f"[{tgt_idx}]" + VISUAL_REFERENCE_TOKEN * ref_len
                            if tgt_idx == 0: unit = '(' + BOU_TOKEN + UNIT_MAP[placeholder] + EOU_TOKEN + unit
                            if tgt_idx == len(tgts) - 1: unit = unit + ')'
                            # if tgt_idx != 0 and len(tgts) > 1: unit = ', ' + unit
                            units.append(unit)
                    output = map_units(output, units, placeholder)
            # for map placeholder is None
            else:
                cot = f"{BOT_TOKEN}\nUnit decode (False).\n{EOT_TOKEN}\n"
            output = cot + output
            #endregion

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


def okapi_map_fn_stage2(example, vrt_len=256, ref_len=1):
    messages = example['conversations']
    while messages and messages[0]['from'] == 'gpt':
        # Skip the first one if it is from gpt
        example['conversations'] = example['conversations'][1:]

    res = target_map_fn(example)
    conversation = conversation_map_fn(example, vrt_len, ref_len)
    res.update(conversation)
    return res