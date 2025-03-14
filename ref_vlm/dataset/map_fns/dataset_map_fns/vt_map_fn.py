# Copyright (c) OpenMMLab. All rights reserved.
import re
from typing import List, Dict, Any, Tuple, Union
from xtuner.utils import IGNORE_INDEX
from xtuner.dataset.map_fns import llava_map_fn
from ref_vlm.utils.constants import (
    BOXES_PLACEHOLDER, 
    POINTS_PLACEHOLDER, 
    MASKS_PLACEHOLDER,
    DEFAULT_IMAGE_TOKEN
    )

Box = List[Union[float, int]]
Boxes = List[Box]
BoxesSeq = List[Boxes]

small_brackets_pat = re.compile(r'\(\d(?:\.\d*)?(?:,\d(?:\.\d*)?){3}(?:;\d(?:\.\d*)?(?:,\d(?:\.\d*)?){3})*\)')
small_brackets_point_pat = re.compile(r'\(\d(?:\.\d*)?(?:,\d(?:\.\d*)?)(?:;\d(?:\.\d*)?(?:,\d(?:\.\d*)?))*\)')

middle_brackets_pat = re.compile(r'\[\d(?:\.\d*)?(?:,\d(?:\.\d*)?){3}(?:;\d(?:\.\d*)?(?:,\d(?:\.\d*)?){3})*\]')
middle_brackets_point_pat = re.compile(r'\[\d(?:\.\d*)?(?:,\d(?:\.\d*)?)(?:;\d(?:\.\d*)?(?:,\d(?:\.\d*)?))*\]')

# extract points from language, but not useful for us
def extract_point(string: str, use_small_brackets = False) -> List[Boxes]:
    point_pat = small_brackets_point_pat if use_small_brackets else middle_brackets_point_pat
    """ balabala<boxes>balabala<boxes> -> [boxes, boxes] """
    ret = []
    for bboxes_str in point_pat.findall(string):
        bboxes = []
        bbox_strs = bboxes_str.replace("(", "").replace(")", "").replace("[", "").replace("]", "").split(";")
        for bbox_str in bbox_strs:
            bbox = list(map(float, bbox_str.split(',')))
            bboxes.append(bbox)
        ret.append(bboxes)
    return ret

# extract boxes from language, but not useful for us
def extract(string: str, use_small_brackets = False) -> List[Boxes]:
    """ balabala<boxes>balabala<boxes> -> [boxes, boxes] """
    pat = small_brackets_pat if use_small_brackets else middle_brackets_pat
    ret = []
    for bboxes_str in pat.findall(string):
        bboxes = []
        bbox_strs = bboxes_str.replace("(", "").replace(")", "").replace("[", "").replace("]", "").split(";")
        for bbox_str in bbox_strs:
            bbox = list(map(float, bbox_str.split(',')))
            bboxes.append(bbox)
        ret.append(bboxes)
    return ret

def map_obj(boxes_value: List[List[float]], boxes_seq: List[List[int]]) -> List[List[List[float]]]:
    """
    >>> normalized_boxes = [[0.1, 0.1, 0.1, 0.1], [0.2, 0.2, 0.2, 0.2], [0.3, 0.3, 0.3, 0.3]]
    >>> boxes_seq_ = [[3, 1], [2]]
    >>> var = map_obj(normalized_boxes, boxes_seq_)
    >>> assert var == [[[0.3,0.3,0.3,0.3], [0.1,0.1,0.1,0.1]], [0.2,0.2,0.2,0.2]]
    """
    try:
        ret = []
        for boxes in boxes_seq:
            boxes_ret = []
            for box_index in boxes:
                if isinstance(box_index, (list, tuple)):
                    boxes_ret.append(boxes_value[box_index[0]][box_index[1]])
                else:
                    boxes_ret.append(boxes_value[box_index])
            ret.append(boxes_ret)
        return ret
    except:
        raise SystemExit(f"error: map obj {boxes_value} {boxes_seq}")

def format_box_or_points(boxes: Boxes, 
                         precision = 3, 
                         use_small_brackets = False) -> str:
    box_strs = []
    for box in boxes:
        box_strs.append(','.join([f"{elem:.{precision}f}" for elem in box]))
    box_str = ';'.join(box_strs)
    if use_small_brackets:
        return "(" + box_str + ")"
    return "[" + box_str + "]"

def box_map_fn(example):
    if 'target' not in example.keys(): 
        return
    bboxes_token_pat = re.compile(BOXES_PLACEHOLDER)
    target = example['target']

    # convert bboxes_seq
    messages = example['conversations']
    for sentence in messages:
        words: str = sentence['value']
        boxes_seq: List[List[int]] = sentence.get('boxes_seq', None)
        if boxes_seq is not None:
            # map box seq
            boxes_seq: List[Boxes] = map_obj(target['boxes'], boxes_seq)
            # reformat; replace <boxes> placeholder
            all_boxes = bboxes_token_pat.findall(words)
            assert len(all_boxes) == len(boxes_seq), f"not match. sentence: {words}. boxes:{boxes_seq}"
            if len(all_boxes) == 0:
                continue
            bboxes_strs = [format_box_or_points(boxes) for boxes in boxes_seq]
            converted = words.replace(BOXES_PLACEHOLDER, '{}').format(*bboxes_strs)
            # sentence['raw_value'] = sentence['value']
            sentence['value'] = converted

def point_map_fn(example):
    if 'target' not in example.keys(): 
        return
    
    points_token_pat = re.compile(POINTS_PLACEHOLDER)
    target = example['target']

    messages = example['conversations']
    for sentence in messages:
        words: str = sentence['value']
        points_seq: List[List[int]] = sentence.get('points_seq', None)
        if points_seq is not None:
            # map point seq
            points_seq: List[Boxes] = map_obj(target['points'], points_seq)
            # reformat; replace <points> placeholder
            all_points = points_token_pat.findall(words)
            assert len(all_points) == len(points_seq), f"not match. sentence: {words}. points:{points_seq}"
            if len(all_points) == 0:
                continue
            points_strs = [format_box_or_points(points) for points in points_seq]
            converted = words.replace(POINTS_PLACEHOLDER, '{}').format(*points_strs)        
            # sentence['raw_value'] = sentence['value']
            sentence['value'] = converted

def vt_map_fn(example):
    box_map_fn(example)
    point_map_fn(example)
    res = llava_map_fn(example)
    return res