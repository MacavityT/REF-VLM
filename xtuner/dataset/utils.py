# Copyright (c) OpenMMLab. All rights reserved.
import base64
import copy
import io
from io import BytesIO
from itertools import chain

import torch
import numpy as np
import requests
from PIL import Image

from xtuner.utils import DEFAULT_IMAGE_TOKEN, IGNORE_INDEX, IMAGE_TOKEN_INDEX
from xtuner.utils import VISUAL_PROMPT_PLACEHOLDER ,VISUAL_PROMPT_INDEX

import cv2
import mmengine.fileio as fileio
from mmengine.utils import is_str
from cv2 import (IMREAD_COLOR, IMREAD_GRAYSCALE, IMREAD_IGNORE_ORIENTATION,
                 IMREAD_UNCHANGED)
imread_flags = {
    'color': IMREAD_COLOR,
    'grayscale': IMREAD_GRAYSCALE,
    'unchanged': IMREAD_UNCHANGED,
    'color_ignore_orientation': IMREAD_IGNORE_ORIENTATION | IMREAD_COLOR,
    'grayscale_ignore_orientation':
    IMREAD_IGNORE_ORIENTATION | IMREAD_GRAYSCALE
} 

def get_bos_eos_token_ids(tokenizer):
    if tokenizer.__class__.__name__ in [
            'QWenTokenizer', 'QWen2Tokenizer', 'Qwen2TokenizerFast'
    ]:
        bos_token_id = []
        eos_token_id = tokenizer.eos_token_id
        assert eos_token_id is not None, \
            'Please set eos_token for Qwen tokenizer!'
    elif tokenizer.__class__.__name__ == 'ChatGLMTokenizer':
        bos_token_id = [64790, 64792]
        eos_token_id = tokenizer.eos_token_id
    else:
        bos_token_id = tokenizer.bos_token_id
        eos_token_id = tokenizer.eos_token_id
    if isinstance(bos_token_id, int):
        bos_token_id = [bos_token_id]
    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    return bos_token_id, eos_token_id


def encode_fn(example,
              tokenizer,
              max_length,
              input_ids_with_output=True,
              with_image_token=False):
    def get_vpt_slices():
        pass

    """We only support the following three scenarios:

    1. Incremental pretraining dataset.
        example['conversation'] = [
                {
                    'input': '',
                    'output': '### Human: Can you write xxx'
                }
            ]

    2. Single-turn conversation dataset.
        example['conversation'] = [
                {
                    'input': 'Give three tips for staying healthy.',
                    'output': '1.Eat a balanced diet xxx'
                }
            ]

    3. Multi-turn conversation dataset.
        example['conversation'] = [
                {
                    'input': 'Give three tips for staying healthy.',
                    'output': '1.Eat a balanced diet xxx'
                },
                {
                    'input': 'Please expand on the second point.',
                    'output': 'Here is an expanded explanation of the xxx'
                }
            ]
    """
    bos_token_id, eos_token_id = get_bos_eos_token_ids(tokenizer)
    is_multi_turn_conversation = len(example['conversation']) > 1
    if is_multi_turn_conversation:
        assert input_ids_with_output

    input_ids, labels = [], []
    next_needs_bos_token = True
    for single_turn_conversation in example['conversation']:
        input = single_turn_conversation['input']
        if DEFAULT_IMAGE_TOKEN in input and with_image_token:
            chunk_encode = []
            for chunk_image in input.split(DEFAULT_IMAGE_TOKEN):
                if VISUAL_PROMPT_PLACEHOLDER in chunk_image:
                    chunk_vpt = [
                            tokenizer.encode(chunk, add_special_tokens=False)
                            for chunk in chunk_image.split(VISUAL_PROMPT_PLACEHOLDER)
                    ]
                else:
                    chunk_vpt = tokenizer.encode(chunk_image, add_special_tokens=False)
                
                chunk_encode.append(chunk_vpt)

            assert len(chunk_encode) == 2
            input_encode = []
            for idx_img, cur_chunk_encode in enumerate(chunk_encode):
                if isinstance(cur_chunk_encode[0], list):
                    for idx_vpt, cur_chunk_encode_vpt in enumerate(cur_chunk_encode):
                        input_encode.extend(cur_chunk_encode_vpt)
                        if idx_vpt != len(cur_chunk_encode) - 1:
                            input_encode.append(VISUAL_PROMPT_INDEX)
                    
                else:
                    input_encode.extend(cur_chunk_encode)
                
                if idx_img != len(chunk_encode) - 1:
                    input_encode.append(IMAGE_TOKEN_INDEX)
        else:
            if VISUAL_PROMPT_PLACEHOLDER in input:
                chunk_encode = [
                        tokenizer.encode(chunk, add_special_tokens=False)
                        for chunk in input.split(VISUAL_PROMPT_PLACEHOLDER)
                ]
                input_encode = []
                for idx, cur_chunk_encode in enumerate(chunk_encode):
                    input_encode.extend(cur_chunk_encode)
                    if idx != len(chunk_encode) - 1:
                        input_encode.append(VISUAL_PROMPT_INDEX)
            else:
                input_encode = tokenizer.encode(input, add_special_tokens=False)
        
        if next_needs_bos_token:
            input_ids += bos_token_id
            labels += [IGNORE_INDEX] * len(bos_token_id)
        input_ids += input_encode
        labels += [IGNORE_INDEX] * len(input_encode)
        if input_ids_with_output:
            # Add output
            output_with_loss = single_turn_conversation.get(
                'output_with_loss', True)
            output = single_turn_conversation['output']
            output_encode = tokenizer.encode(output, add_special_tokens=False)
            input_ids += output_encode
            if output_with_loss:
                labels += copy.deepcopy(output_encode)
            else:
                labels += [IGNORE_INDEX] * len(output_encode)
            # Add EOS_TOKEN (with loss)
            if single_turn_conversation.get('need_eos_token', True):
                next_needs_bos_token = True
                input_ids += eos_token_id
                if output_with_loss:
                    labels += copy.deepcopy(eos_token_id)
                else:
                    labels += [IGNORE_INDEX] * len(eos_token_id)
            else:
                next_needs_bos_token = False
            # Add SEP (without loss)
            sep = single_turn_conversation.get('sep', '')
            if sep != '':
                sep_encode = tokenizer.encode(sep, add_special_tokens=False)
                input_ids += sep_encode
                labels += [IGNORE_INDEX] * len(sep_encode)

    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        labels = labels[:max_length]
    return {'input_ids': input_ids, 'labels': labels}


class Packer:
    """Pack multiple pieces of data into one."""

    def __init__(self,
                 chunk_size=2048,
                 use_varlen_attn=False,
                 drop_last=False):
        self.chunk_size = chunk_size
        self.residual = {'input_ids': [], 'labels': []}
        self.use_varlen_attn = use_varlen_attn
        self.drop_last = drop_last
        if use_varlen_attn:
            self.residual_cumulative_len = [0]

    def get_cumulative_len(self, chunk_num):
        ptr_l = 0
        cumulative_len = []
        for chunk_idx in range(chunk_num):
            length_train = (chunk_idx + 1) * self.chunk_size
            ptr_r = np.searchsorted(
                self.residual_cumulative_len, length_train, side='left')
            if self.residual_cumulative_len[ptr_r] == length_train:
                cumulative_len_cur = \
                    self.residual_cumulative_len[ptr_l:ptr_r + 1]
                ptr_l = ptr_r + 1
            else:
                cumulative_len_cur = self.residual_cumulative_len[
                    ptr_l:ptr_r] + [length_train]
                ptr_l = ptr_r
            cumulative_len_cur = [
                num - chunk_idx * self.chunk_size for num in cumulative_len_cur
            ]
            if cumulative_len_cur[0] != 0:
                cumulative_len_cur = [0] + cumulative_len_cur

            cumulative_len.append(cumulative_len_cur)

        self.residual_cumulative_len = [
            num - length_train for num in self.residual_cumulative_len[ptr_l:]
        ]
        if len(self.residual_cumulative_len) == 0:
            self.residual_cumulative_len = [0]
        elif self.residual_cumulative_len[0] != 0:
            self.residual_cumulative_len = [0] + self.residual_cumulative_len

        return cumulative_len

    def get_position_ids(self, cumulative_len):
        position_ids = []
        for cumulative_len_cur in cumulative_len:
            index_cur = []
            for i in range(len(cumulative_len_cur) - 1):
                index_cur.extend(
                    list(
                        range(cumulative_len_cur[i + 1] -  # noqa: W504
                              cumulative_len_cur[i])))
            position_ids.append(index_cur)
        return position_ids

    def __call__(self, batch):
        concatenated_samples = {
            k: v + list(chain(*batch[k]))
            for k, v in self.residual.items()
        }

        if self.use_varlen_attn:
            for input_id in batch['input_ids']:
                self.residual_cumulative_len.append(
                    self.residual_cumulative_len[-1] + len(input_id))

        total_length = len(concatenated_samples[list(
            concatenated_samples.keys())[0]])

        if total_length >= self.chunk_size:
            chunk_num = total_length // self.chunk_size
            result = {
                k: [
                    v[i:i + self.chunk_size] for i in range(
                        0,
                        chunk_num *  # noqa: W504
                        self.chunk_size,
                        self.chunk_size)
                ]
                for k, v in concatenated_samples.items()
            }
            self.residual = {
                k: v[(chunk_num * self.chunk_size):]
                for k, v in concatenated_samples.items()
            }

            if self.use_varlen_attn:
                cumulative_len = self.get_cumulative_len(chunk_num)
                result['cumulative_len'] = cumulative_len
                result['position_ids'] = self.get_position_ids(cumulative_len)
        else:
            if self.drop_last:
                result = {k: [] for k, v in concatenated_samples.items()}
            else:
                result = {k: [v] for k, v in concatenated_samples.items()}

            self.residual = {k: [] for k in concatenated_samples.keys()}

            if self.use_varlen_attn:
                result['cumulative_len'] = [] if self.drop_last else [
                    self.residual_cumulative_len
                ]
                result['position_ids'] = [] if self.drop_last \
                    else self.get_position_ids([self.residual_cumulative_len])
                self.residual_cumulative_len = [0]

        return result


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result

def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def decode_base64_to_image(base64_string):
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data))
    return image


def imfrombytes(filename,
                flag: str = 'color',
                channel_order: str = 'bgr') -> np.ndarray:
    img_bytes = fileio.get(filename)
    img_np = np.frombuffer(img_bytes, np.uint8)
    flag = imread_flags[flag] if is_str(flag) else flag
    img = cv2.imdecode(img_np, flag)
    if flag == IMREAD_COLOR and channel_order == 'rgb':
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
    return img


# for boxes format
def _box_xyxy_expand2square(box, w, h):
    if w == h:
        return box
    if w > h:
        x1, y1, x2, y2 = box
        y1 += (w - h) // 2
        y2 += (w - h) // 2
        box = x1, y1, x2, y2
        return box
    assert w < h
    x1, y1, x2, y2 = box
    x1 += (h - w) // 2
    x2 += (h - w) // 2
    box = x1, y1, x2, y2
    return box

def mask_transform(mask, image_processor):
    result = np.expand_dims(mask, axis=2)
    result = np.repeat(result, 3, axis=2)
    result = image_processor.preprocess(
        result,
        do_rescale=False,
        do_normalize=False)['pixel_values'][0]
    result = result[0, ...]
    return result

def _mask_expand2square(mask):
    height, width = mask.shape
    if width == height:
        result = mask
    elif width > height:
        result = np.zeros((width, width))
        start = (width - height) // 2
        stop = start + height
        result[start:stop, :] = mask
    else:
        start = (height - width) // 2
        stop = start + width
        result = np.zeros((height, height))
        result[:, start:stop] = mask
    return result

def _point_xy_expand2square(point, w, h):
    pseudo_box = (point[0], point[1], point[0], point[1])
    expanded_box = _box_xyxy_expand2square(box=pseudo_box, w=w, h=h)
    expanded_point = (expanded_box[0], expanded_box[1])
    return expanded_point

def boxes_xyxy_expand2square(bboxes, width, height):
    expanded_bboxes = [_box_xyxy_expand2square(bbox, w=width, h=height) for bbox in bboxes]
    return expanded_bboxes

def points_xy_expand2square(points, width, height):
    expanded_points = [_point_xy_expand2square(point, w=width, h=height) for point in points]
    return expanded_points

def masks_expand2square(masks):
    expanded_masks = [_mask_expand2square(mask) for mask in masks]
    return expanded_masks


def de_norm_box_xyxy(box, w, h):
    x1, y1, x2, y2 = box
    x1 = x1 * w
    x2 = x2 * w
    y1 = y1 * h
    y2 = y2 * h
    box = x1, y1, x2, y2
    return box


def box_xywh_to_xyxy(box, w=None, h=None):
    x, y, bw, bh = box
    x2 = x + bw
    y2 = y + bh
    if w is not None:
        x2 = min(x2, w)
    if h is not None:
        y2 = min(y2, h)
    box = x, y, x2, y2
    return box


def norm_box_xyxy(box, w, h):
    x1, y1, x2, y2 = box

    # Calculate the normalized coordinates with min-max clamping
    norm_x1 = max(0.0, min(x1 / w, 1.0))
    norm_y1 = max(0.0, min(y1 / h, 1.0))
    norm_x2 = max(0.0, min(x2 / w, 1.0))
    norm_y2 = max(0.0, min(y2 / h, 1.0))

    # Return the normalized box coordinates
    normalized_box = (round(norm_x1, 3), round(norm_y1, 3), round(norm_x2, 3), round(norm_y2, 3))
    return normalized_box


def norm_point_xyxy(point, w, h):
    x, y = point
    norm_x = max(0.0, min(x / w, 1.0))
    norm_y = max(0.0, min(y / h, 1.0))
    point = norm_x, norm_y
    return point


def bbox2mask(box, width, height):
    mask = np.zeros((height, width))
    x1, y1, x2, y2 = box
    mask[int(x1):int(x2), int(y1):int(y2)] = 1
    return mask

def point2mask(point, radius, height, width):
    mask = np.zeros((height, width))
    center_x, center_y, _, _ = point
    center_x = int(center_x)
    center_y = int(center_y)

    y, x = np.ogrid[0:height, 0:width]
    mask = (x-center_x)**2 + (y-center_y)**2 <= radius**2
    return mask.astype(int)

def scribble2mask(scribble, radus):
    pass