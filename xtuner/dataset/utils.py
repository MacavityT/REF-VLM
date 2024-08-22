# Copyright (c) OpenMMLab. All rights reserved.
import base64
import copy
import io
from io import BytesIO
from itertools import chain

import torch
import torch.nn.functional as F
import numpy as np
import requests
import random
from PIL import Image

from xtuner.utils import DEFAULT_IMAGE_TOKEN, IGNORE_INDEX, IMAGE_TOKEN_INDEX
from xtuner.utils import VISUAL_PROMPT_PLACEHOLDER ,VISUAL_PROMPT_INDEX
import matplotlib.pyplot as plt
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
              with_image_token=False,
              visual_prompts=None):
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
    for idx_trun, single_turn_conversation in enumerate(example['conversation']):
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

        # vpt check
        if VISUAL_PROMPT_INDEX in input_encode:
            vpt_num = input_encode.count(VISUAL_PROMPT_INDEX)
            if visual_prompts is None:
                vpt_length = 0
            else:
                if visual_prompts[idx_trun] is None:
                    vpt_length = 0
                else:
                    vpt_length = len(visual_prompts[idx_trun])
            assert vpt_num == vpt_length, f"vpt_num:{vpt_num} must equal to vpt_length:{vpt_length}"

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

def get_pixel_mask(image, origin_width, origin_height):
    height, width = image.shape[-2:]
    if height != width:
        return torch.ones((height, width)).to(torch.bool)

    mask = torch.zeros((height, width)).to(torch.bool)
    if origin_width == origin_height:
        return mask
    if origin_width > origin_height:
        ratio = origin_height / origin_width
        real_height = width * ratio
        top = int((height - real_height) // 2)
        bottom = int(top + real_height)
        mask[top:bottom, :] = True
    else:
        ratio = origin_width / origin_height
        real_width = height * ratio
        left = int((width - real_width) // 2)
        right = int(left + real_width)
        mask[:, left:right] = True
    return mask

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
    if mask is None:
        return None
    result = np.expand_dims(mask, axis=2)
    result = np.repeat(result, 3, axis=2)
    result = image_processor.preprocess(
        result,
        do_rescale=False,
        do_normalize=False)['pixel_values'][0]
    result = result[0, ...]
    return result

def _mask_expand2square(mask):
    if mask is None:
        return None
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

def mask_square2origin(mask, origin_width, origin_height):
    target_size = max(origin_width, origin_height)
    
    mask = F.interpolate(
        mask, 
        size=(target_size, target_size), 
        mode='bilinear', 
        align_corners=False
    )
    if origin_width == origin_height:
        return mask

    if origin_width > origin_height:
        top = (origin_width - origin_height) // 2
        bottom = top + origin_height
        result = mask[top:bottom, :]
    else:
        left = (origin_height - origin_width) // 2
        right = left + origin_width
        result = mask[:, left:right]
    return result

def box_xywh_to_xyxy(box, w=None, h=None):
    x_center, y_center, bw, bh = box
    x1 = x_center - bw * 0.5
    y1 = y_center - bh * 0.5
    x2 = x_center + bw * 0.5
    y2 = y_center + bh * 0.5
    if w is not None:
        x2 = min(x2, w)
    if h is not None:
        y2 = min(y2, h)
    box = (round(x1, 3), round(y1, 3), round(x2, 3), round(y2, 3))
    return box

def box_xyxy_to_xywh(box):
    x1, y1, x2, y2 = box
    bw = x2 - x1
    bh = y2 - y1
    x_center = (x1 + x2) / 2
    y_center = (y1 + y2) / 2
    box = (round(x_center, 3), round(y_center, 3), round(bw, 3), round(bh, 3))
    return box

def de_norm_box_xyxy(box, w, h):
    x1, y1, x2, y2 = box
    x1 = x1 * w
    x2 = x2 * w
    y1 = y1 * h
    y2 = y2 * h
    box = x1, y1, x2, y2
    return box

def de_norm_box_xywh(box, w, h):
    box = box_xywh_to_xyxy(box, w, h)
    box = de_norm_box_xyxy(box, w, h)
    return box

def de_norm_box_xyxy_square2origin(box, origin_width, origin_height):
    if origin_width == origin_height:
        return box

    if origin_width > origin_height:
        x1, y1, x2, y2 = [i * origin_width for i in box]
        y1 -= (origin_width - origin_height) // 2
        y2 -= (origin_width - origin_height) // 2
    else:
        x1, y1, x2, y2 = [i * origin_height for i in box]
        x1 -= (origin_height - origin_width) // 2
        x2 -= (origin_height - origin_width) // 2

    box = x1, y1, x2, y2
    return box

def denorm_box_xywh_square2origin(box, origin_width, origin_height):
    box = box_xywh_to_xyxy(box)
    box = de_norm_box_xyxy_square2origin(box, origin_width, origin_height)
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

def norm_box_xywh(box, w, h):
    box = box_xywh_to_xyxy(box, w, h)
    normalized_box = norm_box_xyxy(box, w, h)
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
    mask[int(y1):int(y2), int(x1):int(x2)] = 1
    return mask

def point2mask(point, radius, height, width):
    mask = np.zeros((height, width))
    center_x, center_y, _, _ = point
    center_x = int(center_x)
    center_y = int(center_y)

    y, x = np.ogrid[0:height, 0:width]
    mask = (x-center_x)**2 + (y-center_y)**2 <= radius**2
    return mask.astype(int)

def visualize_mask(image, masks, alpha=0.5, beta=1.0):
    if isinstance(image, Image.Image):
        image = np.array(image)
    if not isinstance(masks, list):
        masks = [masks]

    for mask in masks:
        assert mask.shape == image.shape[:-1]
        mask = mask * 255
        mask = mask.astype(np.uint8)
        random_color = [random.randint(0, 255) for _ in range(3)]
        colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        for i in range(3):
            colored_mask[:, :, i] = mask * (random_color[i] / 255.0)

        alpha_channel = np.ones_like(mask) * int(alpha * 255)
        colored_mask = cv2.merge([colored_mask, alpha_channel])

        if image.shape[2] == 3:
            b, g, r = cv2.split(image)
            alpha_channel = np.ones(b.shape, dtype=b.dtype) * 255
            image = cv2.merge([b, g, r, alpha_channel])
    
        image = cv2.addWeighted(image, beta, colored_mask, alpha, 0)

    return image


def visualize_mask_single(image, mask, alpha=0.5, beta=1.0):
    if isinstance(image, Image.Image):
        image = np.array(image)

    assert mask.shape == image.shape[:-1]
    mask = mask * 255
    mask = mask.astype(np.uint8)
    random_color = [random.randint(0, 255) for _ in range(3)]
    colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for i in range(3):
        colored_mask[:, :, i] = mask * (random_color[i] / 255.0)

    alpha_channel = np.ones_like(mask) * int(alpha * 255)
    colored_mask = cv2.merge([colored_mask, alpha_channel])

    if image.shape[2] == 3:
        b, g, r = cv2.split(image)
        alpha_channel = np.ones(b.shape, dtype=b.dtype) * 255
        image = cv2.merge([b, g, r, alpha_channel])

    image = cv2.addWeighted(image, beta, colored_mask, alpha, 0)

    return image

def visualize_keypoints(image,keypoints,skeleton,index):
    x = keypoints[:,0]
    y = keypoints[:,1]
    v = keypoints[:,2]

    plt.figure(figsize=(10,10))
    plt.imshow(image)
    for sk in skeleton:
        sk = [item-1 for item in sk]
        if np.all(v[sk] > 0): # if two joint points' visualization > 0, draw line
            plt.plot(x[sk], y[sk], linewidth=2, color='red')
    plt.plot(x[v > 0], y[v > 0], 'o', markersize=2, markerfacecolor='red', markeredgecolor='k', markeredgewidth=2)
    plt.axis('off')
    plt.show()
    plt.savefig(f"keypoint_{index}")


def visualize_box_single(image, box, line_thickness=10):
    if isinstance(image, Image.Image):
        image = np.array(image)
    x1, y1, x2, y2 = box
    left_top = tuple((int(x1), int(y1)))
    right_bottom = tuple((int(x2), int(y2)))
    line_color = tuple(random.randint(0, 255) for _ in range(3)) 
    cv2.rectangle(image, left_top, right_bottom, line_color, line_thickness)

    return image

def visualize_point(image, points):
    if isinstance(image, Image.Image):
        image = np.array(image)
    if not isinstance(points, list):
        points = [points]

    result = visualize_mask(image, points)
    return result

def visualize_box(image, boxes, line_thickness=2):
    if isinstance(image, Image.Image):
        image = np.array(image)
    if not isinstance(boxes, list):
        boxes = [boxes]
    
    for box in boxes:
        x1, y1, x2, y2 = box
        left_top = tuple((x1, y1))
        right_bottom = tuple((x2, y2))
        line_color = tuple(random.randint(0, 255) for _ in range(3)) 
        cv2.rectangle(image, left_top, right_bottom, line_color, line_thickness)

def convert_bbox(bbox):
    """
    Convert bounding box from (x, y, w, h) to (xmin, xmax, ymin, ymax).

    Parameters:
    x (int or float): The x-coordinate of the center.
    y (int or float): The y-coordinate of the center.
    w (int or float): The width of the bounding box.
    h (int or float): The height of the bounding box.

    Returns:
    tuple: A tuple containing (xmin, xmax, ymin, ymax).
    """
    x, y, w, h = bbox
    xmin = x
    xmax = x + w
    ymin = y
    ymax = y + h
    return (xmin, ymin, xmax, ymax)