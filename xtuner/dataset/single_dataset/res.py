import sys
import logging
import warnings
from typing import Dict, Any, Sequence
import torch
import pycocotools.mask as mask_utils
from pycocotools.mask import decode

from xtuner.registry import DATASETS, METRICS
from xtuner.utils.constants import (
    PHRASE_ST_PLACEHOLDER_STAGE2,
    PHRASE_ED_PLACEHOLDER_STAGE2,
    IMAGE_PLACEHOLDER,
    MASKS_PLACEHOLDER,
    EXPR_PLACEHOLDER,
)
from xtuner.evaluation.metrics import BaseComputeMetrics
from .mixin import MInstrDataset




@DATASETS.register_module()
class RESDataset(MInstrDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, placeholders=(IMAGE_PLACEHOLDER, EXPR_PLACEHOLDER))
        self.map_placeholders = {'output':[MASKS_PLACEHOLDER]}

    def __getitem__(self, index):
        offline_item = super().__getitem__(index)
        if offline_item is not None:
            return offline_item
        
        item = self.get_raw_item(index)
        img_path = item['img_path']
        expr = item['expression']
        mask = item['mask']
        rleObjs = mask_utils.frPyObjects(mask, item["height"], item["width"])
        mask = decode(rleObjs)
        if len(mask.shape) == 3:
            mask = mask[:,:,0]
        image = self.get_image(img_path)
        question = self.get_template().replace(EXPR_PLACEHOLDER, expr)

        value = PHRASE_ST_PLACEHOLDER_STAGE2 + 'target' + PHRASE_ED_PLACEHOLDER_STAGE2 + MASKS_PLACEHOLDER
        ret = {
            'image': image,
            'target': {
                'masks': [mask],
            },
            'conversations': [
                {
                    'from':'system',
                    'value':[{'task':{'task_name':'grounding_segmentation','element':['phrase'],'use_unit':True},'unit':['mask']}],
                },
                {
                    'from': 'human',
                    'value': question,
                },
                {
                    'from': 'gpt',
                    'value': f'{value}.',
                    'masks_seq': [[0]],
                }
            ]
        }
        ret['map_placeholders'] = self.map_placeholders
        return ret

