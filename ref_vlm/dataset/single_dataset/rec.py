import sys
import logging
import warnings
from typing import Dict, Any, Sequence

import torch
from torchvision.ops import box_iou

# from ..process_function import (
#     BoxFormatter,
# )

from ref_vlm.registry import DATASETS, METRICS
from ref_vlm.utils.constants import (
    PHRASE_ST_PLACEHOLDER_STAGE2,
    PHRASE_ED_PLACEHOLDER_STAGE2,
    IMAGE_PLACEHOLDER,
    BOXES_PLACEHOLDER,
    EXPR_PLACEHOLDER,
)
from ref_vlm.evaluation.metrics import BaseComputeMetrics
from .mixin import MInstrDataset


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout), ],
)


@DATASETS.register_module()
class RECDataset(MInstrDataset):
    def __init__(self, *args, target=False, **kwargs):
        super().__init__(*args, **kwargs, placeholders=(IMAGE_PLACEHOLDER, EXPR_PLACEHOLDER))
        self.map_placeholders = {'output':[BOXES_PLACEHOLDER]}
        self.target = target

    def __getitem__(self, index):
        offline_item = super().__getitem__(index)
        if offline_item is not None:
            return offline_item
        
        item = self.get_raw_item(index)
        img_path = item['img_path']
        expr = item['expression']
        bbox = item['bbox']

        image = self.get_image(img_path)
        question = self.get_template().replace(EXPR_PLACEHOLDER, expr)

        if self.stage == 1:
            ret = {
                'image': image,
                'target': {
                    'boxes': [bbox],
                },
                'conversations': [
                    {
                        'from': 'human',
                        'value': question,
                    },
                    {
                        'from': 'gpt',
                        'value': f'Answer: {BOXES_PLACEHOLDER}.',
                        'boxes_seq': [[0]],
                    }
                ]
            }


        if self.stage == 2:
            if self.target:
                value = PHRASE_ST_PLACEHOLDER_STAGE2 + expr + PHRASE_ED_PLACEHOLDER_STAGE2 + BOXES_PLACEHOLDER
            else:
                value = PHRASE_ST_PLACEHOLDER_STAGE2 + 'target' + PHRASE_ED_PLACEHOLDER_STAGE2 + BOXES_PLACEHOLDER
            ret = {
                'image': image,
                'target': {
                    'boxes': [bbox],
                },
                'conversations': [
                    {
                        'from':'system',
                        'value':[{'task':{'task_name':'grounding_detection','element':['phrase'],'use_unit':True},'unit':['box']}],
                    },
                    {
                        'from': 'human',
                        'value': question,
                    },
                    {
                        'from': 'gpt',
                        'value': f'{value}.',
                        'boxes_seq': [[0]],
                    }
                ]
            }
            ret['map_placeholders'] = self.map_placeholders
        return ret

