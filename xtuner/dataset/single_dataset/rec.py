import sys
import logging
import warnings
from typing import Dict, Any, Sequence

import torch
from torchvision.ops import box_iou

# from ..process_function import (
#     BoxFormatter,
# )

from xtuner.registry import DATASETS, METRICS
from xtuner.utils.constants import (
    PHRASE_ST_PLACEHOLDER_STAGE2,
    PHRASE_ED_PLACEHOLDER_STAGE2,
    IMAGE_PLACEHOLDER,
    BOXES_PLACEHOLDER,
    EXPR_PLACEHOLDER,
)
from xtuner.evaluation.metrics import BaseComputeMetrics
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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, placeholders=(IMAGE_PLACEHOLDER, EXPR_PLACEHOLDER))
        self.map_placeholders = {'output':[BOXES_PLACEHOLDER]}

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
                        'value': f'Answer: {value}.',
                        'boxes_seq': [[0]],
                    }
                ]
            }
            ret['map_placeholders'] = self.map_placeholders
        return ret


# @METRICS.register_module()
# class RECComputeMetrics(BaseComputeMetrics):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.box_formatter: BoxFormatter = self.preprocessor['target']['boxes']

#     def calculate_metric(self, preds: Sequence[str], targets: Sequence[str]) -> Dict[str, Any]:
#         failed = 0
#         target_failed = 0

#         pred_boxes, target_boxes = [], []
#         for pred, target in zip(preds, targets):
#             extract_pred = self.extract_ans(pred)
#             extract_target = self.extract_ans(target)
#             if extract_target is None:
#                 target_failed += 1
#                 logger.warning(f"failed to extract ans for target: {target}")
#                 continue
#             if extract_pred is None:
#                 failed += 1
#                 logger.warning(f"failed to extract ans for pred: {pred}")
#                 extract_pred = [0, 0, 0, 0]
#             target_boxes.append(extract_target)
#             pred_boxes.append(extract_pred)

#         with torch.no_grad():
#             target_boxes = torch.tensor(target_boxes)
#             pred_boxes = torch.tensor(pred_boxes)
#             # normalized box value is too small, so that the area is 0.
#             ious = box_iou(pred_boxes * 1000, target_boxes * 1000)
#             ious = torch.einsum('i i -> i', ious)  # take diag elem
#             # NOTE: please note iou only calculate for success target
#             iou = ious.mean().item()
#             correct = (ious > 0.5).sum().item()

#         # HACK: currently we expand image to square. so this iou is the real iou.
#         warn_message = "this iou is calculate on normalized box. just for non-rigorous training progress checking." \
#                        "the value is consistent with real iou only if image.width == image.height."
#         warnings.warn(warn_message)

#         return {
#             'accuracy': 1.0 * correct / len(targets),
#             'target_failed': target_failed,
#             'failed': failed,
#             'iou': iou,
#             'warning': warn_message,
#         }

#     def extract_ans(self, string: str):
#         try:
#             list_of_boxes = self.box_formatter.extract(string)
#             if len(list_of_boxes) != 1 or len(list_of_boxes[0]) != 1:
#                 return None
#             box = list_of_boxes[0][0]
#             if len(box) != 4:
#                 return None
#             return box
#         except Exception as e:
#             logger.warning(f"extract_ans for {string} but get exception: {e}")
#             return None
