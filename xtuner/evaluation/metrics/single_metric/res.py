import json
import sys
import torch
import logging
from typing import Dict, Any, Union, Sequence,List
from pycocoevalcap.eval import Cider, Meteor, Bleu, Spice, PTBTokenizer
from mmengine.logging import print_log
from mmengine.registry.root import METRICS
from ..okapi_metric import BaseComputeMetrics


# TODO: need to fix, because our model has decoders, bboxes do not generate in llm words
@METRICS.register_module()
class RESComputeMetrics(BaseComputeMetrics):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.box_formatter: BoxFormatter = self.preprocessor['target']['boxes']

    def __call__(self, eval_preds) -> Dict[str, Any]:
        preds, targets = eval_preds
        ciou = preds.sum()/targets.sum()
        fails = (preds<=0).sum()
        return {
            'ciou': ciou,
            'failed': fails,
        }
        # logger.warning(f"preds shape: {preds.shape}. targets shape: {targets.shape}")
        # assert len(preds) == len(targets)
        # return self.calculate_metric(preds, targets)
    def calculate_metric(self, preds: Sequence[str], targets: Sequence[str]) -> Dict[str, Any]:
        failed = 0
        target_failed = 0

        pred_boxes = preds
        target_boxes = targets
        with torch.no_grad():
            target_boxes = torch.tensor(target_boxes)
            pred_boxes = torch.tensor(pred_boxes)
            # normalized box value is too small, so that the area is 0.
            ious = box_iou(pred_boxes * 1000, target_boxes * 1000)
            ious = torch.einsum('i i -> i', ious)  # take diag elem
            # NOTE: please note iou only calculate for success target
            iou = ious.mean().item()
            correct = (ious > 0.5).sum().item()

        # HACK: currently we expand image to square. so this iou is the real iou.
        warn_message = "this iou is calculate on normalized box. just for non-rigorous training progress checking." \
                       "the value is consistent with real iou only if image.width == image.height."
        warnings.warn(warn_message)

        return {
            'accuracy': 1.0 * correct / len(targets),
            'target_failed': target_failed,
            'failed': failed,
            'iou': iou,
            'warning': warn_message,
        }

    def extract_ans(self, string: str):
        try:
            list_of_boxes = self.box_formatter.extract(string)
            if len(list_of_boxes) != 1 or len(list_of_boxes[0]) != 1:
                return None
            box = list_of_boxes[0][0]
            if len(box) != 4:
                return None
            return box
        except Exception as e:
            logger.warning(f"extract_ans for {string} but get exception: {e}")
            return None
