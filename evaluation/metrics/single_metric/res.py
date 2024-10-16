import json
import sys
import re
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import os
import cv2
import logging
from typing import Dict, Any, Union, Sequence,List
from pycocoevalcap.eval import Cider, Meteor, Bleu, Spice, PTBTokenizer
from mmengine.logging import print_log
from mmengine.registry.root import METRICS
from xtuner.dataset.utils import mask_square2origin,visualize_mask
from xtuner.utils import IGNORE_INDEX
from utils.constants import BOT_TOKEN,EOT_TOKEN
from enum import Enum
from ..base import BaseComputeMetrics


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f", summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if isinstance(self.sum, np.ndarray):
            total = torch.tensor(
                self.sum.tolist()
                + [
                    self.count,
                ],
                dtype=torch.float32,
                device=device,
            )
        else:
            total = torch.tensor(
                [self.sum, self.count], dtype=torch.float32, device=device
            )
        if total.shape[0] > 2:
            self.sum, self.count = total[:-1].cpu().numpy(), total[-1].cpu().item()
        else:
            self.sum, self.count = total.tolist()
        self.avg = self.sum / (self.count + 1e-5)

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ""
        if self.summary_type is Summary.NONE:
            fmtstr = ""
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = "{name} {avg:.3f}"
        elif self.summary_type is Summary.SUM:
            fmtstr = "{name} {sum:.3f}"
        elif self.summary_type is Summary.COUNT:
            fmtstr = "{name} {count:.3f}"
        else:
            raise ValueError("invalid summary type %r" % self.summary_type)

        return fmtstr.format(**self.__dict__)


def intersectionAndUnionGPU(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert output.dim() in [1, 2, 3]
    assert output.shape == target.shape
    output = output.view(-1)
    target = target.view(-1)
    output[target == ignore_index] = ignore_index
    intersection = output[output == target]
    area_intersection = torch.histc(intersection, bins=K, min=0, max=K - 1)
    area_output = torch.histc(output, bins=K, min=0, max=K - 1)
    area_target = torch.histc(target, bins=K, min=0, max=K - 1)
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target

# TODO: need to fix, because our model has decoders, bboxes do not generate in llm words
@METRICS.register_module()
class RESComputeMetrics(BaseComputeMetrics):
    def __init__(self, *args, dataset_name, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset_name = dataset_name

    def process(self, data_batch:Any, data_samples:Sequence[dict]) -> None:
        """Process one batch of data samples and predictions. The processed
        results should be stored in ``self.results``, which will be used to
        compute the metrics when all batches have been processed.

        Args:
            data_batch (Any): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from
                the model.  
            {'generate_ids': generate ids}
        """
        
        for sample, gt_text, gt_masks,image_path in zip(data_samples,
                                                        data_batch['data']['labels'],
                                                        data_batch['data']['decode_labels'],
                                                        data_batch['data']['image_path']):
            image = Image.open(image_path).convert('RGB')
            decode_pred_string = sample['generate_ids']
            decode_pred_string = self.decode_generate_ids(ids=decode_pred_string,skip_special_tokens=False)
            target_string = gt_text[gt_text != IGNORE_INDEX]  # filter pad tokens (notes: better to use formal parameters)
            target_string = self.decode_generate_ids(ids=target_string,skip_special_tokens=False)

            target_masks = torch.tensor(gt_masks['mask']).float()
            target_masks = torch.stack([mask_square2origin(target_mask,image.width,image.height) for target_mask in target_masks])
            if sample['decoder_outputs'] is not None:
                # decode_masks = (sample['decoder_outputs']['mask'] > 0.5) * 1
                decode_masks = [sample['decoder_outputs']['mask'][0]]
                decode_masks = torch.stack([mask_square2origin(decode_mask,image.width,image.height,threshold=0.4) for decode_mask in decode_masks])
                decode_masks = decode_masks.float()
            else:
                decode_masks = torch.zeros_like(target_masks).float()

            decode_pred = {'text':decode_pred_string,'masks':decode_masks.float()}
            target = {'text':target_string,'masks':target_masks.float()}
            
            if self.stage == 2:
                decode_pred_string = re.sub(f"{BOT_TOKEN}.*?{EOT_TOKEN}", "", decode_pred_string, flags=re.DOTALL)
                target_string = re.sub(f"{BOT_TOKEN}.*?{EOT_TOKEN}", "", target_string, flags=re.DOTALL)
            target_string = target_string.replace('</s>','').strip()
            decode_pred_string = decode_pred_string.replace('</s>','').strip()

            if self.save_dir is not None:
                self.save_outputs(decode_pred_string,target_string,f"{self.prefix}_{self.dataset_name}")

            self.results.append((decode_pred, target))

            image_name = os.path.basename(image_path)
            image_id = image_name.split('.')[0]
            decode_masks = [decode_mask for decode_mask in decode_masks.cpu().numpy()]
            target_masks = [target_mask for target_mask in target_masks.cpu().numpy()]
            vis_mask_pred = visualize_mask(image, decode_masks, alpha=0.8)
            vis_mask_target = visualize_mask(image, target_masks, alpha=0.8)
            pred_save_path = f'pred_{image_id}.jpg'
            target_save_path = f'target_{image_id}.jpg'
            save_dir = os.path.join(self.save_dir,f'{self.dataset_name}')
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            cv2.imwrite(os.path.join(save_dir,pred_save_path),vis_mask_pred)
            cv2.imwrite(os.path.join(save_dir,target_save_path),vis_mask_target)

    
    def compute_metrics(self, results: list) -> dict:

        preds = []
        targets = []
        for i, (pred, target) in enumerate(results):
            preds.append(pred['masks'])
            targets.append(target['masks'])
        
        metrics = self.calculate_metric(preds,targets)
        
        return metrics
    

    def calculate_metric(self, preds: Sequence[str], targets: Sequence[str]) -> Dict[str, Any]:
        trackers = {
            "intersection": AverageMeter("Intersec", ":6.3f", Summary.SUM),
            "union": AverageMeter("Union", ":6.3f", Summary.SUM),
            "gIoU": AverageMeter("gIoU", ":6.3f", Summary.SUM)
        }
        intersection, union, accuracy_iou = 0.0, 0.0, 0.0
        for target, prediction in zip(targets,preds):
            intersect, union_, _ = intersectionAndUnionGPU(
                prediction.contiguous().clone(), target.contiguous(), 2, ignore_index=255
            )
            intersection = intersect
            union = union_
            accuracy_iou = intersect / (union_ + 1e-5)
            # handles no-object targets
            accuracy_iou[union_ == 0] += 1.0
            intersection, union = intersection.cpu().numpy(), union.cpu().numpy()
            trackers["intersection"].update(intersection)
            trackers["union"].update(union)
            accuracy_iou = accuracy_iou.cpu().numpy() / target.shape[0]
            trackers["gIoU"].update(accuracy_iou, n=target.shape[0])

        for meter in trackers.values():
            meter.all_reduce()

        iou_per_class = trackers["intersection"].sum / (trackers["union"].sum + 1e-10)
        class_iou = iou_per_class[1]
        global_iou = trackers["gIoU"].avg[1]

        return {
            'class_iou': class_iou,
            'global_iou': global_iou,
        }
        

