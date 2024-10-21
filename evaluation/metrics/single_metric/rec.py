import torch
import re
import numpy as np
from typing import Dict, Any, Union, Sequence,List
from mmengine.logging import print_log
from torchvision.ops import box_iou
from mmengine.registry.root import METRICS
from xtuner.utils import IGNORE_INDEX
from utils.constants import BOT_TOKEN,EOT_TOKEN
from dataset.utils import box_xywh_to_xyxy
from ..base import BaseComputeMetrics

# TODO: need to fix, because our model has decoders, bboxes do not generate in llm words
@METRICS.register_module()
class RECComputeMetrics(BaseComputeMetrics):
    def __init__(self, *args, dataset_name, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset_name = dataset_name

    def process(self, data_batch:Any, data_samples:Any) -> None:
        """Process one batch of data samples and predictions. The processed
        results should be stored in ``self.results``, which will be used to
        compute the metrics when all batches have been processed.

        Args:
            data_batch (Any): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from
                the model.  
            {'generate_ids': generate ids}
        """
        
        for sample, gt_text, gt_boxes in zip(data_samples,data_batch['data']['labels'],data_batch['data']['decode_labels']):
            # TODO: change the process
            decode_pred_string = sample['generate_ids']

            decode_pred_string = self.decode_generate_ids(ids=decode_pred_string,skip_special_tokens=False)
            target_string = gt_text[gt_text != IGNORE_INDEX]  # filter pad tokens (notes: better to use formal parameters)
            target_string = self.decode_generate_ids(ids=target_string,skip_special_tokens=False)

            if sample['decoder_outputs'] is not None:
                if sample['decoder_outputs']['box'] is not None:
                    decode_boxes = sample['decoder_outputs']['box'].float().cpu().numpy().tolist()
                    decode_boxes = [box_xywh_to_xyxy(decode_box) for decode_box in decode_boxes]
                else:
                    decode_boxes = torch.zeros((1,4)).numpy().tolist()
            else:
                decode_boxes = torch.zeros((1,4)).numpy().tolist()
            target_boxes = torch.tensor(gt_boxes['box']).float().cpu().numpy().tolist()
            target_boxes = [box_xywh_to_xyxy(target_box) for target_box in target_boxes]
            decode_boxes = torch.tensor(decode_boxes)
            target_boxes = torch.tensor(target_boxes)

            decode_pred = {'text':decode_pred_string,'boxes':decode_boxes}
            target = {'text':target_string,'boxes':target_boxes}
            
            if self.stage == 2:
                decode_pred_string = re.sub(f"{BOT_TOKEN}.*?{EOT_TOKEN}", "", decode_pred_string, flags=re.DOTALL)
                target_string = re.sub(f"{BOT_TOKEN}.*?{EOT_TOKEN}", "", target_string, flags=re.DOTALL)
            target_string = target_string.replace('</s>','').strip()
            decode_pred_string = decode_pred_string.replace('</s>','').strip()

            decode_pred_string += str(decode_boxes.float().cpu().numpy().tolist())
            target_string += str(target_boxes.float().cpu().numpy().tolist())

            if self.save_dir is not None:
                self.save_outputs(decode_pred_string,target_string,f"{self.prefix}_{self.dataset_name}")

            self.results.append((decode_pred, target))

    
    def compute_metrics(self, results: list) -> dict:

        preds = []
        targets = []
        for i, (pred, target) in enumerate(results):
            preds.append(pred)
            targets.append(target)
        
        metrics = self.calculate_metric(preds,targets)
        
        return metrics


    def calculate_metric(self, preds, targets) -> Dict[str, Any]:

        pred_boxes = [pred['boxes'] for pred in preds]
        target_boxes = [target['boxes'] for target in targets]
        all_corrects = 0
        all_ious = []
        with torch.no_grad():
            for pred_box, target_box in zip(pred_boxes,target_boxes):
                # normalized box value is too small, so that the area is 0.
                ious = box_iou(pred_box * 1000, target_box * 1000)
                try:
                    ious = torch.einsum('i i -> i', ious)  # take diag elem
                except:
                    ious = ious.max()
                # NOTE: please note iou only calculate for success target
                iou = ious.mean().item()
                all_corrects += (ious > 0.5).sum().item()
                all_ious.append(iou)
        # HACK: currently we expand image to square. so this iou is the real iou.
        warn_message = "Warning: this iou is calculate on normalized box. just for non-rigorous training progress checking." \
                       "the value is consistent with real iou only if image.width == image.height."
        print_log(warn_message)

        return {
            'accuracy': 1.0 * all_corrects / len(targets),
            'iou': np.mean(all_ious),
            'warning': warn_message,
        }
