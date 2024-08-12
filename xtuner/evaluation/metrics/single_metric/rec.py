import torch
from typing import Dict, Any, Union, Sequence,List
from mmengine.logging import print_log
from torchvision.ops import box_iou
from mmengine.registry.root import METRICS
from ..okapi_metric import BaseComputeMetrics



# TODO: need to fix, because our model has decoders, bboxes do not generate in llm words
@METRICS.register_module()
class RECComputeMetrics(BaseComputeMetrics):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.box_formatter = self.preprocessor

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
        
        for sample, gt in zip(data_samples,data_batch['data']['labels']):
            # TODO: change the process
            decode_pred = sample
            target = gt
                
            self.results.append((decode_pred, target))

    
    def compute_metrics(self, results: list) -> dict:

        preds = []
        targets = []
        for i, (pred, target) in enumerate(results):
            preds.append(pred)
            targets.append(target)
        
        metrics = self.calculate_metric(preds,targets)
        
        return metrics


    def calculate_metric(self, preds: Sequence[str], targets: Sequence[str]) -> Dict[str, Any]:

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
        warn_message = "Warning: this iou is calculate on normalized box. just for non-rigorous training progress checking." \
                       "the value is consistent with real iou only if image.width == image.height."
        print_log(warn_message)

        return {
            'accuracy': 1.0 * correct / len(targets),
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
            print_log(f"Warning: extract_ans for {string} but get exception: {e}")
            return None
