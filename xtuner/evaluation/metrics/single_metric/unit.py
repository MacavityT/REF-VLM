import json
import sys
import torch
import re
import logging
from typing import Dict, Any, Union, Sequence,List
from pycocoevalcap.eval import Cider, Meteor, Bleu, Spice, PTBTokenizer
from mmengine.logging import print_log
from mmengine.registry.root import METRICS
from xtuner.utils import IGNORE_INDEX
from xtuner.utils.constants import BOT_TOKEN,EOT_TOKEN,VISUAL_REPRESENTATION_TOKEN
from ..okapi_metric import BaseComputeMetrics


@METRICS.register_module()
class UnitComputeMetrics(BaseComputeMetrics):

    """
    Tasks: COT tests: <Task>Unit decode (False). VRT prepared (False). Generate VRT (False).</Task>
    Metrics: 
    """

    def __init__(self, eval_type, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_type = eval_type
        assert self.eval_type in ['old_cot', 'new_cot'], "evaluation type for COTComputeMetrics should be in cot or vrt"

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
        
        for sample, gt in zip(
            data_samples, data_batch['data']['labels']):
            generate_ids = sample['generate_ids']
            decode_pred = self.decode_generate_ids(ids=generate_ids,skip_special_tokens=False)
            gt = gt[gt != IGNORE_INDEX]  # filter pad tokens (notes: better to use formal parameters)
            target = self.decode_generate_ids(ids=gt,skip_special_tokens=False)


            # extract content from target
            classes = 0


            
            if self.eval_type == 'old_cot':
                # old_cot extract answer only
                decode_pred = re.search(f"{BOT_TOKEN}(.*?){EOT_TOKEN}", decode_pred).group(1)
                target = re.search(f"{BOT_TOKEN}(.*?){EOT_TOKEN}", target).group(1)
            else:
                # new_cot extract cot and answer both
                cot_content = re.search(f"{BOT_TOKEN}(.*?){EOT_TOKEN}", decode_pred).group(1)
                


                
                            
            if self.save_dir is not None:
                self.save_outputs(decode_pred, target, f"cot_{self.eval_type}")

            self.results.append((decode_pred, target))

    def compute_metrics(self, results: list) -> dict:

        preds = []
        targets = []
        for i,(pred, target) in enumerate(results):
            preds.append(pred)
            targets.append(target)

        acc = self.accuracy(preds,targets)

        metrics = {
            'accuracy': acc,
        }

        return metrics
    

    def accuracy(self,preds,targets):
        true = 0
        for pred, target in zip(preds,targets):   
            if pred == target:
                true += 1

        acc = float(true) / float(len(preds))
        return acc

@METRICS.register_module()
class COTComputeMetrics(BaseComputeMetrics):
    def __init__(self, eval_type, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_type = eval_type
        assert self.eval_type in ['cot','all'], "evaluation type for COTComputeMetrics should be in cot or vrt"