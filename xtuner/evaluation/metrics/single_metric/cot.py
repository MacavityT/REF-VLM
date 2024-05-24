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
from xtuner.utils.constants import BOT_TOKEN,EOT_TOKEN
from ..okapi_metric import BaseComputeMetrics


@METRICS.register_module()
class COTComputeMetrics(BaseComputeMetrics):

    """
    Tasks: COT tests: <Task>Unit decode (False). VRT prepared (False). Generate VRT (False).</Task>
    Metrics: 
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


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
            data_samples,data_batch['data']['labels']):
            generate_ids =sample['generate_ids']
            decode_pred = self.decode_generate_ids(ids=generate_ids)
            gt = gt[gt != IGNORE_INDEX]  # filter pad tokens (notes: better to use formal parameters)
            target = self.decode_generate_ids(ids=gt)
            if self.stage == 2:
                decode_pred = re.search(f"{BOT_TOKEN}(.*?){EOT_TOKEN}", decode_pred).group(1)
                target = re.search(f"{BOT_TOKEN}(.*?){EOT_TOKEN}", target).group(1)
            self.results.append((decode_pred, target))

    def compute_metrics(self, results: list) -> dict:

        preds = []
        targets = []
        for i,(pred, target) in enumerate(results):
            pred = self.extract_ans(pred)
            target = self.extract_ans(target)
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


    def extract_ans(self, string: str):
        """
        extract prediction strings from model output
        Args:
            string (str): USER: <image>\nPlease describe this picture ASSISTANT: augment reality in opencv.</s>

        Return:
            string (str): e.g. aument reality in opencv 
        """
        try:
            string = string.split("ASSISTANT: ")[-1].lower().split("</s>")[0]
            return string
        except Exception as e:
            print_log(f"Warning: extract_ans for {string} but get exception: {e}")
            return None
        