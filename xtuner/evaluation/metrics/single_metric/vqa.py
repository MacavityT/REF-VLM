import re
import sys
import json
import sys
import torch
import logging
from typing import Dict, Any, Union, Sequence,List
from pycocoevalcap.eval import Cider, Meteor, Bleu, Spice, PTBTokenizer
from mmengine.logging import print_log
from mmengine.registry.root import METRICS
from ..okapi_metric import BaseComputeMetrics



class VQAComputeMetrics(BaseComputeMetrics):
    """
    Tasks: VQA
    Metrics: Accuracy@1
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
        tasks = data_batch['data_samples']['tasks']
        
        for sample, task, gt in zip(
            data_samples,tasks,data_batch['data']['labels']):
            generate_ids =sample['generate_ids']
            decode_pred = self.decode_generate_ids(ids=generate_ids)
            gt = gt[gt != -100]  # filter pad tokens (notes: better to use formal parameters)
            target = self.decode_generate_ids(ids=gt)

            self.results.append((task, decode_pred, target))
    
    def compute_metrics(self, results: list) -> dict:

        preds = []
        targets = []
        for i,(task, pred, target) in enumerate(results):
            pred = self.extract_ans(pred)
            preds.append(pred)
            targets.append(target)

        acc = self.accuracy(preds,targets)

        metrics = {
            'accuracy': acc,
        }
        metrics['task'] = task

        self._print_results(metrics)
        
        return metrics
        

    def accuracy(self,preds,targets):

        true = 0
        for pred, target in zip(preds,targets):
            if target in pred.split(" "):
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