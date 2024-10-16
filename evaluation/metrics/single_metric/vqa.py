import re
import sys
import json
import sys
import torch
import logging
import ast
from typing import Dict, Any, Union, Sequence,List
from pycocoevalcap.eval import Cider, Meteor, Bleu, Spice, PTBTokenizer
from mmengine.logging import print_log
from mmengine.registry.root import METRICS
from xtuner.utils import IGNORE_INDEX
from utils.constants import BOT_TOKEN,EOT_TOKEN
from ..base import BaseComputeMetrics



class VQAComputeMetrics(BaseComputeMetrics):
    """
    Tasks: VQA
    Metrics: Accuracy@1
    """

    def __init__(self, *args, chunk=None,**kwargs):
        super().__init__(*args, **kwargs)
        self.chunk = chunk

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
            decode_pred = self.decode_generate_ids(ids=generate_ids,skip_special_tokens=False)
            gt = gt[gt != IGNORE_INDEX]  # filter pad tokens (notes: better to use formal parameters)
            target = self.decode_generate_ids(ids=gt,skip_special_tokens=False)
            if self.stage == 2:
                decode_pred = re.sub(f"{BOT_TOKEN}.*?{EOT_TOKEN}", "", decode_pred, flags=re.DOTALL)
                target = re.sub(f"{BOT_TOKEN}.*?{EOT_TOKEN}", "", target, flags=re.DOTALL)
            target = target.replace('</s>','').strip()
            decode_pred = decode_pred.replace('</s>','').strip()
            target = target.strip()
            decode_pred = decode_pred.strip()
            if self.save_dir is not None:
                if self.chunk is not None:
                    self.save_outputs(decode_pred,target,f"{self.prefix}_chunk{self.chunk}")
                else:
                    self.save_outputs(decode_pred,target,f"{self.prefix}")
                
            self.results.append((decode_pred, target))
    
    def compute_metrics(self, results: list) -> dict:

        preds = []
        targets = []
        for i,(pred, target) in enumerate(results):
            pred = self.extract_ans(pred)
            if self.prefix == 'okvqa':
                target = self.extract_answer(target)
            else:
                target = self.extract_ans(target)

            preds.append(pred)
            targets.append(target)

        acc = self.accuracy(preds,targets)

        metrics = {
            'accuracy': acc,
        }

        # self._print_results(metrics)
        
        return metrics
        

    def accuracy(self,preds,targets):

        true = 0
        for pred, target in zip(preds,targets):  # ppl vqa 

            if self.prefix == 'okvqa':
                if pred in target:
                    true += 1
            
            else:
                if len(target.split(" ")) == 1:
                    if target in pred.split(" "):
                        true += 1
                else:
                    if target == pred:
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
                string = string.split("the answer is ")[-1].lower().split(".")[0]
                return string
            except Exception as e:
                print_log(f"Warning: extract_ans for {string} but get exception: {e}")
                return None

    def extract_answer(self, string: str):
        """
        extract prediction strings from model output
        Args:
            string (str): "['pine', 'pine', 'pine', 'pine', 'shingles', 'shingles', 'wood', 'wood', 'cedar', 'cedar']"

        Return:
            list: ['pine', 'pine', 'pine', 'pine', 'shingles', 'shingles', 'wood', 'wood', 'cedar', 'cedar']
        """
        
        list_obj = ast.literal_eval(string)

        return list_obj