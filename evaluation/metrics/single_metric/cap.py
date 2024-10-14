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
from ..base import BaseComputeMetrics


@METRICS.register_module()
class ImgCapComputeMetrics(BaseComputeMetrics):

    """
    Tasks: Image caption, Region caption
    Metrics: CIDEr, Meteor, BLEU4, SPICE
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
            decode_pred = self.decode_generate_ids(ids=generate_ids,skip_special_tokens=False)
            gt = gt[gt != IGNORE_INDEX]  # filter pad tokens (notes: better to use formal parameters)
            target = self.decode_generate_ids(ids=gt,skip_special_tokens=False)
            if self.stage == 2:
                decode_pred = re.sub(f"{BOT_TOKEN}.*?{EOT_TOKEN}", "", decode_pred, flags=re.DOTALL)
                target = re.sub(f"{BOT_TOKEN}.*?{EOT_TOKEN}", "", target, flags=re.DOTALL)
            target = target.replace('</s>','').strip()
            decode_pred = decode_pred.replace('</s>','').strip()
            if self.save_dir is not None:
                self.save_outputs(decode_pred,target,f"{self.prefix}")
                
            self.results.append((decode_pred, target))


    def compute_metrics(self, results: list) -> dict:

        preds = []
        targets = []
        for i, (pred, target) in enumerate(results):
            pred = self.extract_ans(pred)
            preds.append(pred)
            targets.append(target)

        preds = {i: [{"caption": x}] for i, x in enumerate(preds)}
        targets = {i: [{"caption": x}] for i, x in enumerate(targets)}


        tokenizer = PTBTokenizer()
        targets  = tokenizer.tokenize(targets)
        preds = tokenizer.tokenize(preds)
        cider_score, meteor_score, bleu_score = Cider(), Meteor(), Bleu(4)
        cider_rst, _ = cider_score.compute_score(targets, preds)
        meteor_rst, _ = meteor_score.compute_score(targets, preds)
        blue_rst, _ = bleu_score.compute_score(targets,preds)
        # spice_rst, _ = spice_score.compute_score(targets,preds)

        metrics = {
            "CIDEr": cider_rst*100,
            "Meteor": meteor_rst,
            "BLEU4": blue_rst,
            # "SPICE": spice_rst
        }

        # self._print_results(metrics)

        return metrics
    
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
        
