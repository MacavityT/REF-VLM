import json
import sys
import torch
import re
import logging
import numpy as np
from typing import Dict, Any, Union, Sequence,List
from mmengine.logging import print_log
from mmengine.registry.root import METRICS
from xtuner.utils import IGNORE_INDEX
from utils.constants import (
    BOT_TOKEN, EOT_TOKEN,
    BOU_TOKEN, EOU_TOKEN,
    BOV_TOKEN, EOV_TOKEN,
    PHRASE_ST_PLACEHOLDER_STAGE2,
    PHRASE_ED_PLACEHOLDER_STAGE2,
    VISUAL_REPRESENTATION_TOKEN
)
from ..base import BaseComputeMetrics


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

    def cot_extract(self, content):
        # pattern string
        cot_pattern = BOT_TOKEN + r'(.*?)' + EOT_TOKEN
        phrase_pattern = PHRASE_ST_PLACEHOLDER_STAGE2 + r'(.*?)' + PHRASE_ED_PLACEHOLDER_STAGE2
        unit_pattern = BOU_TOKEN + r'(.*?)' + EOU_TOKEN
        num_pattern = r'\d+'

        t_matches = re.findall(cot_pattern, content, re.DOTALL)[0]
        all_targets = t_matches.split('\n')[2:-1]

        classes = []
        units = []
        nums = []
        for tgt in all_targets:
            cls = re.findall(phrase_pattern, tgt)[0]
            unit = re.findall(unit_pattern, tgt)[0]
            num = re.findall(num_pattern, tgt)[0]
            classes.append(cls)
            units.append(unit)
            nums.append(num)

        return dict(
            classes = classes,
            units = units,
            nums = nums
        )
    
    def answer_extract_old(self, content):
        cot_pattern = r'<Task>.*?</Task>'
        vrt_pattern = BOV_TOKEN + r'.*?' + EOV_TOKEN
        answer = re.sub(cot_pattern, '', content, flags=re.DOTALL)
        answer = re.sub(vrt_pattern, '', answer, flags=re.DOTALL)

        pattern = re.compile(r'<Phrase>(.*?)</Phrase>\((.*?)\)')
        matches = pattern.findall(answer)

        classes = []
        units = []
        nums = []
        for match in matches:
            phrase = match[0]
            unit_string = match[1]
            
            ref_pattern = re.compile(r'<Unit>(.*?)</Unit><REF>\[(\d+)\]')
            ref_matches = ref_pattern.findall(unit_string)
            unit = ref_matches[0][0] if ref_matches else ""
            refs = [int(ref[1]) for ref in ref_matches]

            classes.append(phrase)
            units.append(unit)
            nums.append(len(refs))
        
        return dict(
            classes = classes,
            units = units,
            nums = nums
        )

    def answer_extract(self, content):
        cot_pattern = BOT_TOKEN + r'.*?' + EOT_TOKEN
        answer = re.sub(cot_pattern, '', content, flags=re.DOTALL)
        answer_pattern = re.compile(r'<Phrase>(.*?)</Phrase>\(<Unit>(.*?)</Unit>(.*?)\)')
        matches = answer_pattern.findall(answer)

        classes = []
        units = []
        nums = []
        for match in matches:
            phrase = match[0]
            unit = match[1]
            ref_string = match[2]

            ref_pattern = re.compile(r'\[(\d+)\]<REF>')
            ref_matches = ref_pattern.findall(ref_string)
            ref_count = len(ref_matches)
            
            classes.append(phrase)
            units.append(unit)
            nums.append(ref_count)
        
        return dict(
            classes = classes,
            units = units,
            nums = nums
        )

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

            if self.save_dir is not None:
                self.save_outputs(decode_pred, target, f"cot_{self.eval_type}")

            if self.eval_type == 'old_cot':
                # old_cot extract answer only
                answer_gt = self.answer_extract_old(target)
                answer_pred = self.answer_extract_old(decode_pred)
                result = (answer_pred, answer_gt)
            else:
                cot_gt = self.cot_extract(target)
                cot_pred = self.cot_extract(decode_pred)
                answer_gt = self.answer_extract(target)
                answer_pred = self.answer_extract(decode_pred)
                result = ((cot_pred, cot_gt), (answer_pred, answer_gt))
                
            self.results.append(result)

    def compute_metrics(self, results: list) -> dict:
        if self.eval_type == 'old_cot':
            preds = []
            targets = []
            for answer_pred, answer_target in results:
                preds.append(answer_pred)
                targets.append(answer_target)
            answer_result = self.metrics(preds, targets)
            for k, v in answer_pred:
                metrics[f'answer_{k}'] = v
        else:
            cot_preds = []
            cot_targets = []
            answer_preds = []
            answer_targets = []
            for cot_result, answer_result in results:
                cot_pred, cot_target = cot_result
                answer_pred, answer_target = answer_result
                cot_preds.append(cot_pred)
                cot_targets.append(cot_target)
                answer_preds.append(answer_pred)
                answer_targets.append(answer_target)
            
            cot_result = self.metrics(cot_preds, cot_targets)
            answer_result = self.metrics(answer_preds, answer_targets)

            metrics = {}
            for k, v in cot_result:
                metrics[f'cot_{k}'] = v
            for k, v in answer_pred:
                metrics[f'answer_{k}'] = v
        
        return metrics
    

    def metrics(self, preds, targets):
        result = {}
        for key in preds.keys():
            assert key in targets.keys()

            accuracy = []
            precision = []
            recall = []
            for pred, target in zip(preds, targets):
                common_elements = set(pred).intersection(set(target))
                union_set = set(pred).union(set(target))
                accuracy = float(len(common_elements)) / float(len(union_set))
                precision = float(len(common_elements)) / float(len(set(pred)))
                recall = float(len(common_elements)) / float(len(set(target)))

            accuracy = np.mean(accuracy)
            precision = np.mean(precision)
            recall = np.mean(recall)
            result[f'{key}_acc'] = accuracy
            result[f'{key}_prec'] = precision
            result[f'{key}_rec'] = recall
        return result