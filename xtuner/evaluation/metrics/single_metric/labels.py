import json
import sys
import torch
import numpy as np
import re
import logging
from typing import Dict, Any, Union, Sequence,List
from pycocoevalcap.eval import Cider, Meteor, Bleu, Spice, PTBTokenizer
from mmengine.logging import print_log
from mmengine.registry.root import METRICS
from xtuner.utils import IGNORE_INDEX
from xtuner.utils.constants import BOT_TOKEN,EOT_TOKEN,VISUAL_REFERENCE_TOKEN,PHRASE_ED_PLACEHOLDER_STAGE2
from ..okapi_metric import BaseComputeMetrics


@METRICS.register_module()
class LabelsComputeMetrics(BaseComputeMetrics):

    """
    Tasks: examine the detection & segmentation results based on LLM
    Metrics: Accuracy
    Sample: <Task>Unit decode (True). VRT prepared (True). Generate VRT (False).</Task>
             <Phrase>the slower crew</Phrase>(<Unit>box</Unit><REF>[0]), 
             <Phrase>the John Hancock Tower</Phrase>(<Unit>box</Unit><REF>[0])
    """

    def __init__(self, eval_type, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_type = eval_type

        assert self.eval_type in ['phrase','count'], "evaluation type for PhraseComputeMetrics should be in phrase or count"

    # def process(self, data_batch:Any, data_samples:Sequence[dict]) -> None:
    #     """Process one batch of data samples and predictions. The processed
    #     results should be stored in ``self.results``, which will be used to
    #     compute the metrics when all batches have been processed.

    #     Args:
    #         data_batch (Any): A batch of data from the dataloader.
    #         data_samples (Sequence[dict]): A batch of outputs from
    #             the model.  
    #         {'generate_ids': generate ids}

    #     for type 'phrase':
    #         * pred: ['pred_label1', 'pred_label2', 'pred_label3']
    #         * target: ['target_label1', 'target_label2','target_label3', 'target_label4']
    #     for type 'count':
    #         * pred: [('pred_label1',1), ('pred_label2',2), ('pred_label3',3)]
    #         * target: [('target_label1',1), ('target_label2',2),('target_label3',3), ('target_label4',4)]
    #     """
        
    #     for sample, gt in zip(
    #         data_samples,data_batch['data']['labels']):
    #         generate_ids =sample['generate_ids']
    #         decode_pred = self.decode_generate_ids(ids=generate_ids)
    #         gt = gt[gt != IGNORE_INDEX]  # filter pad tokens (notes: better to use formal parameters)
    #         target = self.decode_generate_ids(ids=gt)

    #         # remove cot tests
    #         decode_pred = re.sub(f"{BOT_TOKEN}.*?{EOT_TOKEN}", "", decode_pred)
    #         target = re.sub(f"{BOT_TOKEN}.*?{EOT_TOKEN}", "", target)

    #         # retrieve phrase and units
    #         decode_pred = decode_pred.replace(f"{PHRASE_ED_PLACEHOLDER_STAGE2} ",PHRASE_ED_PLACEHOLDER_STAGE2)
    #         phrase_content_pred = re.findall(r"<Phrase>(.*?)</Phrase>\((.*?)\)", decode_pred)
    #         phrase_content_target = re.findall(r"<Phrase>(.*?)</Phrase>\((.*?)\)", target)

    #         if self.eval_type == 'phrase':
    #             decode_pred = [item[0].strip(" ") for item in phrase_content_pred]
    #             target = [item[0].strip(" ") for item in phrase_content_target]
    #         elif self.eval_type == 'count':  # [('the slower crew', 2), ('the John Hancock Tower', 1)]
    #             decode_pred = [(item[0].strip(" "), item[1].count(VISUAL_REFERENCE_TOKEN)) for item in phrase_content_pred]
    #             target = [(item[0].strip(" "), item[1].count(VISUAL_REFERENCE_TOKEN)) for item in phrase_content_target]

    #         if self.save_dir is not None:
    #             self.save_outputs(decode_pred,target,f"labels_{self.eval_type}")

    #         self.results.append((decode_pred, target))

    def process(self, data_batch:Any, data_samples:Sequence[dict]) -> None:
        """Process one batch of data samples and predictions. The processed
        results should be stored in ``self.results``, which will be used to
        compute the metrics when all batches have been processed.

        Args:
            data_batch (Any): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from
                the model.  
            {'generate_ids': generate ids}

        for type 'phrase':
            * pred: ['pred_label1', 'pred_label2', 'pred_label3']
            * target: ['target_label1', 'target_label2','target_label3', 'target_label4']
        for type 'count':
            * pred: [('pred_label1',1), ('pred_label2',2), ('pred_label3',3)]
            * target: [('target_label1',1), ('target_label2',2),('target_label3',3), ('target_label4',4)]
        """
        
        for sample, gt in zip(
            data_samples,data_batch['data']['labels']):
            generate_ids =sample['generate_ids']
            decode_pred = self.decode_generate_ids(ids=generate_ids)
            gt = gt[gt != IGNORE_INDEX]  # filter pad tokens (notes: better to use formal parameters)
            target = self.decode_generate_ids(ids=gt)
            print(f"decode_pred:{decode_pred}")
            print(f"target:{target}")
            # get contents from <Task> * </Task>
            decode_pred = re.search(f"{BOT_TOKEN}(.*?){EOT_TOKEN}", decode_pred, re.DOTALL).group(1).strip()
            target = re.search(f"{BOT_TOKEN}(.*?){EOT_TOKEN}", target, re.DOTALL).group(1).strip()

            # retrieve phrase and units
            if self.eval_type == 'phrase':
                decode_pred = re.findall(r"<Phrase>(.*?)</Phrase>", decode_pred)
                target = re.findall(r"<Phrase>(.*?)</Phrase>", target)
            elif self.eval_type == 'count':  # [('the slower crew',mask/box, 2), ('the John Hancock Tower', mask/box, 1)]
                entry_pattern = r"- Name: <Phrase>(.*?)</Phrase> Unit: <Unit>(.*?)</Unit> Num: (\d+)"
                decode_pred = re.findall(entry_pattern, decode_pred)
                target = re.findall(entry_pattern, target)

            if self.save_dir is not None:
                self.save_outputs(decode_pred,target,f"labels_{self.eval_type}")

            self.results.append((decode_pred, target))

    def compute_metrics(self, results: list) -> dict:

        preds = []
        targets = []
        for i,(pred, target) in enumerate(results):
            preds.append(pred)
            targets.append(target)

        accuracy, precision, recall = self.accuracy(preds,targets)

        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall
        }

        return metrics
    

    def accuracy(self,preds,targets):

        accuracy = []
        precision = []
        recall = []
        for pred, target in zip(preds,targets):

            common_elements = set(pred).intersection(set(target))
            union_set = set(pred).union(set(target))
            acc = float(len(common_elements)) / float(len(union_set))
            precision = float(len(common_elements)) / float(len(set(pred)))
            recall = float(len(common_elements)) / float(len(set(target)))

        accuracy = np.mean(accuracy)
        precision = np.mean(precision)
        recall = np.mean(recall)
        return accuracy, precision, recall

        