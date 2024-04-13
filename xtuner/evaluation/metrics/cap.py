import json
import sys
import torch
from typing import Dict, Any, Union, Sequence,List
from pycocoevalcap.eval import Cider, Meteor, Bleu, Spice, PTBTokenizer
from mmengine.registry.root import METRICS
from xtuner.evaluation.metrics.okapi_metric import BaseComputeMetrics



@METRICS.register_module()
class ImgCapComputeMetrics(BaseComputeMetrics):
    """
    eval_dataloader通过collect_fn中的eval_collate_fn.py定义
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
        """
        tasks = data_batch['data_samples']['tasks']
        preds = []

        for sample, attn_mask, task, gt in zip(
            data_samples,data_batch['data']['attention_mask'],tasks,data_batch['data']['labels']):
            pred_logits = sample['logits']   # TODO: 需要确认调用还是调用generate方法？ 这一块还要后续再确认一下
            first_zero_idx = self.find_first_zero_index(attn_mask)
            pred_idx = -1 if first_zero_idx is None else first_zero_idx - 1
            pred_logits_filter = pred_logits[pred_idx]
            pred = torch.argmax(pred_logits_filter,dim=1).item()
            preds.append(pred)
            self.results.append((task, pred, gt))


    def compute_metrics(self, results: list) -> dict:

        task,preds, targets = results

        preds = [self.extract_ans(p) for p in preds]
        preds = {i: [{"caption": x}] for i, x in enumerate(preds)}

        targets = [self.extract_ans(t) for t in targets]
        targets = {i: [{"caption": x}] for i, x in enumerate(targets)}
        json.dump({"preds": preds, "targets": targets}, open("rst.json", "w"))

        tokenizer = PTBTokenizer()
        targets  = tokenizer.tokenize(targets)
        preds = tokenizer.tokenize(preds)
        cider_score, meteor_score, bleu_score,spice_score = Cider(), Meteor(), Bleu(4), Spice()
        cider_rst, _ = cider_score.compute_score(targets, preds)
        meteor_rst, _ = meteor_score.compute_score(targets, preds)
        blue_rst, _ = bleu_score.compute_score(targets,preds)
        spice_rst, _ = spice_score.compute_score(targets,preds)

        return {
            "CIDEr": cider_rst*100,
            "Meteor": meteor_rst,
            "BLEU4": blue_rst,
            "SPICE": spice_rst
        }

    def extract_ans(self, string: str):
        try:
            string = string.split("ASSISTANT: ")[-1].lower().split("</s>")[0]
            return string
        except Exception as e:
            logger.warning(f"extract_ans for {string} but get exception: {e}")
            return None
