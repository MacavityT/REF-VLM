import re
from mmengine.registry.root import METRICS
from typing import Dict, Any, Union, Sequence,List
from ..okapi_metric import BaseComputeMetrics

ANS_EXTRACT_PAT = re.compile(r'(?:(?:(?:(?:(?:So t)|(?:T)|(?:t))he answer is)|(?:Answer:)) (.+))')

@METRICS.register_module()
class PopeComputeMetrics(BaseComputeMetrics):
    
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
        for i, (task, pred, target) in enumerate(results):
            preds.append(self.extract_ans(pred))
            targets.append(self.extract_ans(target))

        correct = 0
        failed = 0
        target_failed = 0
        acc, precision, recall, f1, yes_ratio = self.other_metric(preds, targets)
        for pred, target in zip(preds, targets):
            extract_pred = pred
            extract_target = target
            if extract_target is None:
                target_failed += 1
                continue
            if extract_pred is None:
                failed += 1
            if extract_pred == extract_target:
                correct += 1

        metrics = {
            'accuracy': acc,
            'precision': precision,
            'recall': recall,
            "f1": f1,
            "yes_ratio": yes_ratio,
            'target_failed': target_failed,
            'failed': failed,
        }
        metrics['task'] = task
        
        self._print_results(metrics)
        
        return metrics

    def other_metric(self, answers, label_list):
        for i in range(len(label_list)):
            if label_list[i] == 'no':
                label_list[i] = 0
            else:
                label_list[i] = 1

        pred_list = []
        for answer in answers:
            if answer == 'no':
                pred_list.append(0)
            else:
                pred_list.append(1)

        pos = 1
        neg = 0
        yes_ratio = pred_list.count(1) / len(pred_list)

        TP, TN, FP, FN = 0, 0, 0, 0
        for pred, label in zip(pred_list, label_list):
            if pred == pos and label == pos:
                TP += 1
            elif pred == pos and label == neg:
                FP += 1
            elif pred == neg and label == neg:
                TN += 1
            elif pred == neg and label == pos:
                FN += 1

        precision = float(TP) / float(TP + FP)
        recall = float(TP) / float(TP + FN)
        f1 = 2 * precision * recall / (precision + recall)
        acc = (TP + TN) / (TP + TN + FP + FN)
        return acc, precision, recall, f1, yes_ratio
    
    def extract_ans(self, string: str):
        if "ASSISTANT: " in string:
            string = string.split("ASSISTANT: ")[-1].lower()
        try:
            found = ANS_EXTRACT_PAT.findall(string.strip())
            if len(found) != 1:
                if "yes" in string:
                    return "yes"
                else:
                    return "no"
            return found[0].strip().rstrip('.').strip()
        except (IndexError, AttributeError):
            return None
