import os
import json
import torch
import re
import numpy as np
import argparse
from typing import Dict, Any, Union, Sequence,List
from mmengine.registry.root import METRICS
from tqdm import tqdm
import torch.nn.functional as F
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as mask_utils
from pycocoevalcap.eval import COCOEvalCap
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from pycocoevalcap.eval import Cider, Meteor, Bleu, Spice, PTBTokenizer
from xtuner.utils.constants import IGNORE_INDEX

from .utils.process import SEGDETProcessor
from ..okapi_metric import BaseComputeMetrics




@METRICS.register_module()
class GCGComputeMetrics(BaseComputeMetrics):
    def __init__(self, *args, type, **kwargs):
        super().__init__(*args, **kwargs)
        self.processor = SEGDETProcessor(task=self.prefix)
        self.type = type

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
        
        coco_pred_file = []
        for sample, gt in zip(data_samples,data_batch['data']):
            decode_pred = {}
            target = {}
            generate_ids =sample['generate_ids']
            decode_pred = self.decode_generate_ids(ids=generate_ids,skip_special_tokens=False)
            gt = gt['labels'][gt['labels'] != IGNORE_INDEX]  # filter pad tokens (notes: better to use formal parameters)
            target = self.decode_generate_ids(ids=gt,skip_special_tokens=False)
            target = target.replace('</s>','').strip()
            decode_pred = decode_pred.replace('</s>','').strip()

            pattern = r"Name:\s*(.+?)\s*Unit:\s*<Unit>mask</Unit>\s*Num:\s*(\d+)"
            matches_pred = re.findall(pattern, decode_pred)
            matches_target = re.findall(pattern, target)

            image_path = gt['image_path']
            image_name = os.path.basename(image_path)
            image_id = image_name.split('.')[0]

            # TODO: change the format ,add caption
            pred_masks = sample['masks']   # torch.Tensor
            gt_masks = gt['masks']

            pred_mask_length = sum([int(pred[1]) for pred in matches_pred])
            assert len(pred_masks) == pred_mask_length,  \
                f"pred mask num: {len(pred_masks)} does not equal to llm's output num: {pred_mask_length}"
            
            dt_labels = []
            for i in len(matches_pred):
                cur_pred_label = matches_pred[i][0]
                cur_pred_num = int(matches_pred[i][1])
                for _ in range(cur_pred_num):
                    dt_labels.append(cur_pred_label)
            
            gt_labels = []
            for i in len(matches_target):
                cur_gt_label = matches_target[i][0]
                cur_gt_num = int(matches_target[i][1])
                for _ in range(cur_gt_num):
                    gt_labels.append(cur_gt_label)

            decode_pred['dt_labels'] = dt_labels
            decode_pred['pred_masks'] = pred_masks
            decode_pred['caption'] = pred_caption
            target['gt_labels'] = gt_labels
            target['gt_masks'] = gt_masks
            target['caption'] = target_caption
                      
            if self.type == 'class':

                text_sims = np.zeros((len(dt_labels), len(gt_labels)))
                for i, dt_label in enumerate(dt_labels):
                    for j, gt_label in enumerate(gt_labels):
                        if isinstance(gt_label,str):
                            text_sims[i, j] = self.processor.text_similarity_bert(dt_label,gt_label)
                        elif isinstance(gt_label,list):
                            max_similarity = 0
                            for single_label in gt_label:
                                similarity = self.processor.text_similarity_bert(dt_label,single_label)
                                if similarity > max_similarity:
                                    max_similarity = similarity
                            text_sims[i, j] = max_similarity
                        else:
                            raise NotImplementedError
                scores, ids = text_sims.max(1)
                pred_class_names = [gt_labels[index] for index in ids]

                for i, mask in enumerate(pred_masks):
                    score = scores[i]
                    class_name = pred_class_names[i]
                    rle_mask = mask_utils.encode(mask)
                    coco_pred_file.append({"image_id": image_id, "category_id": convert_cls_to_id(class_name), 
                                           "segmentation": rle_mask, "score": score})
                decode_pred['scores'] = scores
                decode_pred['pred_classes'] = pred_class_names

            elif self.type == 'whole':
                for i, mask in enumerate(pred_masks):
                    rle_mask = mask_utils.encode(mask)
                    coco_pred_file.append({"image_id": image_id, "category_id": 1, "segmentation": rle_mask, "score": 1.0})
        
        # Save gcg_coco_predictions
        with open(f"{self.prefix}_{self.type}.json", 'w') as f:
            json.dump(coco_pred_file, f)

            
        self.results.append((decode_pred, target))



    
    def compute_metrics(self, results: list) -> dict:

        preds = []
        targets = []
        for i, (pred, target) in enumerate(results):
            preds.append(pred)
            targets.append(target)
        
        miou = self.processor.evaluate_mask_miou(preds,targets,self.type)
        recall = self.processor.evaluate_recall_with_mapping(preds,targets)
        
        preds_labels = {i: [{"caption": x}] for i, x in enumerate(preds['caption'])}
        targets_labels = {i: [{"caption": x}] for i, x in enumerate(targets['caption'])}


        tokenizer = PTBTokenizer()
        targets_labels  = tokenizer.tokenize(targets_labels)
        preds_labels = tokenizer.tokenize(preds_labels)
        cider_score, meteor_score, bleu_score = Cider(), Meteor(), Bleu(4)
        cider_rst, _ = cider_score.compute_score(targets_labels, preds_labels)
        meteor_rst, _ = meteor_score.compute_score(targets_labels, preds_labels)
        blue_rst, _ = bleu_score.compute_score(targets_labels,preds_labels)

        metrics = {
            "CIDEr": cider_rst*100,
            "Meteor": meteor_rst*100,
            "BLEU4": blue_rst,
            'miou': miou,
            'recall': recall,
        }

        return metrics

