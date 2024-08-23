import torch
import re
import os
import json
import numpy as np
import torch.nn.functional as F
from detectron2.structures import Boxes, ImageList, Instances
from mmengine.logging import print_log
from typing import Dict, Any, Union, Sequence,List
from mmengine.registry.root import METRICS
from enum import Enum
from pycocotools import mask as mask_utils
from sentence_transformers import SentenceTransformer, util
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, build_detection_test_loader
from detectron2.utils.memory import retry_if_cuda_oom

from xtuner.utils import IGNORE_INDEX
from xtuner.utils.constants import BOT_TOKEN,EOT_TOKEN
from .utils.register_ade20k_panoptic import register_all_ade20k_panoptic,register_all_ade20k_semantic
from .utils.register_cityscapes_panoptic import register_all_cityscapes_panoptic
from .utils.get_cot import get_matches_from_text
from .utils.process import semantic_inference, panoptic_inference, instance_inference,build_evaluator, SEGDETProcessor
from ..okapi_metric import BaseComputeMetrics

@METRICS.register_module()
class SEGComputeMetrics(BaseComputeMetrics):
    def __init__(self, *args, version, task, dataset_root,
                 bert_model='/model/Aaronzhu/all-MiniLM-L6-v2', num_queries=30, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset_root = dataset_root
        self.version = version
        assert self.version in ['general','prompt']
        self.task = task
        self.processor = SEGDETProcessor(task=self.prefix)
    

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
        for i,(sample, gt) in enumerate(zip(data_samples,data_batch['data'])):
            generate_ids =sample['generate_ids']
            decode_pred = self.decode_generate_ids(ids=generate_ids,skip_special_tokens=False)
            gt = gt['labels'][gt['labels'] != IGNORE_INDEX]  # filter pad tokens (notes: better to use formal parameters)
            target = self.decode_generate_ids(ids=gt,skip_special_tokens=False)
            target = target.replace('</s>','').strip()
            decode_pred = decode_pred.replace('</s>','').strip()

            matches_pred = get_matches_from_text(decode_pred)
            matches_target = get_matches_from_text(target)
            
            # TODO: change the format
            pred_masks = sample['masks'].cpu().numpy()   # torch.Tensor
            gt_masks = gt['masks'].cpu().numpy()
            decode_pred['pred_masks'] = pred_masks
            target['gt_masks'] = gt_masks

            image_path = gt['image_path']
            image_name = os.path.basename(image_path)
            image_id = image_name.split('.')[0]
            batch_input = {
                'file_name': image_name,
                'image_id': image_id,
            }
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
            target['gt_labels'] = gt_labels
                      
            if self.type == 'class':

                text_sims = np.zeros((len(dt_labels), len(self.processor.test_class_names)))
                for i, dt_label in enumerate(dt_labels):
                    for j, gt_label in enumerate(self.processor.test_class_names):
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
                pred_class_names = [self.processor.test_class_names[index] for index in ids]

                if self.mask:
                    for i, mask in enumerate(pred_masks):
                        score = scores[i]
                        class_name = pred_class_names[i]
                        rle_mask = mask_utils.encode(mask)
                        coco_pred_file.append({"image_id": image_id, "category_id": convert_cls_to_id(class_name), 
                                            "segmentation": rle_mask, "score": score})
               
                decode_pred['scores'] = scores
                decode_pred['pred_classes'] = pred_class_names

            elif self.type == 'whole':
                if self.mask:
                    for i, mask in enumerate(pred_masks):
                        rle_mask = mask_utils.encode(mask)
                        coco_pred_file.append({"image_id": image_id, "category_id": 1, "segmentation": rle_mask, "score": 1.0})
     

            processed_results = []
            mask_pred_results = pred_masks
            mask_cls_results = text_sims
            for mask_cls_result, mask_pred_result in zip(mask_cls_results, mask_pred_results):
                processed_results.append({})
                # semantic segmentation inference
                mask_cls_result = mask_cls_result.to(mask_pred_result)

                r = retry_if_cuda_oom(semantic_inference)(
                    mask_cls_result, mask_pred_result)
                processed_results[-1]["sem_seg"] = r

                # panoptic segmentation inference
                panoptic_r = retry_if_cuda_oom(panoptic_inference)(
                    mask_cls_result, mask_pred_result)
                processed_results[-1]["panoptic_seg"] = panoptic_r

                # instance segmentation inference
                instance_r = retry_if_cuda_oom(instance_inference)(
                    mask_cls_result, mask_pred_result)
                processed_results[-1]["instances"] = instance_r

            self.task_evaluator.process(batch_input,processed_results)  # gt: image_id, file_name
            self.results.append((decode_pred, target))
        # Save gcg_coco_predictions
        with open(f"{self.prefix}_{self.type}.json", 'w') as f:
            json.dump(coco_pred_file, f)
            



    def compute_metrics(self, results: list) -> dict:

        metrics = self.task_evaluator.evaluate()
        
        return metrics            


    

