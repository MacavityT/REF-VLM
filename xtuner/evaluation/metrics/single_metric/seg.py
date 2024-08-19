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
            pred_mask = sample['masks']   # torch.Tensor
            pred_mask_with_queries = torch.zeros(1, self.num_queries, *pred_mask[0].shape) 
            image_path = gt['image_path']
            image_name = os.path.basename(image_path)
            image_id = image_name.split('.')[0]
            batch_input = {
                'file_name': image_name,
                'image_id': image_id,
            }
            pred_mask_length = sum([int(pred[1]) for pred in matches_pred])
            assert len(pred_mask) == pred_mask_length,  \
                f"pred mask num: {len(pred_mask)} does not equal to llm's output num: {pred_mask_length}"
            pred_mask_with_queries[0,:pred_mask_length] = pred_mask
            is_void = torch.ones(1, self.num_queries, 1)
            is_void[0,:pred_mask_length,0] = torch.zeros(1,pred_mask_length,1)
            is_void = is_void.cuda()

            cnt = 0
            batch_cosine_scores = []
            for i in len(matches_pred):
                cur_pred_label = matches_pred[i][0]
                cur_pred_num = int(matches_pred)[i][1]
                cur_pred_mask = pred_mask_with_queries[cnt:cur_pred_num]

                cnt += cur_pred_num

                # calculate cosine similarity
                outputs_embeddings = self.bert_model.encode(cur_pred_label, convert_to_tensor=True)
                cosine_scores = util.cos_sim(outputs_embeddings, self.class_sentence_embeddings)
                final_cosine_scores = []
                cur_idx = 0
                for num_t in self.region_test_num_templates:
                    final_cosine_scores.append(
                        cosine_scores[:, cur_idx: cur_idx + num_t].max(-1).values)
                    cur_idx += num_t
                final_pred_logits = torch.stack(final_cosine_scores, dim=-1)
                for _ in range(cur_pred_num):
                    batch_cosine_scores.append(final_pred_logits)

            region_cosine_scores = torch.concat(batch_cosine_scores, dim=0)
            cls_results = region_cosine_scores.unsqueeze(dim=0)  
            mask_cls_probs = torch.cat([
                cls_results.softmax(-1) * (1.0 - is_void),  # [1,queries,class_num+1]
                is_void], dim=-1)
            mask_cls_results = torch.log(mask_cls_probs + 1e-8)
            mask_pred_results = pred_mask_with_queries.cuda()

            processed_results = []
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


    def compute_metrics(self, results: list) -> dict:

        metrics = self.task_evaluator.evaluate()
        
        return metrics            


    

