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
from xtuner.dataset.utils import box_xywh_to_xyxy
from xtuner.dataset.map_fns.dataset_map_fns.okapi_map_fn_stage2 import get_cot_elements
from .utils.process import SEGDETProcessor
from .utils.get_cot import get_matches_from_text, get_caption_text
from ..okapi_metric import BaseComputeMetrics




@METRICS.register_module()
class DETComputeMetrics(BaseComputeMetrics):
    def __init__(self, *args, eval_type, **kwargs):
        super().__init__(*args, **kwargs)
        self.processor = SEGDETProcessor(task=self.prefix)
        self.eval_type = eval_type

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
        
        if 'decode_labels' not in data_batch['data'].keys():
            return None

        coco_pred_file = []
        for sample, gt_text, gt_boxes_masks,image_path in zip(data_samples,data_batch['data']['labels'],
                                                   data_batch['data']['decode_labels'],
                                                   data_batch['data']['image_path']):
            # initialize the output
            decode_pred = {}
            target = {}
            
            decode_pred_id = sample['generate_ids']
            decode_pred_string = self.decode_generate_ids(ids=decode_pred_id,skip_special_tokens=False)
            target_string = gt_text[gt_text != IGNORE_INDEX]  # filter pad tokens (notes: better to use formal parameters)
            target_string = self.decode_generate_ids(ids=target_string,skip_special_tokens=False)


            matches_target = get_matches_from_text(target_string)
            if 'decode_groups' in sample.keys():
                dt_labels = []
                for decoder_group in sample['decode_groups']:
                    phrase_pair = decoder_group['phrase_pair']
                    ref_ids = decoder_group['ref_index']
                    for _ in range(len(ref_ids)):
                        if len(phrase_pair) > 0:
                            label = self.decode_generate_ids(decode_pred_id[phrase_pair[0]+1:phrase_pair[1]],
                                                            skip_special_tokens=False)
                        else:
                            label = None
                        dt_labels.append(label)
            gt_labels = []
            for i in range(len(matches_target)):
                cur_gt_label = matches_target[i][0].strip()
                cur_gt_num = int(matches_target[i][1])
                for _ in range(cur_gt_num):
                    gt_labels.append(cur_gt_label)

            image_name = os.path.basename(image_path)
            image_id = int(image_name.split('.')[0])

            if sample['decoder_outputs'] is not None:
                pred_boxes_masks = sample['decoder_outputs']['box'].float().cpu().numpy().tolist()
                pred_boxes_masks = [box_xywh_to_xyxy(decode_box) for decode_box in pred_boxes_masks]
            else:
                # process unlimited generation
                pred_boxes_masks = torch.zeros((1,4)).numpy().tolist()
                dt_labels = [None]

            decode_pred['pred_boxes'] = pred_boxes_masks
            target_boxes = torch.tensor(gt_boxes_masks['box']).float().cpu().numpy().tolist()
            target_boxes = [box_xywh_to_xyxy(target_box) for target_box in target_boxes]
            target['gt_boxes'] = target_boxes

            pred_box_mask_length = len(dt_labels)
            assert len(pred_boxes_masks) == pred_box_mask_length,  \
                f"pred mask num: {len(pred_boxes_masks)} does not equal to llm's output num: {pred_box_mask_length}"
            
            decode_pred['dt_labels'] = dt_labels
            target['gt_labels'] = gt_labels

            if self.eval_type == 'class':

                text_sims = np.zeros((len(dt_labels), len(self.processor.test_class_names)))

                if self.processor.test_class_features is None:
                    for i, dt_label in enumerate(dt_labels):
                        for j, gt_label in enumerate(self.processor.test_class_names):
                            if isinstance(gt_label,str):
                                text_sims[i, j] = self.processor.text_similarity(dt_label,gt_label)
                            elif isinstance(gt_label,list):
                                max_similarity = 0
                                for single_label in gt_label:
                                    similarity = self.processor.text_similarity(dt_label,single_label)
                                    if similarity > max_similarity:
                                        max_similarity = similarity
                                text_sims[i, j] = max_similarity
                            else:
                                raise NotImplementedError
                else:
                    dt_labels_features = self.processor.get_clip_embedding(dt_labels)
                    for i, dt_label in enumerate(dt_labels):
                        dt_label_features = dt_labels_features[i].reshape(1,-1)
                        for j, gt_label in enumerate(self.processor.test_class_names):
                            gt_class_feature = self.processor.test_class_features[j].reshape(1,-1)
                            text_sims[i, j] = self.processor.text_similarity(dt_label_features,gt_class_feature)


                scores = text_sims.max(1)
                ids = text_sims.argmax(1)
                pred_class_names = [self.processor.test_class_names[index] for index in ids]

                for i, box in enumerate(pred_boxes_masks):
                    score = np.clip(scores[i],0,1)
                    class_name = pred_class_names[i]
                    coco_pred_file.append({"image_id": image_id, "category_id": self.processor.convert_cls_to_id(class_name), 
                                        "bbox": box, "score": score})                    
                decode_pred['scores'] = scores
                decode_pred['pred_classes'] = pred_class_names

            elif self.eval_type == 'whole':
                for i, box in enumerate(pred_boxes_masks):
                    coco_pred_file.append({"image_id": image_id, "category_id": 1, "bbox": box, "score": 1.0})

            self.results.append((decode_pred, target))
            
        # Save gcg_coco_predictions
        with open(os.path.join(self.save_dir,f"{self.prefix}_{self.eval_type}.json"), 'a') as f:
            json_line = json.dumps(coco_pred_file)
            f.write(json_line+'\n')
            f.close()
            

    def compute_metrics(self, results: list) -> dict:

        preds = []
        targets = []
        for i, (pred, target) in enumerate(results):
            preds.append(pred)
            targets.append(target)
        
        miou = self.processor.evaluate_box_mask_miou(preds,targets,self.eval_type,mask=False)
        recall = self.processor.evaluate_recall_with_mapping(preds,targets,mask=False, global_softmax=False)
        
        metrics = {
            'miou': miou,
            'recall': recall,
        }

        return metrics
