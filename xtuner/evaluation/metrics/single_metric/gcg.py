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
class GCGComputeMetrics(BaseComputeMetrics):
    def __init__(self, *args, eval_type, mask, **kwargs):
        super().__init__(*args, **kwargs)
        self.processor = SEGDETProcessor(task=self.prefix)
        self.eval_type = eval_type
        self.mask = mask   # output bbox or masks

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
        for sample, gt_text, gt_boxes_masks,image_path in zip(data_samples,data_batch['data']['labels'],
                                                   data_batch['data']['decode_labels'],
                                                   data_batch['data']['image_path']):
            # initialize the output
            decode_pred = {}
            target = {}
            
            decode_pred_string = sample['generate_ids']
            decode_pred_string = self.decode_generate_ids(ids=decode_pred_string,skip_special_tokens=False)
            target_string = gt_text[gt_text != IGNORE_INDEX]  # filter pad tokens (notes: better to use formal parameters)
            target_string = self.decode_generate_ids(ids=target_string,skip_special_tokens=False)


            matches_pred = get_matches_from_text(decode_pred_string)
            matches_target = get_matches_from_text(target_string)


            # get caption text
            pred_caption = get_caption_text(decode_pred_string)
            target_caption = get_caption_text(target_string)

            image_name = os.path.basename(image_path)
            image_id = image_name.split('.')[0]

            # TODO: change the format ,add caption
            if self.mask:
                target_masks = torch.tensor(gt_boxes_masks['mask']).float()
                if sample['decoder_outputs'] is not None:
                    pred_boxes_masks = (sample['decoder_outputs']['masks'] > 0.5) * 1
                    pred_boxes_masks = F.interpolate(pred_boxes_masks.float().unsqueeze(0),
                                                size=(target_masks.shape[1],target_masks.shape[2]),
                                                    mode='bilinear', 
                                                    align_corners=False).squeeze(0)
                else:
                    pred_boxes_masks = torch.zeros_like(target_masks).float()
                    matches_pred = [('None',1)]
            else:
                if sample['decoder_outputs'] is not None:
                    pred_boxes_masks = sample['decoder_outputs']['boxes'].float().cpu().numpy().tolist()
                    pred_boxes_masks = [box_xywh_to_xyxy(decode_box) for decode_box in pred_boxes_masks]
                else:
                    # process unlimited generation
                    pred_boxes_masks = torch.zeros((1,4)).numpy().tolist()
                    matches_pred = [('None',1)]
                    pred_caption = ''

                decode_pred['pred_boxes'] = pred_boxes_masks
                target_boxes = torch.tensor(gt_boxes_masks['box']).float().cpu().numpy().tolist()
                target_boxes = [box_xywh_to_xyxy(target_box) for target_box in target_boxes]
                target['gt_boxes'] = target_boxes

            pred_box_mask_length = sum([int(pred[1]) for pred in matches_pred])
            assert len(pred_boxes_masks) == pred_box_mask_length,  \
                f"pred mask num: {len(pred_boxes_masks)} does not equal to llm's output num: {pred_box_mask_length}"
            
            dt_labels = []
            for i in range(len(matches_pred)):
                cur_pred_label = matches_pred[i][0]
                cur_pred_num = int(matches_pred[i][1])
                for _ in range(cur_pred_num):
                    dt_labels.append(cur_pred_label)
            
            gt_labels = []
            for i in range(len(matches_target)):
                cur_gt_label = matches_target[i][0]
                cur_gt_num = int(matches_target[i][1])
                for _ in range(cur_gt_num):
                    gt_labels.append(cur_gt_label)

            decode_pred['dt_labels'] = dt_labels
            decode_pred['caption'] = pred_caption
            target['gt_labels'] = gt_labels
            target['caption'] = target_caption
                      
            if self.eval_type == 'class':

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

                if self.mask:
                    for i, mask in enumerate(pred_boxes_masks):
                        score = scores[i]
                        class_name = pred_class_names[i]
                        rle_mask = mask_utils.encode(mask)
                        coco_pred_file.append({"image_id": image_id, "category_id": convert_cls_to_id(class_name), 
                                            "segmentation": rle_mask, "score": score})
                else:
                    for i, box in enumerate(pred_boxes_masks):
                        score = scores[i]
                        class_name = pred_class_names[i]
                        coco_pred_file.append({"image_id": image_id, "category_id": convert_cls_to_id(class_name), 
                                            "bbox": box, "score": score})                    
                decode_pred['scores'] = scores
                decode_pred['pred_classes'] = pred_class_names

            elif self.eval_type == 'whole':
                if self.mask:
                    for i, mask in enumerate(pred_boxes_masks):
                        rle_mask = mask_utils.encode(mask)
                        coco_pred_file.append({"image_id": image_id, "category_id": 1, "segmentation": rle_mask, "score": 1.0})
                else:
                    for i, box in enumerate(pred_boxes_masks):
                        coco_pred_file.append({"image_id": image_id, "category_id": 1, "bbox": box, "score": 1.0})

            self.results.append((decode_pred, target))
            
        # Save gcg_coco_predictions
        with open(os.path.join(self.save_dir,f"{self.prefix}_{self.eval_type}.json"), 'a') as f:
            # json.dump(coco_pred_file, f)
            json_line = json.dumps(coco_pred_file)
            f.write(json_line+'\n')
            f.close()
            


    def compute_metrics(self, results: list) -> dict:

        preds = []
        targets = []
        for i, (pred, target) in enumerate(results):
            preds.append(pred)
            targets.append(target)
        
        miou = self.processor.evaluate_box_mask_miou(preds,targets,self.eval_type,self.mask)
        recall = self.processor.evaluate_recall_with_mapping(preds,targets,self.mask)
        
        preds_labels = {i: [{"caption": x['caption']}] for i, x in enumerate(preds)}
        targets_labels = {i: [{"caption": x['caption']}] for i, x in enumerate(targets)}


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

