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
from PIL import Image

from vt_plug.utils.constants import IGNORE_INDEX
from vt_plug.dataset.utils import box_xywh_to_xyxy, mask_square2origin,de_norm_keypoint_square2origin,de_norm_box_xyxy_square2origin
from vt_plug.dataset.map_fns.dataset_map_fns.vt_map_fn_stage2 import get_cot_elements
from .utils.process import SEGDETProcessor
from .utils.get_cot import get_matches_from_text, get_caption_text
from ..base import BaseComputeMetrics


def post_process_pose(output_boxes, keypoints_output, keypoints_cls_output, target_sizes, image_id, num_body_points=17):
    
    output_boxes = output_boxes.float().cpu().numpy().tolist()
    output_boxes = [de_norm_box_xyxy_square2origin(decode_box,target_sizes[1],target_sizes[0]) for decode_box in output_boxes]
    
    keypoints_processed = []
    visibility = []
    for i,keypoints in enumerate(keypoints_output):
        keypoints = keypoints.float().cpu().numpy()
        keypoints_class = keypoints_cls_output[i].int().cpu().numpy()
        denorm_keypoints = de_norm_keypoint_square2origin(keypoints,target_sizes[1],target_sizes[0])
        keypoints_processed.append(denorm_keypoints)
        visibility.append(keypoints_class)
    
    keypoints_reform = torch.cat([torch.tensor(keypoints_processed),torch.tensor(visibility).unsqueeze(-1)],axis=-1).view(-1,num_body_points*3).numpy().tolist()

    results = []
    for keypoints,box in zip(keypoints_reform,output_boxes):
        results.append({"image_id":image_id,"score": 1, "category_id": 1, "boxes": box, "keypoints": keypoints})
    return results
    

@METRICS.register_module()
class POSEComputeMetrics(BaseComputeMetrics):
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
        
        coco_pred_file = []
        for sample, gt_text,image_path in zip(data_samples,data_batch['data']['labels'],data_batch['data']['image_path']):

            # initialize the output
            decode_pred = {}
            target = {}
            
            decode_pred_id = sample['generate_ids']
            decode_pred_string = self.decode_generate_ids(ids=decode_pred_id,skip_special_tokens=False)
            target_string = gt_text[gt_text != IGNORE_INDEX]  # filter pad tokens (notes: better to use formal parameters)
            target_string = self.decode_generate_ids(ids=target_string,skip_special_tokens=False)

            output_boxes = sample['decoder_outputs']['pose']['pred_boxes']
            output_keypoints = sample['decoder_outputs']['pose']['pred_kpts']
            output_keypoints_cls = sample['decoder_outputs']['pose']['pred_cls']
            image = Image.open(image_path)
            target_sizes = (image.height,image.width)
            image_name = os.path.basename(image_path)
            image_id = int(image_name.split('.')[0])

            results = post_process_pose(output_boxes,output_keypoints,output_keypoints_cls,target_sizes,image_id)

            self.results.append((decode_pred, target))
            self.save_outputs(decode_pred_string,target_string,f"{self.prefix}_pose")

            # Save gcg_coco_predictions
            with open(os.path.join(self.save_dir,f"{self.prefix}.json"), 'a') as f:
                json_line = json.dumps(results)
                f.write(json_line+'\n')
                f.close()

    def compute_metrics(self, results: list) -> dict:

        preds = []
        targets = []
        for i, (pred, target) in enumerate(results):
            preds.append(pred)
            targets.append(target)
        
        print("finish!")

        return 0
            

