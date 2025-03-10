import torch
import re
import os
import json
import numpy as np
import torch.nn.functional as F
from PIL import Image
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
from ref_vlm.dataset.utils import mask_square2origin
from ref_vlm.utils.constants import BOT_TOKEN,EOT_TOKEN
from .utils.register_ade20k_panoptic import register_all_ade20k_panoptic,register_all_ade20k_semantic
from .utils.register_cityscapes_panoptic import register_all_cityscapes_panoptic
from .utils.get_cot import get_matches_from_text
from .utils.process import semantic_inference, panoptic_inference, instance_inference,build_evaluator, SEGDETProcessor
from ..base import BaseComputeMetrics

def coco_encode_rle(uncompressed_rle):
    h, w = uncompressed_rle["size"]
    rle = mask_utils.frPyObjects(uncompressed_rle, h, w)
    rle["counts"] = rle["counts"].decode("utf-8")  # Necessary to serialize with json
    return rle

def mask_to_rle_pytorch(tensor: torch.Tensor):
    """
    Encodes masks to an uncompressed RLE, in the format expected by
    pycoco tools.
    """
    # Put in fortran order and flatten h,w
    b, h, w = tensor.shape
    tensor = tensor.permute(0, 2, 1).flatten(1)

    # Compute change indices
    diff = tensor[:, 1:] ^ tensor[:, :-1]
    change_indices = diff.nonzero()

    # Encode run length
    out = []
    for i in range(b):
        cur_idxs = change_indices[change_indices[:, 0] == i, 1]
        cur_idxs = torch.cat(
            [torch.tensor([0], dtype=cur_idxs.dtype, device=cur_idxs.device), cur_idxs + 1,
             torch.tensor([h * w], dtype=cur_idxs.dtype, device=cur_idxs.device), ]
        )
        btw_idxs = cur_idxs[1:] - cur_idxs[:-1]
        counts = [] if tensor[i, 0] == 0 else [0]
        counts.extend(btw_idxs.detach().cpu().tolist())
        out.append({"size": [h, w], "counts": counts})

    return out


@METRICS.register_module()
class SEGComputeMetrics(BaseComputeMetrics):
    def __init__(self, *args,eval_type, task='instance', **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_type = eval_type
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
        for sample, gt_text, gt_masks,image_path in zip(data_samples,data_batch['data']['labels'],
                                                   data_batch['data']['decode_labels'],
                                                   data_batch['data']['image_path']):
            # initialize the output
            decode_pred = {}
            target = {}

            decode_pred_id = sample['generate_ids']
            decode_pred = self.decode_generate_ids(ids=decode_pred_id,skip_special_tokens=False)
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
            image_id = image_name.split('.')[0]
            # image_id = image_name.split('.')[0]
            image = Image.open(image_path).convert('RGB')
            target_masks = torch.tensor(gt_masks['mask']).float()
            target_masks = torch.stack([mask_square2origin(target_mask,image.width,image.height) for target_mask in target_masks])
            if sample['decoder_outputs'] is not None:
                decode_masks = sample['decoder_outputs']['mask']
                decode_masks = torch.stack([mask_square2origin(decode_mask,image.width,image.height,threshold=0.4) for decode_mask in decode_masks])
                decode_masks = decode_masks.float()
            else:
                decode_masks = torch.zeros_like(target_masks).float()

            decode_pred = {'pred_masks':decode_masks.float()}
            target = {'gt_mask':target_masks.float()}

            pred_box_mask_length = len(dt_labels)
            if len(decode_masks) != pred_box_mask_length:
                print(f"pred mask num: {len(decode_masks)} does not equal to llm's output num: {pred_box_mask_length}")
                assert len(decode_masks) < pred_box_mask_length
                dt_labels = dt_labels[:len(decode_masks)]
            
            decode_pred['dt_labels'] = dt_labels
            target['gt_labels'] = gt_labels

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

                mask_encode = mask_to_rle_pytorch(decode_masks.int())
                for i, mask_encode_single in enumerate(mask_encode):
                    score = scores[i]
                    class_name = pred_class_names[i]
                    rle_mask = coco_encode_rle(mask_encode_single)
                    coco_pred_file.append({"image_id": image_id, "category_id": self.processor.convert_cls_to_id(class_name), 
                                        "segmentation": rle_mask, "score": score})
            
                decode_pred['scores'] = scores
                decode_pred['pred_classes'] = pred_class_names

            elif self.eval_type == 'whole':

                mask_encode = mask_to_rle_pytorch(decode_masks.int())
                for i, mask_encode_single in enumerate(mask_encode):
                    rle_mask = coco_encode_rle(mask_encode_single)
                    coco_pred_file.append({"image_id": image_id, "category_id": 1, "segmentation": rle_mask, "score": 1.0})
     

            # processed_results = []
            # mask_pred_results = decode_masks
            # mask_cls_results = text_sims
            # for mask_cls_result, mask_pred_result in zip(mask_cls_results, mask_pred_results):
            #     processed_results.append({})
            #     # semantic segmentation inference
            #     mask_cls_result = mask_cls_result.to(mask_pred_result)

            #     r = retry_if_cuda_oom(semantic_inference)(
            #         mask_cls_result, mask_pred_result)
            #     processed_results[-1]["sem_seg"] = r

            #     # panoptic segmentation inference
            #     panoptic_r = retry_if_cuda_oom(panoptic_inference)(
            #         mask_cls_result, mask_pred_result)
            #     processed_results[-1]["panoptic_seg"] = panoptic_r

            #     # instance segmentation inference
            #     instance_r = retry_if_cuda_oom(instance_inference)(
            #         mask_cls_result, mask_pred_result)
            #     processed_results[-1]["instances"] = instance_r

            # self.task_evaluator.process(batch_input,processed_results)  # gt: image_id, file_name
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
        
        miou = self.processor.evaluate_box_mask_miou(preds,targets,self.eval_type,mask=True)
        recall = self.processor.evaluate_recall_with_mapping(preds,targets,mask=True, global_softmax=False)
        
        metrics = {
            'miou': miou,
            'recall': recall,
        }
        return metrics            


    

