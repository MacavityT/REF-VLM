import os
import torch
import json
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from pycocotools.mask import decode
from PIL import Image
from xtuner.registry import DATASETS

from xtuner.utils.constants import (
    BOXES_PLACEHOLDER, 
    MASK_PLACEHOLDER,
    OBJS_PLACEHOLDER,
    IMAGE_PLACEHOLDER,
    PHRASE_ST_PLACEHOLDER,
    PHRASE_ED_PLACEHOLDER,
    PHRASE_ST_PLACEHOLDER_STAGE2,
    PHRASE_ED_PLACEHOLDER_STAGE2
    )
from .mixin import MInstrDataset
from pycocotools import mask as maskUtils

@DATASETS.register_module()
class COCOInteract(MInstrDataset):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, placeholders=(IMAGE_PLACEHOLDER, MASK_PLACEHOLDER), **kwargs)
        self.max_gt_per_img = 15
        self.coco_class_name = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
            'stop sign', 'parking meter', 'bench', 'bird', 'cat',
            'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
            'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
            'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
            'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle',
            'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
            'banana', 'apple', 'sandwich', 'orange', 'broccoli',
            'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
            'couch', 'potted plant', 'bed', 'dining table', 'toilet',
            'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
            'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
            'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
      
    def annToMask(self, mask_ann, h, w):
        if isinstance(mask_ann, list):
            rles = maskUtils.frPyObjects(mask_ann, h, w)
            rle = maskUtils.merge(rles)
        elif isinstance(mask_ann['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(mask_ann, h, w)
        else:
            # rle
            rle = mask_ann
        mask = maskUtils.decode(rle)
        return mask
    
        
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        item = self.text_data[index]
        img_path = item['image']
        image_path_abs = os.path.join(self.image_folder,img_path)
        width = item['image_info']['width']
        height = item['image_info']['height']
        image = {{'path': image_path_abs,'width':width,'height':height}}

        #load annotations
        annotations = item['anns']

        gt_masks = []
        gt_labels = []
        point_visual_prompt_mask = []
        mask_visual_prompt_mask = []
        box_visual_prompt_mask = []
        scribble_visual_prompt_mask = []

        for annotation in annotations: 
            mask = self.annToMask(annotation['segmentation'], height, width)
            gt_masks.append(mask)
            point_visual_prompt_mask.append(decode(annotation['point_visual_prompt_mask']))
            mask_visual_prompt_mask.append(decode(annotation['mask_visual_prompt_mask']))
            box_visual_prompt_mask.append(decode(annotation['box_visual_prompt_mask']))
            scribble_visual_prompt_mask.append(decode(annotation['scribble_visual_prompt_mask']))
    
            cat = self.coco.loadCats(ann['category_id'])
            gt_labels.append(cat[0]['name'])

        #shuffle
        shuffle_ids = torch.randperm(len(gt_masks))
        if len(shuffle_ids) > self.max_gt_per_img:
            shuffle_ids = shuffle_ids[:self.max_gt_per_img]
        gt_masks = gt_masks[shuffle_ids]
        gt_labels = [gt_labels[i] for i in shuffle_ids]

         
        question = self.get_template().replace(OBJS_PLACEHOLDER, BOXES_PLACEHOLDER) #??????
        answer = gt_labels
        
        ret = {
            'image': image,
            'conversations': [
                {
                    'from': 'human',
                    'value': question,
                },
                {
                    'from': 'gpt',
                    'value': answer,
                }
            ]
        }
        return ret