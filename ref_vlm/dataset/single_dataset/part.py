import json
import torch
import numpy as np
import numpy as np
import torch
import json
import os
import re
import random
# from .stage2_data import CustomDataset
# from osprey.train.train import preprocess, preprocess_multimodal
from ref_vlm.registry import DATASETS
from ref_vlm.utils.constants import (
    MASKS_PLACEHOLDER,
    QUESTION_PLACEHOLDER,
    IMAGE_PLACEHOLDER,
    BOXES_PLACEHOLDER,
    OBJS_PLACEHOLDER,)
from .mixin import MInstrDataset
from .dataset_templates import DETAILED_QUESTIONS, WHY_QUESTIONS, Ref_WAY, QUESTIONS
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils

class CustomDataset(MInstrDataset):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, placeholders=(IMAGE_PLACEHOLDER, OBJS_PLACEHOLDER), **kwargs)
        self.begin_str = '' 
        self.coco = COCO(self.text_path)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.max_gt_per_img = 20
    
    def __len__(self):
        return len(self.ids)
    
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
    
    def shuffle(self, max_gt_per_img, gt_masks, gt_labels):
        shuffle_ids = torch.randperm(len(gt_masks))
        if len(shuffle_ids) > max_gt_per_img:
            shuffle_ids = shuffle_ids[:max_gt_per_img]

        gt_masks = np.array(gt_masks)
        gt_masks = torch.from_numpy(gt_masks) 

        gt_masks = gt_masks[shuffle_ids]
        gt_labels = [gt_labels[i] for i in shuffle_ids]
        return gt_masks, gt_labels
    

    def __getitem__(self, index):
        # coco = self.coco
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
     
        #load image info
        info = self.coco.loadImgs(img_id)[0]
        img_path = info['file_name']
        height = int(info['height'])
        width = int(info['width'])
        image = self.get_image(img_path)

        #load annotations
        ann_info = self.coco.loadAnns(ann_ids) 
        gt_masks = []
        gt_labels = []

        for ann in ann_info: 
            mask = self.annToMask(ann['segmentation'], height, width)
            gt_masks.append(mask)
            cat = self.coco.loadCats(ann['category_id'])
            gt_labels.append(cat[0]['name'])

        #shuffle
        gt_masks, gt_labels = self.shuffle(self.max_gt_per_img, gt_masks, gt_labels)

        # #check mask
        # load_and_display_mask(os.path.join(self.image_folder, img_path), gt_masks)

        #ret 
        ret = {
            'image': image,
            # 'target': {'boxes': gt_masks},
            'conversations': [

            ]
        }

        for i in range(len(gt_labels)):
            question = self.get_template().replace(OBJS_PLACEHOLDER, MASKS_PLACEHOLDER)
            # print(question)
            if i == 0:
                question = self.begin_str + question

            answer = gt_labels[i]
            ret['conversations'].append({'from': 'human', 'value': question}),
            ret['conversations'].append({'from': 'gpt', 'value': answer})

        return ret  

     
            


@DATASETS.register_module()
class PascalPart(CustomDataset):

    def __init__(self, *args, **kwargs):
        CAT_CLASSES = ('potted plant', 'aeroplane', 'cow', 'cat', 'bus', 'horse', 'car', 
                    'dog', 'bicycle', 'person', 'bird', 'bottle', 'sheep', 'motorbike')

        SUB_CLASSES = ('eye', 'window', 'cap', 'headlight', 'hand', 'mirror', 'arm', 'plant', 
                    'wheel', 'ear', 'pot', 'foot', 'leg', 'nose', 'body', 'horn', 'handlebar', 
                    'neck', 'license plate', 'paw', 'saddle', 'head', 'muzzle', 'tail', 'wing', 
                    'beak', 'hair', 'torso', 'door', 'mouth')

        begin_str = '<image>\n In the conversation below, you simply answer the category and subcategory name based on what you see' \
                            'in the image inside a particular region. It maybe a subpart of an object. '\
                            'I will give you only one region each time. Your answer should in the format of '\
                            'category:subcategory. '
        class_str = 'Categories Containing '+', '.join(CAT_CLASSES)+ '. '
        subclass_str = 'Subcategories Containing ' + ','.join(SUB_CLASSES)
        self.begin_str = begin_str + class_str + subclass_str + '.\n'
        self.max_gt_per_img = 15
        super().__init__(*args, **kwargs)
      

  
@DATASETS.register_module()
class PartImagenet(CustomDataset):

    def __init__(self, *args, **kwargs):
        CAT_CLASSES = (
            'Bottle', 'Biped', 'Quadruped', 'Fish', 'Reptile', 'Bicycle', 'Bird', 'Car', 'Boat', 'Snake', 'Aeroplane'
        )
        SUB_CLASSES = (
            'Tier', 'Hand', 'Wing', 'Mouth', 'Tail', 'Side', 'Fin', 'Engine', 'Foot', 'Head', 'Body', 'Sail', 'Seat'
        )
        begin_str = '<image>\n In the conversation below, you simply answer the category and subcategory name based on what you see' \
                            'in the image inside a particular region. It maybe a subpart of an object. '\
                            'I will give you only one region each time. Your answer should in the format of '\
                            'category:subcategory. '
        class_str = 'Categories Containing '+', '.join(CAT_CLASSES)+ '. '
        subclass_str = 'Subcategories Containing ' + ','.join(SUB_CLASSES)
        self.begin_str = begin_str + class_str + subclass_str + '.\n'
        self.max_gt_per_img = 15
        super().__init__(*args, **kwargs)