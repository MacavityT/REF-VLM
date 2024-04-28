import json
import torch
import numpy as np
from xtuner.registry import DATASETS
from xtuner.utils.constants import (
    QUESTION_PLACEHOLDER,
    IMAGE_PLACEHOLDER,
    BOXES_PLACEHOLDER,
    OBJS_PLACEHOLDER,
    MASK_PLACEHOLDER)
from .mixin import MInstrDataset
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils



# https://github.com/open-mmlab/mmpretrain/blob/17a886cb5825cd8c26df4e65f7112d404b99fe12/mmpretrain/datasets/refcoco.py#L14
# https://github.com/CircleRadon/Osprey/blob/c61d6df3ebc259841add1a542a01a7a88f84fdb4/osprey/datasets/stage2_data.py

@DATASETS.register_module()
class COCODataset(MInstrDataset):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, placeholders=(IMAGE_PLACEHOLDER, OBJS_PLACEHOLDER), **kwargs)
        self.coco = COCO(self.text_path)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.max_gt_per_img = 20
      
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
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
     
        #load image info
        info = coco.loadImgs(img_id)[0]
        img_path = info['file_name']
        height = int(info['height'])
        width = int(info['width'])
        image = self.get_image(img_path)

        #load annotations
        ann_info = coco.loadAnns(ann_ids) 

        gt_masks = []
        gt_labels = []

        for ann in ann_info: 
            mask = self.annToMask(ann['segmentation'], height, width)
            gt_masks.append(mask)
            cat = self.coco.loadCats(ann['category_id'])
            gt_labels.append(cat[0]['name'])
        
        #shuffle ??????????????
        shuffle_ids = torch.randperm(len(gt_masks))
        if len(shuffle_ids) > self.max_gt_per_img:
            shuffle_ids = shuffle_ids[:self.max_gt_per_img]
        gt_masks = np.array(gt_masks)
        gt_masks = torch.from_numpy(gt_masks) 
        gt_masks = gt_masks[shuffle_ids]
        gt_labels = [gt_labels[i] for i in shuffle_ids]

         
        # question = self.get_template().replace(OBJS_PLACEHOLDER, BOXES_PLACEHOLDER)
        # answer = gt_labels

        ret = {
            'image': image,
            # 'target': {'boxes': gt_masks},
            'conversations': [
                # {
                #     'from': 'human',
                #     'value': question,
                # },
                # {
                #     'from': 'gpt',
                #     'value': answer,
                # }
            ]
        }

        #ret 
        for i in range(len(gt_labels)):
            question = self.get_template().replace(OBJS_PLACEHOLDER, MASK_PLACEHOLDER)
            answer = gt_labels[i]
            ret['conversations'].append({'from': 'human', 'value': question}),
            ret['conversations'].append({'from': 'gpt', 'value': answer})

        return ret

     
@DATASETS.register_module()
class RefCOCO(MInstrDataset):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, placeholders=(IMAGE_PLACEHOLDER, OBJS_PLACEHOLDER), **kwargs)
        self.coco = COCO(self.text_path)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.max_gt_per_img = 15
    
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
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
     
        #load image info
        info = coco.loadImgs(img_id)[0]
        img_path = info['file_name']
        height = int(info['height'])
        width = int(info['width'])
        caption = info['caption']
        image = self.get_image(img_path)

        #load annotations
        ann_info = coco.loadAnns(ann_ids) 

        gt_masks = []
        gt_labels = []

        for ann in ann_info: 
            mask = self.annToMask(ann['segmentation'], height, width)
            gt_masks.append(mask)
            # cat = self.coco.loadCats(target['category_id'])
            gt_labels.append(caption) #different

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
    

@DATASETS.register_module()
class RefCOCOP(MInstrDataset):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, placeholders=(IMAGE_PLACEHOLDER, OBJS_PLACEHOLDER), **kwargs)
        self.coco = COCO(self.text_path)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.max_gt_per_img = 15
      
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
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
     
        #load image info
        info = coco.loadImgs(img_id)[0]
        img_path = info['file_name']
        height = int(info['height'])
        width = int(info['width'])
        image = self.get_image(img_path)

        #load annotations
        ann_info = coco.loadAnns(ann_ids) 

        gt_masks = []
        gt_labels = []

        for ann in ann_info: 
            mask = self.annToMask(ann['segmentation'], height, width)
            gt_masks.append(mask)
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


@DATASETS.register_module()
class COCOInteract(MInstrDataset):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, placeholders=(IMAGE_PLACEHOLDER, OBJS_PLACEHOLDER), **kwargs)
        self.coco = COCO(self.text_path)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.max_gt_per_img = 15
      
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
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
     
        #load image info
        info = coco.loadImgs(img_id)[0]
        img_path = info['file_name']
        height = int(info['height'])
        width = int(info['width'])
        image = self.get_image(img_path)

        #load annotations
        ann_info = coco.loadAnns(ann_ids) 

        gt_masks = []
        gt_labels = []
        point_visual_prompt_mask = []
        mask_visual_prompt_mask = []
        box_visual_prompt_mask = []
        scribble_visual_prompt_mask = []

        for ann in ann_info: 
            mask = self.annToMask(ann['segmentation'], height, width)
            gt_masks.append(mask)
            point_visual_prompt_mask.append(ann['point_visual_prompt_mask'])
            mask_visual_prompt_mask.append(ann['mask_visual_prompt_mask'])
            box_visual_prompt_mask.append(ann['box_visual_prompt_mask'])
            scribble_visual_prompt_mask.append(ann['scribble_visual_prompt_mask'])
    
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


@DATASETS.register_module()
class PascalPart(MInstrDataset):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, placeholders=(IMAGE_PLACEHOLDER, OBJS_PLACEHOLDER), **kwargs)
        self.coco = COCO(self.text_path)
        self.ids = list(sorted(self.coco.imgs.keys()))
        CAT_CLASSES = (
            'Bottle', 'Biped', 'Quadruped', 'Fish', 'Reptile', 'Bicycle', 'Bird', 'Car', 'Boat', 'Snake', 'Aeroplane'
        )

        SUB_CLASSES = (
            'Tier', 'Hand', 'Wing', 'Mouth', 'Tail', 'Side', 'Fin', 'Engine', 'Foot', 'Head', 'Body', 'Sail', 'Seat'
        )
        class_str = 'Categories Containing '+', '.join(CAT_CLASSES)+ '. '
        subclass_str = 'Subcategories Containing ' + ','.join(SUB_CLASSES)
        self.begin_str = begin_str + class_str + subclass_str + '.\n'
        self.max_gt_per_img = 20
      
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
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
     
        #load image info
        info = coco.loadImgs(img_id)[0]
        img_path = info['file_name']
        height = int(info['height'])
        width = int(info['width'])
        image = self.get_image(img_path)

        #load annotations
        ann_info = coco.loadAnns(ann_ids) 

        gt_masks = []
        gt_labels = []

        for ann in ann_info: 
            mask = self.annToMask(ann['segmentation'], height, width)
            gt_masks.append(mask)
            cat = self.coco.loadCats(ann['category_id'])
            gt_labels.append(cat[0]['name'])
        
        #shuffle ??????????????
        shuffle_ids = torch.randperm(len(gt_masks))
        if len(shuffle_ids) > self.max_gt_per_img:
            shuffle_ids = shuffle_ids[:self.max_gt_per_img]
        gt_masks = np.array(gt_masks)
        gt_masks = torch.from_numpy(gt_masks) 
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
    

  
@DATASETS.register_module()
class PartImagenet(MInstrDataset):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, placeholders=(IMAGE_PLACEHOLDER, OBJS_PLACEHOLDER), **kwargs)
        self.coco = COCO(self.text_path)
        self.ids = list(sorted(self.coco.imgs.keys()))
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
        self.max_gt_per_img = 20
      
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
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
     
        #load image info
        info = coco.loadImgs(img_id)[0]
        img_path = info['file_name']
        height = int(info['height'])
        width = int(info['width'])
        image = self.get_image(img_path)

        #load annotations
        ann_info = coco.loadAnns(ann_ids) 

        gt_masks = []
        gt_labels = []

        for ann in ann_info: 
            mask = self.annToMask(ann['segmentation'], height, width)
            gt_masks.append(mask)
            cat = self.coco.loadCats(ann['category_id'])
            gt_labels.append(cat[0]['name'])
        
        #shuffle ??????????????
        shuffle_ids = torch.randperm(len(gt_masks))
        if len(shuffle_ids) > self.max_gt_per_img:
            shuffle_ids = shuffle_ids[:self.max_gt_per_img]
        gt_masks = np.array(gt_masks)
        gt_masks = torch.from_numpy(gt_masks) 
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