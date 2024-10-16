import os
import random
import copy
from PIL import Image
import numpy as np
import json
from torch.utils.data import Dataset
from registry import DATASETS
from pycocotools.mask import decode
import shutil
import cv2
from xtuner.dataset.utils import convert_bbox,visualize_mask,visualize_box
import pycocotools.mask as mask_utils

from utils.constants import (
    MASKS_PLACEHOLDER,
    BOXES_PLACEHOLDER,
    IMAGE_PLACEHOLDER,
    PHRASE_ST_PLACEHOLDER_STAGE2,
    PHRASE_ED_PLACEHOLDER_STAGE2,
    CLASS_PLACEHOLDER,
)
from collections import defaultdict
from .mixin import MInstrDataset

def find_duplicate_indices(my_list):
    index_dict = {}

    for idx, elem in enumerate(my_list):
        if elem in index_dict:
            index_dict[elem].append(idx)
        else:
            index_dict[elem] = [idx]

    return index_dict

@DATASETS.register_module()
class COCOREMDataset(MInstrDataset):
    def __init__(self, *args, task_type,**kwargs):
        super().__init__(*args, **kwargs)
        self.task_type = task_type
        self.img_name =  os.listdir(self.image_folder)
        self.dataset = self.read_json()
        self.createIndex()

    def read_json(self):
        with open(self.text_path) as f:
            img_json = json.loads(f.read())
        return img_json

    def createIndex(self):
        # create index
        print('creating index...')
        self.anns, self.cats = {}, {}
        self.imgToAnns = defaultdict(list)
        self.imgs = []
        if 'annotations' in self.dataset:
            for ann in self.dataset['annotations']:
                self.imgToAnns[ann['image_id']].append(ann)
                self.anns[ann['id']] = ann

        if 'images' in self.dataset:
            for img in self.dataset['images']:
                self.imgs.append(img)

        if 'categories' in self.dataset:
            for cat in self.dataset['categories']:
                self.cats[cat['id']] = cat

        print('index created!')

    def __len__(self):
        if (self.offline_processed_text_folder is not None) and \
            os.path.exists(self.offline_processed_text_folder):
            return len(self.text_data)
        else:
            return len(self.imgToAnns.keys())
        
    
    def replace_categories(self, category_id):
        category_name = self.cats[category_id]['name']
        return category_name
    
    def build_caption(self, index):
        boxes_masks = []
        types = []
        boxes_masks_seq = []
        info = self.imgs[index]
        if 'file_name' in info.keys():
            path = os.path.join(self.image_folder,info['file_name'])
        else:
            path = os.path.join(self.image_folder,f"{str(info['id']).zfill(12)}.jpg")
        img_info = {
            'path': path,
            'width': info['width'],
            'height': info['height'],
        }
        id = info['id']
        annotations = self.imgToAnns[id]
        for annotation in annotations:
            if self.task_type == 'mask':
                rleObjs = mask_utils.frPyObjects(annotation["segmentation"], info["height"], info["width"])
                mask = decode(rleObjs)
                boxes_masks.append(mask)
            elif self.task_type == 'box':
                box = list(convert_bbox(annotation['bbox']))
                boxes_masks.append(box)
            type = self.replace_categories(annotation['category_id'])
            types.append(type)

        index_dict = find_duplicate_indices(types)
        caption = ''
        for i, key in enumerate(index_dict.keys()):
            if self.task_type == 'mask':
                caption += (PHRASE_ST_PLACEHOLDER_STAGE2 + key + PHRASE_ED_PLACEHOLDER_STAGE2 +
                            MASKS_PLACEHOLDER * len(index_dict[key]) + (',' if i < len(index_dict.keys()) - 1 else '.'))
            elif self.task_type == 'box':
                caption += (PHRASE_ST_PLACEHOLDER_STAGE2 + key + PHRASE_ED_PLACEHOLDER_STAGE2 +
                            BOXES_PLACEHOLDER * len(index_dict[key]) + (',' if i < len(index_dict.keys()) - 1 else '.'))
            boxes_masks_seq.append(index_dict[key])
        return img_info, boxes_masks, caption, boxes_masks_seq
    
    
        
    def build_conversations(self, index):


        question = self.get_template()
        img_info,boxes_masks,caption,boxes_masks_seq = self.build_caption(index)
        human = {'from':'human','value':question}

        if self.task_type == 'mask':
            answer = {'from':'gpt','value':caption,'masks_seq':boxes_masks_seq}
            type = 'masks'
            task = {'task_name':'segmentation','element':['phrase'],'use_unit':True}
            unit = ['mask']
            system = {'from':'system','value':[{'task':task,'unit':unit}]}
        elif self.task_type == 'box':
            answer = {'from':'gpt','value':caption,'boxes_seq':boxes_masks_seq}
            type = 'boxes'
            task = {'task_name':'detection','element':['phrase'],'use_unit':True}
            unit = ['box']
            system = {'from':'system','value':[{'task':task,'unit':unit}]}


        conversation = [system, human, answer]
        ret = {
            'image':img_info,
            'target':{type: boxes_masks},
            'conversations':conversation
        }
        ret['map_placeholders'] = self.map_placeholders

        ori_path = 'vis_origin.jpg'
        shutil.copy(ret['image']['path'], ori_path)

        image = Image.open(ret['image']['path'])

        if 'masks' in ret['target'].keys():
            masks = ret['target']['masks']
            new_masks = []
            for j,mask in enumerate(masks):
                mask = Image.fromarray(mask)
                mask = mask.resize((int(image.width),int(image.height)), Image.LANCZOS)
                mask = np.array(mask)
                mask[mask!=0] = 1
                new_masks.append(mask)
                # vis_mask = visualize_mask_single(image, mask, alpha=1.0, beta=1.0)
                # save_path = f'vis_mask_{j}.jpg'
                # cv2.imwrite(save_path, vis_mask)
            image = visualize_mask(image,new_masks)
            image = Image.fromarray(image)
            image.save('vis_mask.jpg')
        
        if 'boxes' in ret['target'].keys():
            boxes = ret['target']['boxes']
            vis_box = visualize_box(image,boxes)
            # for k,box in enumerate(boxes):
                # denorm_box = de_norm_box_xyxy(box,width,height)
                # vis_box = visualize_box_single(image.copy(), box)
            save_path = f'vis_box.jpg'
            cv2.imwrite(save_path, vis_box)

        return ret
    
    def __getitem__(self, index):
        offline_item = super().__getitem__(index)
        if offline_item is not None:
            return offline_item
        conversations = self.build_conversations(index)
        return conversations
    
@DATASETS.register_module()
class LVISDataset(COCOREMDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    
@DATASETS.register_module()
class LVISTestDataset(COCOREMDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, index):
        image_info = self.imgs[index]

        if 'file_name' in image_info.keys():
            path = os.path.join(self.image_folder,image_info['file_name'])
        else:
            path = os.path.join(self.image_folder,f"{str(image_info['id']).zfill(12)}.jpg")

        img_info = {
            'path': path,
            'width': image_info['width'],
            'height': image_info['height'],
        }

        question = self.get_template()

        ret = {
            'image':img_info,
            'target':{'boxes':[[0,0,0,0]]},
            'conversations':[
                {'from': 'system', 'value': [{'task':{'task_name':'detection','element':['phrase'],'use_unit':True},'unit':['box']}]},
                {'from':'human','value':question},
                {'from': 'gpt','value':''},
                ]
        }
        ret['map_placeholders'] = self.map_placeholders

        return ret