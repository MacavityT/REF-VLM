import os
import random
import copy
from PIL import Image
import numpy as np
import json
from torch.utils.data import Dataset
from xtuner.registry import DATASETS
from pycocotools.mask import decode
from xtuner.dataset.utils import convert_bbox
import pycocotools.mask as mask_utils
from xtuner.utils.constants import (
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
        return len(self.imgs)
        
    
    def replace_categories(self, category_id):
        category_name = self.cats[category_id]['name']
        return category_name
    
    def build_caption(self, index):
        boxes_masks = []
        types = []
        boxes_masks_seq = []
        info = self.imgs[index]
        img_info = {
            'path': os.path.join(self.image_folder,info['file_name']),
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
        return ret
    
        
    def __getitem__(self, index):
        offline_item = super().__getitem__(index)
        if offline_item is not None:
            return offline_item
        conversations = self.build_conversations(index)
        return conversations

    
