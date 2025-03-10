import os
import random
import copy
from PIL import Image
import numpy as np
import json
from torch.utils.data import Dataset
from pycocotools.mask import decode
from ref_vlm.registry import DATASETS
from ref_vlm.utils.constants import (
    MASKS_PLACEHOLDER,
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
class ADE20k(MInstrDataset):
    def __init__(self, text_path, image_folder=None, target_type='semantic', gt_info=None, **kwargs):
        super().__init__(text_path, image_folder, placeholders=(IMAGE_PLACEHOLDER,), **kwargs)
        self.gt_info = gt_info
        self.mode = "gtFine" 
        self.target_type = target_type #  ("instance", "semantic", "polygon", "color", "depth")
        self.limit = " Answer the question using a short phrase."
        self.begin_str = """<image>\nThis provides an overview of the picture.\n"""
        self.max_gt_per_img = 15
        self.dataset = self.read_json()
        self.createIndex()

    def read_json(self):
        with open(self.text_path) as f:
            img_json = json.loads(f.read())
        return img_json

    def createIndex(self):
        # create index
        print('creating index...')
        self.anns, self.cats, self.imgs = {}, {}, {}
        self.img_name = []
        self.imgToAnns,self.catToImgs = defaultdict(list),defaultdict(list)
        if 'annotations' in self.dataset:
            for ann in self.dataset['annotations']:
                self.imgToAnns[ann['image_id']].append(ann)
                self.anns[ann['id']] = ann
                self.img_name.append(ann['image_id'])

        if 'images' in self.dataset:
            for img in self.dataset['images']:
                self.imgs[img['id']] = img

        if 'categories' in self.dataset:
            for cat in self.dataset['categories']:
                self.cats[cat['id']] = cat

        if 'annotations' in self.dataset and 'categories' in self.dataset:
            for ann in self.dataset['annotations']:
                self.catToImgs[ann['category_id']].append(ann['image_id'])

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
        masks = []
        types = []
        masks_seq = []
        image_name = self.img_name[index].split('.')[0]
        info = self.imgs[image_name]
        img_info = {
            'path': os.path.join(self.image_folder,info['file_name']),
            'width': info['width'],
            'height': info['height'],
        }
        annotations = self.imgToAnns[image_name]
        for annotation in annotations:
            mask = decode(annotation['segmentation'])
            masks.append(mask)
            type = self.replace_categories(annotation['category_id'])
            types.append(type)

        index_dict = find_duplicate_indices(types)
        caption = ''
        for i, key in enumerate(index_dict.keys()):
            caption += (PHRASE_ST_PLACEHOLDER_STAGE2 + key + PHRASE_ED_PLACEHOLDER_STAGE2 +
                        MASKS_PLACEHOLDER * len(index_dict[key]) + (',' if i < len(index_dict.keys()) - 1 else '.'))
            masks_seq.append(index_dict[key])
        return img_info, masks, caption, masks_seq
    
    def get_categories(self):
        img_json = self.read_json()
        categories = {}
        if self.target_type == 'semantic':
            for idx in range(len(img_json['categories'])):
                key = img_json['categories'][idx]['id'] + 1
                value = img_json['categories'][idx]['name']
                categories[key] = value
        elif self.target_type == 'panotic':
            for idx in range(len(img_json['categories'])):
                value = np.array(np.array(img_json['categories'][idx]['color']))
                key = img_json['categories'][idx]['name']
                categories[key] = value
        return categories
    
    def get_gt(self, img_name):
        img_path = self.gt_info + '/' + img_name + '.png'
        img = np.array(Image.open(img_path))
        if self.target_type == "semantic":
            mask_types = np.unique(img)
            mask_types = mask_types[~np.isin(mask_types, [0, 255])]
        elif self.target_type == 'panotic':
            mask_types = np.unique(img.reshape(-1, img.shape[-1]), axis=0)
            mask_type = mask_types[~np.isin(mask_types, [255, 255, 255])]
        types = []
        masks = []
        for mask_type in mask_types:
            mask = np.zeros_like(img, dtype=np.uint8)
            mask[img == mask_type] = 1
            masks.append(mask)
            types.append(mask_type)
        return masks, types
    
    def flatten_annotation(self, img_name):
        masks, types = self.get_gt(img_name)
        categories = self.get_categories()
        if self.target_type == 'semantic':
            combined = list(zip(masks, [categories.get(t) for t in types]))
        elif self.target_type == 'panotic':
            combined = list(zip(masks, [categories.get(t) for t in types]))
        random.shuffle(combined)
        masks, labels = zip(*combined) if combined else ([], [])
        return masks,labels
        
    def build_conversations(self, index):
        if self.template_name == 'SEG':

            type = 'masks'
            task = {'task_name':'segmentation','element':['phrase'],'use_unit':True}
            unit = ['mask']
            system = {'from':'system','value':[{'task':task,'unit':unit}]}
            question = self.get_template()
            img_info,masks,caption,masks_seq = self.build_caption(index)
            human = {'from':'human','value':question}
            answer = {'from':'gpt','value':caption,'masks_seq':masks_seq}
            conversation = [system, human, answer]
            ret = {
                'image':img_info,
                'target':{type: masks},
                'conversations':conversation
            }
            ret['map_placeholders'] = self.map_placeholders
            return ret
        
        elif self.template_name == 'Cond_SEG':
            conversations = []
            img_info = self.get_Imginfo(index)
            masks,categories = self.flatten_annotation(self.img_name[index].split('.')[0])
            for j in range(len(categories)):
                type = 'masks'
                question = self.get_template().replace(CLASS_PLACEHOLDER, categories[j])
                human = {'from':'human','value':question}
                caption = MASKS_PLACEHOLDER
                answer = {'from':'gpt','value':caption}
                conversation = [human,answer]
                ret = {
                    'image': img_info,
                    'target':{type:masks[j]},
                    'conversations':conversation
                }
                conversations.append(ret)
            conversations['map_placeholders'] = self.map_placeholders
            return conversations
        
    
    def random_select_and_remove(self):
        if len(self.list) == 0:
            return None, []
        self.list = list(range(len(self.img_name)))
        selected_value = random.choice(self.list)
        self.list.remove(selected_value)

        return selected_value
    

    def __getitem__(self, index):
        offline_item = super().__getitem__(index)
        if offline_item is not None:
            return offline_item
        if self.target_type == 'instance':
            conversations = self.build_conversations(index)
            return conversations
        elif self.target_type == 'semantic':
            idx = self.random_select_and_remove()
            conversations = self.build_conversations(idx)
            ret = conversations[index]
            return ret
        elif self.target_type == 'panotic':
            idx = self.random_select_and_remove()
            conversations = self.build_conversations(idx)
            ret = conversations[index]
            return ret
    
