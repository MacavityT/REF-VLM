import json
import json
import os
import re
import random
import numpy as np
from collections import namedtuple
from typing import Dict, List
from PIL import Image
from xtuner.registry import DATASETS
from collections import defaultdict
import pycocotools.mask as mask_utils
from pycocotools.mask import decode
from xtuner.utils.constants import (
    IMAGE_PLACEHOLDER,
    QUESTION_PLACEHOLDER, 
    OBJS_PLACEHOLDER,
    BOXES_PLACEHOLDER,
    MASKS_PLACEHOLDER,
    PHRASE_ST_PLACEHOLDER_STAGE2,
    PHRASE_ED_PLACEHOLDER_STAGE2,
    EXPR_PLACEHOLDER,
    CLASS_PLACEHOLDER
)
from .mixin import MInstrDataset
from xtuner.registry import DATASETS
from xtuner.utils.constants import IMAGE_PLACEHOLDER


@DATASETS.register_module()
class LLAVAGrounding(MInstrDataset):
    def __init__(self, *args, version, anno_path, **kwargs):
        self.version = version
        self.anno_path = anno_path
        assert self.version in ['reg','gcg']
        super().__init__(*args, **kwargs)
        self.load_annotations(self.anno_path)
        self.createIndex()

    def load_annotations(self,anno_path):
        with open(anno_path) as f:
            self.dataset = json.loads(f.read())
            f.close()

    def createIndex(self):
        # create index
        print('creating index...')
        self.anns, self.cats, self.imgs = {}, {}, {}

        if 'annotations' in self.dataset:
            for ann in self.dataset['annotations']:
                self.anns[ann['id']] = ann

        if 'images' in self.dataset:
            for img in self.dataset['images']:
                self.imgs[img['file_name'].split('_')[2][:-4]] = img

        if 'categories' in self.dataset:
            for cat in self.dataset['categories']:
                self.cats[cat['id']] = cat

        print('index created!')

    def __getitem__(self, index):
        offline_item = super().__getitem__(index)
        if offline_item is not None:
            return offline_item
        
        item = self.text_data[index]
        image = self.imgs[item['id']]
        image_info = {'path':os.path.join(self.image_folder,image['file_name']),
                      'height':image['height'],
                      'width':image['width']}
        anno_index = None
        if 'gd_ls' in item:
            anno_index = item['gd_ls']
        elif 'q_gd_ls' in item:
            anno_index = item['q_gd_ls']
        assert anno_index is not None
        gt_masks = []
        conversations = []
        flag = True
        if self.version == 'reg':
            assert len(anno_index) == len(item['conversations']) // 2
            for i,conversation in enumerate(item['conversations']):
                if conversation['from'] == 'human':
                    assert i % 2 == 0
                    count = conversation['value'].count('<obj>')
                    if count > 0:
                        assert anno_index[i // 2] is not None
                        masks_seq = []
                        idx_list = anno_index[i // 2]
                        for idx in idx_list:
                            annotation = self.anns[idx]
                            rleObjs = mask_utils.frPyObjects(annotation["segmentation"], image_info["height"], image_info["width"])
                            mask = decode(rleObjs)
                            if len(mask.shape) == 3:
                                gt_masks.append(mask[:,:,0])
                            elif len(mask.shape) == 2:
                                gt_masks.append(mask)
                            else:
                                raise f"mask shape is invalid: {mask.shape}"
                            masks_seq.append(len(gt_masks)-1)
                        
                        conversation['value'] = conversation['value'].replace('<obj>',MASKS_PLACEHOLDER)
                        conversation['masks_seq'] = [masks_seq]
                    else:
                        assert anno_index[i // 2] is None, f"wrong anno index:{anno_index}"
            system = {
                'from':'system',
                'value': [{'task':{'task_name':'vqa','element':['sentence'],'use_unit':False}} for _ in range(len(item['conversations'])//2)],
                }
            

        elif self.version == 'gcg':
            for i,conversation in enumerate(item['conversations']):
                if conversation['from'] == 'gpt':
                    assert i % 2 == 1
                    segments = conversation['value'].split("<seg>")
                    if len(segments) - 1 != len(anno_index):
                            print(conversation)
                            print(f"Mismatch between the number of <seg> tags: {len(segments) - 1} and the gd_ls list lengths: {len(anno_index)}")
                            flag = False
                    
                    new_caption = segments[0]
                    masks_seq = []
                    for idx, sublist in enumerate(anno_index):
                        seg_insert = "<seg>" * len(sublist)
                        new_caption += seg_insert + segments[idx+1]
                        
                        mask_seq = []
                        for i,mask_idx in enumerate(sublist):
                            annotation = self.anns[mask_idx]
                            rleObjs = mask_utils.frPyObjects(annotation["segmentation"], image_info["width"], image_info["height"])
                            mask = decode(rleObjs)
                            if len(mask.shape) == 3:
                                gt_masks.append(mask[:,:,0])
                            elif len(mask.shape) == 2:
                                gt_masks.append(mask)
                            else:
                                raise f"mask shape is invalid: {mask.shape}"
                            mask_seq.append(len(gt_masks)-1)
                        masks_seq.append(mask_seq)

                    conversation['value'] = new_caption
                    conversation['value'] = conversation['value'].replace('<g_s>',PHRASE_ST_PLACEHOLDER_STAGE2)
                    conversation['value'] = conversation['value'].replace('<g_e>',PHRASE_ED_PLACEHOLDER_STAGE2)
                    conversation['value'] = conversation['value'].replace('<seg>',MASKS_PLACEHOLDER)
                    conversation['masks_seq'] = masks_seq

                    phrase_contents = re.findall(r'<Phrase>(.*?)</Phrase>', conversation['value'])
                    mask_counts = [len(re.findall(r'<masks>', content)) for content in phrase_contents]
                    for mask_count in mask_counts:
                        if mask_count != 0:
                            print("wrong conversation, set it to []")
                            flag = False
                    
                    system = {
                        'from': 'system',
                        'value':[{'task':{'task_name':'gcg_segmentation','element':['phrase','sentence'],'use_unit':True},'unit':['mask']}]
                    }



        conversations.append(system)
        conversations.extend(item['conversations'])
        
        if flag == False:
            return []

        ret = {
            'image':image_info,
            'target':{'masks': gt_masks},
            'conversations':conversations,
        }

        ret['map_placeholders'] = self.map_placeholders
        return ret