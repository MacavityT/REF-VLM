import os
import torch
import json
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from pycocotools.mask import decode
from PIL import Image
from xtuner.registry import DATASETS
import random

from xtuner.utils.constants import (
    BOXES_PLACEHOLDER, 
    MASKS_PLACEHOLDER,
    REGION_PLACEHOLDER,
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

    def __init__(self, *args, version,map_placeholders,max_conv_length=None, **kwargs):
        super().__init__(*args, placeholders=(IMAGE_PLACEHOLDER, REGION_PLACEHOLDER), **kwargs)
        self.version = version
        self.map_placeholders = map_placeholders
        self.coco_class_ids = [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17,
            18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
            35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49,
            50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
            64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
            82, 84, 85, 86, 87, 88, 89, 90
        ]
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
        self.length = max_conv_length
      
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
    
    def random_select(self,conversations,length=None,system_value=None):
        if length is None:
            length = len(conversations)

        shuffle_num = [i for i in range(len(conversations))]
        random.shuffle(shuffle_num)

        rand_num = random.randint(1,length)
        conversations = [conversations[i] for i in shuffle_num]
        
        if system_value is not None:
            assert len(conversations) == len(system_value), \
                "the length of conversations and system_values should be the same!"
            conversations = conversations[:rand_num]
            system_value = [system_value[i] for i in shuffle_num]
            system_value = system_value[:rand_num]
            return conversations,system_value
        
        conversations = conversations[:rand_num]
        return conversations
    
    def concat_conversations(self,conversations,concat_all=False):

        all_conversations = []
        for i,conversation in enumerate(conversations):
            if type(conversation) is list:
                for idx,j in enumerate(conversation):
                    all_conversations.append(j)
                    if concat_all:
                        # remove <image> for the rest of conversation
                        if not (i == 0 and idx == 0):   
                            if j['from'] == 'human':
                                j['value'] = j['value'].replace(IMAGE_PLACEHOLDER,'')
                    
            else:
                raise "Multi-conversations must be Lists !"  

        return all_conversations
    

    def __getitem__(self, index):
        item = self.text_data[index]
        img_path = item['image']
        image_path_abs = os.path.join(self.image_folder,img_path)
        width = item['image_info']['width']
        height = item['image_info']['height']
        image = {'path': image_path_abs,'width':width,'height':height}

        #load annotations
        annotations = item['anns']

        gt_masks = []
        gt_boxes = []
        all_conversations = []
        all_system_values = []
        for i,annotation in enumerate(annotations): 
            mask = self.annToMask(annotation['segmentation'], height, width)

            category_name = self.coco_class_name[self.coco_class_ids.index(annotation['category_id'])]
            gt_masks.append(mask)

            interact_list = [decode(annotation['box_visual_prompt_mask']),
                             decode(annotation['point_visual_prompt_mask']),
                             decode(annotation['scribble_visual_prompt_mask']),
                             decode(annotation['mask_visual_prompt_mask']),]
            
            for interact_mask in interact_list:
                gt_masks.append(interact_mask)
            
            question = self.get_template()
            question = question.replace(REGION_PLACEHOLDER,MASKS_PLACEHOLDER)

            if self.version == 's':
                random_num = random.randint(0,len(interact_list)-2)
                answer = category_name.replace(category_name,f'{PHRASE_ST_PLACEHOLDER_STAGE2}{category_name}{PHRASE_ED_PLACEHOLDER_STAGE2}{MASKS_PLACEHOLDER}')
                single_conversation = [
                    {'from':'human','value':question,'masks_seq':[[5*i+1+random_num]]},
                    {'from':'gpt','value':answer,'masks_seq':[[i]]}
                ]
                all_conversations.append(single_conversation)
                all_system_values.append([{'task':{'task_name':'segmentation','element':['phrase'],'use_unit':True},'unit':['mask']}])
    
            elif self.version == 'd':
                gt_boxes.append(annotation['bbox'])
                random_num = random.randint(1,len(interact_list)-1)
                answer = category_name.replace(category_name,f'{PHRASE_ST_PLACEHOLDER_STAGE2}{category_name}{PHRASE_ED_PLACEHOLDER_STAGE2}{BOXES_PLACEHOLDER}')
                single_conversation = [
                    {'from':'human','value':question,'masks_seq':[[5*i+1+random_num]]},
                    {'from':'gpt','value':answer,'boxes_seq':[[i]]}
                ]
                all_conversations.append(single_conversation)
                all_system_values.append([{'task':{'task_name':'detection','element':['phrase'],'use_unit':True},'unit':['box']}])
    

        #random shuffle
        all_conversations, all_system_values = self.random_select(conversations=all_conversations,
                                                                  length=self.length,
                                                                  system_value=all_system_values)
        all_system_values = self.concat_conversations(all_system_values)
        all_conversations = self.concat_conversations(all_conversations,concat_all=True)
        all_conversations.insert(0,[{'from':'system','value':all_system_values}]) 
        

        if self.version == 's':
            ret = {
                'image':image,
                'target':{'masks':gt_masks},
                'conversations': all_conversations
            }
        elif self.version == 'd':
            ret = {
                'image':image,
                'target':{'masks':gt_masks,'boxes':gt_boxes},
                'conversations': all_conversations
            }
        return ret