import os
import torch
import json
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from pycocotools.mask import decode,encode
from PIL import Image
from xtuner.registry import DATASETS
import cv2
import random
import numpy as np
import jsonlines
import pickle
from tqdm import tqdm
from xtuner.dataset.utils import convert_bbox, visualize_box_single
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

def coco_encode_rle(uncompressed_rle):
    h, w = uncompressed_rle["size"]
    # rle = maskUtils.frPyObjects(uncompressed_rle, h, w)
    uncompressed_rle["counts"] = uncompressed_rle["counts"].decode("utf-8")  # Necessary to serialize with json
    return uncompressed_rle

@DATASETS.register_module()
class COCOInteract(MInstrDataset):

    def __init__(self, *args, version, strategy='random', max_conv_length=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.strategy = strategy
        assert self.strategy in ['random','full']
        self.version = version
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
    
    def select_target(self,item,version):

        if version == 'd':
            target = item['target']
            boxes = target['boxes']
            masks = target['masks']
            conversations = item['conversations']

            selected_masks = []
            selected_boxs = []
            for i, conversation in enumerate(conversations):
                if i % 2 == 0:
                    seq = conversation['masks_seq'][0][0]
                    selected_masks.append(masks[seq])
                    assert 'masks_seq' in item['conversations'][i].keys()
                    item['conversations'][i]['masks_seq'] = [[len(selected_masks)-1]]
                else:
                    seq = conversation['boxes_seq'][0][0]
                    selected_boxs.append(boxes[seq])
                    assert 'boxes_seq' in item['conversations'][i].keys()
                    item['conversations'][i]['boxes_seq'] = [[len(selected_boxs)-1]]

            item['target']['boxes'] = selected_boxs
            item['target']['masks'] = selected_masks 
        
        elif version == 's':
            target = item['target']
            masks = target['masks']
            conversations = item['conversations']
            selected_masks = []

            for i, conversation in enumerate(conversations):
                seq = conversation['masks_seq'][0][0]
                selected_masks.append(masks[seq])
                assert 'masks_seq' in item['conversations'][i].keys()
                item['conversations'][i]['masks_seq'] = [[len(selected_masks)-1]]

            item['target']['masks'] = selected_masks

        elif version == 'r':
            target = item['target']
            masks = target['masks']
            conversations = item['conversations']
            selected_masks = []

            for i, conversation in enumerate(conversations):
                if 'masks_seq' in conversation.keys():
                    seq = conversation['masks_seq'][0][0]
                    selected_masks.append(masks[seq])
                    assert 'masks_seq' in item['conversations'][i].keys()
                    item['conversations'][i]['masks_seq'] = [[len(selected_masks)-1]]

            item['target']['masks'] = selected_masks


    
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
        offline_item = super().__getitem__(index)
        if offline_item is not None:
            return offline_item
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
        selected_mask = []
        for i,annotation in enumerate(annotations): 
            mask = self.annToMask(annotation['segmentation'], height, width)

            category_name = self.coco_class_name[self.coco_class_ids.index(annotation['category_id'])]
            gt_masks.append(mask)
            kernel = np.ones((10,10),np.uint8)
            point_mask = decode(annotation['point_visual_prompt_mask'])
            point_mask = cv2.dilate(point_mask,kernel,iterations=1)
            scribble_mask = decode(annotation['scribble_visual_prompt_mask'])
            scribble_mask = cv2.dilate(scribble_mask,kernel,iterations=1)
            interact_list = [decode(annotation['box_visual_prompt_mask']),
                             point_mask,
                             scribble_mask,
                             decode(annotation['mask_visual_prompt_mask']),]
            
            for interact_mask in interact_list:
                gt_masks.append(interact_mask)
            
            question = self.get_template()
            

            if self.version == 's':
                if self.strategy == 'random':
                    question = question.replace(REGION_PLACEHOLDER,MASKS_PLACEHOLDER)
                    random_num = random.randint(0,len(interact_list)-2)
                    answer = category_name.replace(category_name,f'{PHRASE_ST_PLACEHOLDER_STAGE2}{category_name}{PHRASE_ED_PLACEHOLDER_STAGE2}{MASKS_PLACEHOLDER}')
                    single_conversation = [
                        {'from':'human','value':question,'masks_seq':[[5*i+1+random_num]]},
                        {'from':'gpt','value':answer,'masks_seq':[[5*i]]}
                    ]
                    all_conversations.append(single_conversation)
                    all_system_values.append([{'task':{'task_name':'grounding_segmentation','element':['phrase'],'use_unit':True},'unit':['mask']}])
    
                
                elif self.strategy == 'full':
                    for k in range(0,len(interact)-2):
                        question = question.replace(REGION_PLACEHOLDER,MASKS_PLACEHOLDER)
                        answer = category_name.replace(category_name,f'{PHRASE_ST_PLACEHOLDER_STAGE2}{category_name}{PHRASE_ED_PLACEHOLDER_STAGE2}{MASKS_PLACEHOLDER}')
                        single_conversation = [
                                    {'from':'human','value':question,'masks_seq':[[5*i+1+k]]},
                                    {'from':'gpt','value':answer,'masks_seq':[[5*i]]}
                                ]
                        all_conversations.append(single_conversation)
                        all_system_values.append([{'task':{'task_name':'grounding_segmentation','element':['phrase'],'use_unit':True},'unit':['mask']}])
                        

            if self.version == 'r':
                answer =  category_name
                if self.strategy == 'random':
                    random_num = random.randint(0,len(interact_list)-1)
                    selected_mask.append(interact_list[random_num])
                    masks_seq = [[len(selected_mask)-1]]
                    single_conversation = [
                        {'from':'human','value':question,'masks_seq':masks_seq},
                        {'from':'gpt','value':answer}
                    ]
                    all_conversations.append(single_conversation)
                    all_system_values.append([{'task':{'task_name':'vqa','element':['sentence'],'use_unit':False}}])

                elif self.strategy == 'full':
                    for j,interact in enumerate(interact_list):
                        selected_mask.append(interact)
                        mask_seq = [[len(selected_mask)-1]]
                        single_conversation = [
                            {'from':'human','value':question,'masks_seq':masks_seq},
                            {'from':'gpt','value':answer}
                        ]
                        all_conversations.append(single_conversation)
                        all_system_values.append([{'task':{'task_name':'vqa','element':['sentence'],'use_unit':False}}])  
             
            elif self.version == 'd':
                bbox = annotation['bbox']
                bbox = convert_bbox(bbox)

                # image_pil = Image.open(image['path']).convert('RGB')
                # vis_box = visualize_box_single(image_pil,bbox)
                # save_path = f'vis_box_{i}.jpg'
                # cv2.imwrite(save_path, vis_box)

                gt_boxes.append(bbox)
                if self.strategy == 'random':
                    question = question.replace(REGION_PLACEHOLDER,MASKS_PLACEHOLDER)
                    random_num = random.randint(1,len(interact_list)-1)
                    selected_mask.append(interact_list[random_num])
                    masks_seq = [[len(selected_mask)-1]]
                    answer = category_name.replace(category_name,f'{PHRASE_ST_PLACEHOLDER_STAGE2}{category_name}{PHRASE_ED_PLACEHOLDER_STAGE2}{BOXES_PLACEHOLDER}')
                    single_conversation = [
                        {'from':'human','value':question,'masks_seq':masks_seq},
                        {'from':'gpt','value':answer,'boxes_seq':[[i]]}
                    ]
                    all_conversations.append(single_conversation)
                    all_system_values.append([{'task':{'task_name':'grounding_detection','element':['phrase'],'use_unit':True},'unit':['box']}])
                   

                elif self.strategy == 'full':
                    for m in range(1,len(interact)-1):
                        question = question.replace(REGION_PLACEHOLDER,MASKS_PLACEHOLDER)
                        selected_mask.append(interact_list[m])
                        masks_seq = [[len(selected_mask)-1]]
                        answer = category_name.replace(category_name,f'{PHRASE_ST_PLACEHOLDER_STAGE2}{category_name}{PHRASE_ED_PLACEHOLDER_STAGE2}{BOXES_PLACEHOLDER}')
                        single_conversation = [
                            {'from':'human','value':question,'masks_seq':masks_seq},
                            {'from':'gpt','value':answer,'boxes_seq':[[i]]}
                        ]
                        all_conversations.append(single_conversation)
                        all_system_values.append([{'task':{'task_name':'grounding_detection','element':['phrase'],'use_unit':True},'unit':['box']}])
    

        #random shuffle
        if self.strategy == 'random':
            all_conversations, all_system_values = self.random_select(conversations=all_conversations,
                                                                    length=self.length,
                                                                    system_value=all_system_values)
        elif self.strategy == 'full':
            all_conversations, all_system_values = self.random_select(conversations=all_conversations,
                                                                    length=None,
                                                                    system_value=all_system_values)            
        all_system_values = self.concat_conversations(all_system_values)
        all_conversations = self.concat_conversations(all_conversations,concat_all=True)
        

        if self.version == 's':
            ret = {
                'image':image,
                'target':{'masks':gt_masks},
                'conversations': all_conversations
            }
        elif self.version == 'r':
            ret = {
                'image':image,
                'target':{'masks':selected_mask},
                'conversations': all_conversations
            }
        elif self.version == 'd':
            ret = {
                'image':image,
                'target':{'masks':selected_mask,'boxes':gt_boxes},
                'conversations': all_conversations
            }
        
        self.select_target(ret,self.version)
        ret['map_placeholders'] = self.map_placeholders

        ret['conversations'].insert(0,{'from':'system','value':all_system_values}) 
        return ret
    


@DATASETS.register_module()
class COCOInteractSingle(MInstrDataset):

    def __init__(self, *args, version, split='val', **kwargs):
        super().__init__(*args, **kwargs)
        self.split = split
        assert self.split in ['train','val']
        self.version = version
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
        self.save_data()

        self.question = 'Please generate a distinguishing description for the region <masks> in the image<image>.'
      
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
    
    def save_data(self):

        self.all_items = []
        for item in tqdm(self.text_data):
            img_path = item['image']
            image_path_abs = os.path.join(self.image_folder,img_path)
            width = item['image_info']['width']
            height = item['image_info']['height']
            image = {'path': image_path_abs,'width':width,'height':height}

            #load annotations
            annotations = item['anns']

            for i,annotation in enumerate(annotations): 
                mask = annotation['segmentation']
                # mask = self.annToMask(annotation['segmentation'], height, width)
                box = convert_bbox(annotation['bbox'])
                category_name = self.coco_class_name[self.coco_class_ids.index(annotation['category_id'])]
                kernel = np.ones((10,10),np.uint8)
                point_mask = decode(annotation['point_visual_prompt_mask'])
                point_mask = cv2.dilate(point_mask,kernel,iterations=1)
                scribble_mask = decode(annotation['scribble_visual_prompt_mask'])
                scribble_mask = cv2.dilate(scribble_mask,kernel,iterations=1)
                interact_list = [decode(annotation['box_visual_prompt_mask']),
                                point_mask,
                                scribble_mask,
                                decode(annotation['mask_visual_prompt_mask'])]
                interact_list = [encode(np.array(mask,order='F',dtype='uint8')) for mask in interact_list]
                for interact in interact_list:
                    interact = coco_encode_rle(interact)
                # for interact in interact_list:
                single_item = {
                    'prompt':interact_list,
                    'target':mask,
                    'image':image,
                    'category_name':category_name,
                }

                self.all_items.append(single_item)


        with open("/data/Aaronzhu/DatasetStage2and3/COCO_interactive/interactive_val_single.json","w") as f:
            json.dump(self.all_items,f)
            f.close()

    def __len__(self):
        return len(self.all_items)                        
                        
    def __getitem__(self, index):
        offline_item = super().__getitem__(index)
        if offline_item is not None:
            return offline_item
        
        item = self.all_items[index]
        if self.split == 'train':
            question = self.get_template()
        elif self.split == 'val':
            question = self.question
        category_name = item['category_name']

        if self.version == 's':
            question = question.replace(REGION_PLACEHOLDER,MASKS_PLACEHOLDER)
            answer = category_name.replace(category_name,f'{PHRASE_ST_PLACEHOLDER_STAGE2}{category_name}{PHRASE_ED_PLACEHOLDER_STAGE2}{MASKS_PLACEHOLDER}')
            single_conversation = [
                {'from':'system','value':[{'task':{'task_name':'grounding_segmentation','element':['phrase'],'use_unit':True},'unit':['mask']}]},
                {'from':'human','value':question,'masks_seq':[[0]]},
                {'from':'gpt','value':answer,'masks_seq':[[1]]}
            ]

        elif self.version == 'r':
            answer = category_name
            single_conversation = [
                {'from':'system','value':[{'task':{'task_name':'vqa','element':['sentence'],'use_unit':False}}]},
                {'from':'human','value':question,'masks_seq':[[0]]},
                {'from':'gpt','value':answer}
            ]
        
        elif self.version == 'd':
            question = question.replace(REGION_PLACEHOLDER,MASKS_PLACEHOLDER)
            answer = category_name.replace(category_name,f'{PHRASE_ST_PLACEHOLDER_STAGE2}{category_name}{PHRASE_ED_PLACEHOLDER_STAGE2}{BOXES_PLACEHOLDER}')
            single_conversation = [
                {'from':'system','value':[{'task':{'task_name':'grounding_detection','element':['phrase'],'use_unit':True},'unit':['box']}]},
                {'from':'human','value':question,'masks_seq':[[0]]},
                {'from':'gpt','value':answer,'boxes_seq':[[0]]}
            ]

        ret = {
            'image': item['image'],
            'target': item['target'],
            'conversations': single_conversation,
            'map_placeholders': self.map_placeholders
        }

        return ret


@DATASETS.register_module()
class COCOInteractSingleTask(MInstrDataset):

    def __init__(self, *args, version, split='val', **kwargs):
        super().__init__(*args, **kwargs)
        self.split = split
        assert self.split in ['train','val']
        self.version = version
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
        self.dataset = self.read_json()

        self.version_dict = {
            'box': 0,
            'point': 1,
            'scribble': 2,
            'mask': 3,
        }

    def read_json(self):
        with open(self.text_path) as f:
            img_json = json.loads(f.read())
        return img_json
    
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
    
    def __getitem__(self, index):
        item = self.dataset[index]
        image = item['image']
        question = self.get_template()
        category_name = item['category_name']

        question = question.replace(REGION_PLACEHOLDER,MASKS_PLACEHOLDER)
        answer = category_name.replace(category_name,f'{PHRASE_ST_PLACEHOLDER_STAGE2}{category_name}{PHRASE_ED_PLACEHOLDER_STAGE2}{MASKS_PLACEHOLDER}')
        single_conversation = [
                {'from':'system','value':[{'task':{'task_name':'grounding_segmentation','element':['phrase'],'use_unit':True},'unit':['mask']}]},
                {'from':'human','value':question,'masks_seq':[[0]]},
                {'from':'gpt','value':answer,'masks_seq':[[1]]}
            ]
        
        prompt = decode(item['prompt'][self.version_dict[self.version]])
        target = self.annToMask(item['target'],image['height'],image['width'])

        ret = {
            'image': image,
            'target': {'masks':[prompt,target]},
            'conversations': single_conversation,
            'map_placeholders': self.map_placeholders
        }

        return ret