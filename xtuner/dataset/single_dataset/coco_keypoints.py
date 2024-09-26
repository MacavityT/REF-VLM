import os
import random
import copy
from PIL import Image
import numpy as np
import cv2
import json
from torch.utils.data import Dataset
from xtuner.registry import DATASETS
from pycocotools.mask import decode
from xtuner.dataset.utils import convert_bbox
import pycocotools.mask as mask_utils
from xtuner.dataset.utils import visualize_keypoints,visualize_box_single,visualize_mask_single
from xtuner.utils.constants import (
    KEYPOINTS_PLACEHOLDER,
    IMAGE_PLACEHOLDER,
    REGION_PLACEHOLDER,
    BOXES_PLACEHOLDER,
    MASKS_PLACEHOLDER,
    PHRASE_ST_PLACEHOLDER_STAGE2,
    PHRASE_ED_PLACEHOLDER_STAGE2,
    CLASS_PLACEHOLDER,
)
from collections import defaultdict
from .mixin import MInstrDataset




@DATASETS.register_module()
class COCOKeypointsDataset(MInstrDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset = self.read_json()
        self.createIndex()

    def read_json(self):
        with open(self.text_path) as f:
            img_json = json.loads(f.read())
        return img_json

    def createIndex(self):
        print('creating index...')
        self.anns, self.cats,self.imgs = {}, {}, {}
        # self.imgs = defaultdict(list)
        self.imgToAnns= defaultdict(list)
        if 'annotations' in self.dataset:
            for ann in self.dataset['annotations']:
                self.imgToAnns[ann['image_id']].append(ann)
                self.anns[ann['id']] = ann

        if 'images' in self.dataset:
            for img in self.dataset['images']:
                self.imgs[img['id']] = img

        if 'categories' in self.dataset:
            for cat in self.dataset['categories']:
                self.cats[cat['id']] = cat

        print('index created!')

    def __len__(self):
        if (self.offline_processed_text_folder is not None) and os.path.exists(self.offline_processed_text_folder):
            return len(self.text_data)
        else:
            return len(self.imgToAnns.keys())

    def __getitem__(self, index):
        anno_item = list(self.imgToAnns.items())[index]
        info = self.imgs[anno_item[0]]
        img_info = {
            'path': os.path.join(self.image_folder,info['file_name']),
            'width': info['width'],
            'height': info['height'],
        }
        id = info['id']
        annotations = anno_item[1]

        all_keypoints = []
        all_boxes = []
        keypoints_seq = []
        question = self.get_template()
        count = 0
        for annotation in annotations:
            num_keypoints = annotation['num_keypoints']
            if num_keypoints > 0:
                keypoints = np.array(annotation['keypoints']).reshape(-1,3)
                bbox = list(convert_bbox(annotation['bbox']))
                # start
                # image = np.array(Image.open(img_info['path']).convert('RGB'))
                # skeleton = self.cats[1]['skeleton']
                # visualize_keypoints(image=image,keypoints=keypoints,skeleton=skeleton,index=count)
                # end

                all_boxes.append(bbox)
                all_keypoints.append(keypoints)
                keypoints_seq.append(count)

                count += 1
        
        if all_keypoints != []:
            # answer = f'{PHRASE_ST_PLACEHOLDER_STAGE2}person{PHRASE_ED_PLACEHOLDER_STAGE2}{KEYPOINTS_PLACEHOLDER*len(all_keypoints)}'
            PLACE_HOLDER = BOXES_PLACEHOLDER + KEYPOINTS_PLACEHOLDER
            answer = f'{PHRASE_ST_PLACEHOLDER_STAGE2}person{PHRASE_ED_PLACEHOLDER_STAGE2}{PLACE_HOLDER*len(all_keypoints)}'
            
            ret = {
                'image':img_info,
                'target': {'boxes':all_boxes,'keypoints':all_keypoints},
                'conversations':[
                    # {'from':'system','value':[[{'task':{'task_name':'keypoint_detection','element':['phrase'],'use_unit':True},'unit':['keypoint']}]]},
                    {'from': 'system', 'value': [{'task':{'task_name':'detection','element':['phrase'],'use_unit':True},'unit':['box']}]},
                    {'from':'human','value':question},
                    {'from':'gpt','value':answer,'keypoints_seq':[keypoints_seq],'boxes_seq':[keypoints_seq]}
                ]
            }
        else:
            answer = 'In this image, we cannot find valid keypoints!'
            ret = {
                'image':img_info,
                'conversations':[
                    # {'from':'system','value':[[{'task':{'task_name':'keypoint_detection','element':['phrase'],'use_unit':True},'unit':['keypoint']}]]},
                    {'from':'system','value':[{'task':{'task_name':'vqa','element':['sentence'],'use_unit':False}}]},
                    {'from':'human','value':question},
                    {'from':'gpt','value':answer}
                ]
            }

        ret['map_placeholders'] = self.map_placeholders
        return ret



@DATASETS.register_module()
class COCOKeypointsRECDataset(MInstrDataset):
    def __init__(self, *args, version, **kwargs):
        super().__init__(*args, **kwargs)
        self.version = version
        assert self.version in ['box','mask']
        self.dataset = self.read_json()
        self.annotations = self.dataset['annotations']
        self.createIndex()

    def read_json(self):
        with open(self.text_path) as f:
            img_json = json.loads(f.read())
        return img_json

    def createIndex(self):
        print('creating index...')
        self.imgs = {}

        if 'images' in self.dataset:
            for img in self.dataset['images']:
                self.imgs[img['id']] = img

        print('index created!')
    
    def __len__(self):
        if (self.offline_processed_text_folder is not None) and os.path.exists(self.offline_processed_text_folder):
            return len(self.text_data)
        else:
            return len(self.dataset['annotations'])
    
    def __getitem__(self, index):
        annotation = self.annotations[index]
        img_id = annotation['image_id']
        info = self.imgs[img_id]
        img_info = {
            'path': os.path.join(self.image_folder,info['file_name']),
            'width': info['width'],
            'height': info['height'],
        }

        question = self.get_template()

        target = {}
        
        if self.version == 'box':
            question = question.replace(REGION_PLACEHOLDER,BOXES_PLACEHOLDER)
            bbox = [annotation['bbox']]
            box_seq = [[0]]
            conversation_human = {'from':'human','value':question,'boxes_seq':box_seq}
            target['boxes'] = bbox

        elif self.version == 'mask':
            question = question.replace(REGION_PLACEHOLDER,MASKS_PLACEHOLDER)
            rleObjs = mask_utils.frPyObjects(annotation["segmentation"], info["height"], info["width"])
            mask_decode = decode(rleObjs)
            if len(mask_decode.shape) == 3:
                mask = [decode(rleObjs)[:,:,0]]
            elif len(mask_decode.shape) == 2:
                mask = [decode(rleObjs)]
            else:
                raise NotImplementedError
            mask_seq = [[0]]
            conversation_human = {'from':'human','value':question,'masks_seq':mask_seq}
            target['masks'] = mask

        num_keypoints = annotation['num_keypoints']

        if num_keypoints > 0:
            keypoints = [np.array(annotation['keypoints']).reshape(-1,3)]
            target['keypoints'] = keypoints
            answer = f'{PHRASE_ST_PLACEHOLDER_STAGE2}person{PHRASE_ED_PLACEHOLDER_STAGE2}{KEYPOINTS_PLACEHOLDER}'
            keypoints_seq = [[0]]
            conversation_gpt = {'from':'gpt','value':answer,'keypoints_seq':keypoints_seq}
        else:
            answer = 'No keypoints were detected in this box or mask. The figure in the image is either too small or too imcomplete to identify keypoints.'
            conversation_gpt = {'from':'gpt','value':answer}

        # start
        if num_keypoints > 0:
            image = np.array(Image.open(img_info['path']))
            skeleton = self.dataset['categories'][0]['skeleton']
            visualize_keypoints(image=image,keypoints=keypoints[0],skeleton=skeleton,index=index)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if 'boxes'  in target.keys():
                image_with_boxes = visualize_box_single(image,bbox[0])
                save_path = f'vis_box_{index}.jpg'
                cv2.imwrite(save_path, image_with_boxes)
            if 'masks' in target.keys():
                image_with_masks = visualize_mask_single(image,mask[0],alpha=1.0, beta=1.0)
                save_path_mask = f'vis_mask_{index}.jpg'
                cv2.imwrite(save_path_mask, image_with_masks)

                bbox = [annotation['bbox']]
                image_with_boxes = visualize_box_single(image,bbox[0])
                save_path_box = f'vis_box_{index}.jpg'
                cv2.imwrite(save_path_box, image_with_boxes)
        # end

        ret = {
            'image':img_info,
            'target':target,
            'conversations':[
                {'from':'system','value':[[{'task':{'task_name':'grounding_keypoints','element':['phrase'],'use_unit':True},'unit':['keypoint']}]]},
                conversation_human,
                conversation_gpt,
            ]
        }

        ret['map_placeholders'] = self.map_placeholders
        return ret