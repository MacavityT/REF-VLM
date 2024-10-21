import json
import json
import os
import random
import numpy as np
from collections import namedtuple
from typing import Dict, List
from PIL import Image
from vt_plug.registry import DATASETS
from vt_plug.utils.constants import (
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
from vt_plug.registry import DATASETS


"""
Reference: https://github.com/facebookresearch/detectron2/blob/main/detectron2/data/datasets/cityscapes_panoptic.py
           https://github.com/CircleRadon/Osprey/blob/c61d6df3ebc259841add1a542a01a7a88f84fdb4/osprey/eval/eval_open_vocab_seg_detectron2.py
           https://github.com/VainF/DeepLabV3Plus-Pytorch/blob/4e1087de98bc49d55b9239ae92810ef7368660db/datasets/cityscapes.py#L12
           https://github.com/Xujxyang/OpenTrans/blob/aea33ff3682403aec6d55a670030d6cb4b0623cd/mask2former/data/datasets/openseg_classes.py

"""

def resize_mask(mask,width,height,ratio=0.3):
    if mask is None:
        return None
    mask = Image.fromarray(mask)
    mask = mask.resize((int(width*ratio),int(height*ratio)), Image.LANCZOS)
    mask = np.array(mask)
    mask[mask!=0] = 1
    return mask.astype(np.uint8)


@DATASETS.register_module()
class Cityscapes(MInstrDataset):

    CityscapesClass = namedtuple('CityscapesClass', ['name', 'id', 'train_id', 'category', 'category_id',
                                                     'has_instances', 'ignore_in_eval', 'color'])
    classes = [
        CityscapesClass('unlabeled',            0, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('ego vehicle',          1, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('rectification border', 2, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('out of roi',           3, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('static',               4, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('dynamic',              5, 255, 'void', 0, False, True, (111, 74, 0)),
        CityscapesClass('ground',               6, 255, 'void', 0, False, True, (81, 0, 81)),
        CityscapesClass('road',                 7, 0, 'flat', 1, False, False, (128, 64, 128)),
        CityscapesClass('sidewalk',             8, 1, 'flat', 1, False, False, (244, 35, 232)),
        CityscapesClass('parking',              9, 255, 'flat', 1, False, True, (250, 170, 160)),
        CityscapesClass('rail track',           10, 255, 'flat', 1, False, True, (230, 150, 140)),
        CityscapesClass('building',             11, 2, 'construction', 2, False, False, (70, 70, 70)),
        CityscapesClass('wall',                 12, 3, 'construction', 2, False, False, (102, 102, 156)),
        CityscapesClass('fence',                13, 4, 'construction', 2, False, False, (190, 153, 153)),
        CityscapesClass('guard rail',           14, 255, 'construction', 2, False, True, (180, 165, 180)),
        CityscapesClass('bridge',               15, 255, 'construction', 2, False, True, (150, 100, 100)),
        CityscapesClass('tunnel',               16, 255, 'construction', 2, False, True, (150, 120, 90)),
        CityscapesClass('pole',                 17, 5, 'object', 3, False, False, (153, 153, 153)),
        CityscapesClass('polegroup',            18, 255, 'object', 3, False, True, (153, 153, 153)),
        CityscapesClass('traffic light',        19, 6, 'object', 3, False, False, (250, 170, 30)),
        CityscapesClass('traffic sign',         20, 7, 'object', 3, False, False, (220, 220, 0)),
        CityscapesClass('vegetation',           21, 8, 'nature', 4, False, False, (107, 142, 35)),
        CityscapesClass('terrain',              22, 9, 'nature', 4, False, False, (152, 251, 152)),
        CityscapesClass('sky',                  23, 10, 'sky', 5, False, False, (70, 130, 180)),
        CityscapesClass('person',               24, 11, 'human', 6, True, False, (220, 20, 60)),
        CityscapesClass('rider',                25, 12, 'human', 6, True, False, (255, 0, 0)),
        CityscapesClass('car',                  26, 13, 'vehicle', 7, True, False, (0, 0, 142)),
        CityscapesClass('truck',                27, 14, 'vehicle', 7, True, False, (0, 0, 70)),
        CityscapesClass('bus',                  28, 15, 'vehicle', 7, True, False, (0, 60, 100)),
        CityscapesClass('caravan',              29, 255, 'vehicle', 7, True, True, (0, 0, 90)),
        CityscapesClass('trailer',              30, 255, 'vehicle', 7, True, True, (0, 0, 110)),
        CityscapesClass('train',                31, 16, 'vehicle', 7, True, False, (0, 80, 100)),
        CityscapesClass('motorcycle',           32, 17, 'vehicle', 7, True, False, (0, 0, 230)),
        CityscapesClass('bicycle',              33, 18, 'vehicle', 7, True, False, (119, 11, 32)),
        CityscapesClass('license plate',        -1, 255, 'vehicle', 7, False, True, (0, 0, 142)),
    ]


    # Convert ids to train_ids :from 34 categories to 19
    voidClass = 19
    id2trainid = np.array([label.train_id for label in classes if label.train_id >= 0], dtype='uint8')
    id2trainid[np.where(id2trainid==255)] = voidClass

    # Convert train_ids to colors
    mask_colors = [list(label.color) for label in classes if label.train_id >= 0 and label.train_id <= 19]
    mask_colors.append([0,0,0])
    mask_colors = np.array(mask_colors)


     # Convert train_ids to ids
    trainid2id = np.zeros((256), dtype='uint8')
    for label in classes:
        if label.train_id >= 0 and label.train_id < 255:
            trainid2id[label.train_id] = label.id

     # List of valid class ids
    validClasses = np.unique([label.train_id for label in classes if label.id >= 0])
    validClasses[np.where(validClasses==255)] = voidClass
    validClasses = list(validClasses)

    # Create list of totsl class names 19 + void
    classLabels = [label.name for label in classes if not (label.ignore_in_eval or label.id < 0)]
    classLabels.append('void')


    def __init__(self, *args, **kwargs):
        self.split = kwargs.pop('split',None)
        self.target_type = kwargs.pop('target_type',None) #  ("instance", "semantic", "polygon", "color", "depth")
        assert self.split is not None
        assert self.target_type is not None
        super().__init__(*args, **kwargs)
        self.mode = "gtFine" 
         
        self.targets_dir = self.text_path 
        self.images_dir = self.image_folder
        self.sem_question = "Can you segment <cls> in the image <image> and provide the masks for this class?"
        self.ins_question = "Can you segment this image<image> and display segmentation results?"
        self.data_infos = self.load_annotations(self.images_dir, self.targets_dir)
        
    def load_annotations(self, images_dir, targets_dir):
        
        data_infos = []

        for city in os.listdir(images_dir):
            img_dir = os.path.join(images_dir, city)
            target_dir = os.path.join(targets_dir, city)

            for file_name in os.listdir(img_dir):

                img_path = os.path.join(img_dir, file_name)
                target_name = '{}_{}'.format(file_name.split('_leftImg8bit')[0],
                                            self._get_target_suffix(self.mode, self.target_type))
                annotation = '{}_{}'.format(file_name.split('_leftImg8bit')[0],
                                            self._get_target_suffix(self.mode, "polygon"))
                target_path = os.path.join(target_dir, target_name)
                annotation_path = os.path.join(target_dir, annotation)
                annotation = json.load(open(annotation_path))
                imgHeight, imgWidth = annotation['imgHeight'], annotation['imgWidth']
                objects = annotation['objects']

                if self.target_type == "semantic":
                    # load masks and labels 
                    target = Image.open(target_path) 
                    target = self.id2trainid[target] # Convert class ids to train_ids 
                    unique_labels = np.unique(target)
                
                    gt_masks = []
                    gt_ids = []
                    gt_names = []

                    for i in range(len(unique_labels)):  
                        mask = (target == unique_labels[i]).astype(np.uint8)
                        gt_masks.append(mask)
                        gt_id = self.trainid2id[unique_labels[i]] 
                        gt_name = self.classLabels[unique_labels[i]]  #19 + void
                        gt_ids.append(gt_id)
                        gt_names.append(gt_name)
               
                    # one image with one mask
                    for i in range(len(gt_ids)): 
                        data_infos.append(dict(
                            img_path = img_path,
                            gt_mask = gt_masks[i],
                            gt_id = gt_ids[i], 
                            gt_name = gt_names[i], 
                            mask_seq = [i],
                            height = imgHeight,
                            width = imgWidth,
                        ))
                     

                elif self.target_type == "instance":
                    data_infos.append(dict(
                        img_path = img_path,
                        target_path = target_path,
                        objects = objects,
                        height = imgHeight,
                        width = imgWidth,
                    ))  
                     
                else:
                    raise NotImplementedError
                
        return data_infos


    def __len__(self):
        if (self.offline_processed_text_folder is not None) and \
            os.path.exists(self.offline_processed_text_folder):
            return len(self.text_data)
        else:
            return len(self.data_infos)
    

    def _get_target_suffix(self, mode, target_type):
        if target_type == 'instance':
            return '{}_instanceIds.png'.format(mode)
        elif target_type == 'semantic':
            return '{}_labelIds.png'.format(mode) 
        elif target_type == 'color':
            return '{}_color.png'.format(mode)
        elif target_type == 'polygon':
            return '{}_polygons.json'.format(mode)
        elif target_type == 'depth':
            return '{}_disparity.png'.format(mode)


    # def __getitem__(self, index):
    #     offline_item = super().__getitem__(index)
    #     if offline_item is not None:
    #         return offline_item
    #     data_info = self.data_infos[index]
    #     img_path = data_info['img_path']
    #     gt_mask = data_info['gt_mask']
    #     # gt_id = data_info['gt_id']
    #     gt_name = data_info['gt_name']
    #     mask_seq = data_info['mask_seq']
    #     height = data_info['height']
    #     width = data_info['width']

    #     # get conversation
    #     task = {'task_name': 'segmentation', 'element': ['phrase'], 'use_unit': True}
    #     unit = ['mask']
    #     system = {'from': 'system', 'value': [{'task': task, 'unit': unit}]}
    #     human = {'from': 'human', 'value': self.sem_question}
    #     value = PHRASE_ST_PLACEHOLDER_STAGE2 + gt_name + PHRASE_ED_PLACEHOLDER_STAGE2 +  MASKS_PLACEHOLDER + ', '
    #     answer = {'from': 'gpt', 'value': value, 'masks_seq': mask_seq}

    #     conversation = [system, human, answer]

    #     ret = {
    #         'image': {'path': img_path, 'width': width, 'height': height},
    #         'target':  {'mask': gt_mask},
    #         'conversations': conversation
    #     }
    #     ret['map_placeholders'] = self.map_placeholders
    #     return ret



@DATASETS.register_module()
class CityscapesSemantic(Cityscapes):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    
@DATASETS.register_module()
class CityscapesInstance(Cityscapes):
    def __init__(self, *args,**kwargs):
        self.ratio = kwargs.pop('ratio',1)
        super().__init__( *args, **kwargs)
        self.classes = ['person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']
        self.class_id = [24, 25, 26, 27, 28, 31, 32, 33]
        self.place_holder = MASKS_PLACEHOLDER
         

    def get_ins_ids(self, target):
        ids = [iid for iid in np.unique(target) if iid >= 1000]  # <1000: sem, >=1000: ins
        return ids

    def __getitem__(self, index):
        offline_item = super().__getitem__(index)
        if offline_item is not None:
            return offline_item
        data_info = self.data_infos[index]
        img_path = data_info['img_path']
        target_path = data_info['target_path']

        height = int(data_info['height'] * self.ratio)
        width = int(data_info['width'] * self.ratio)

        target = Image.open(target_path)

        target = np.array(target)
        target_idx = self.get_ins_ids(target)

        gt_masks = []
        gt_names = []
        answers = []
        unique_labels = np.unique(target_idx)
        
        class_label_counts = {}
        for i in range(len(unique_labels)):  
            class_id = unique_labels[i] // 1000

            if class_id in self.class_id: 
                class_index = self.class_id.index(class_id)
                class_label  = self.classes[class_index]
                mask = (target == unique_labels[i]).astype(np.uint8)
                gt_masks.append(resize_mask(mask,data_info['width'],data_info['height'],self.ratio))
                class_label_counts[class_label] = class_label_counts.get(class_label, 0) + 1 
                gt_names.append(class_label) 
   
        answers = {'from': 'gpt', 'value': '', 'masks_seq': []}

        mask_seq = {}
        start_index = 0
        for i, (class_label, count) in enumerate(class_label_counts.items()):
            indices = [str(i) for i in range(start_index, start_index + count)]
            mask_seq[class_label] = indices
            start_index += count
            answers['masks_seq'].extend([list(map(int, indices))])
            answers['value'] += f"{PHRASE_ST_PLACEHOLDER_STAGE2}{class_label}{PHRASE_ED_PLACEHOLDER_STAGE2}"
            if count > 0:
                answers['value'] += "<masks>" * count
            
            if i != len(class_label_counts.items()) - 1:
                answers['value'] += ', '
            else:
                answers['value'] += '.'

        # get conversation
        task = {'task_name': 'segmentation', 'element': ['phrase'], 'use_unit': True}
        unit = ['mask']
        system = {'from': 'system', 'value': [{'task': task, 'unit': unit}]}
        if self.split == 'train':
            question = self.get_template()
        elif self.split == 'val' or self.split == 'test':
            question = self.ins_question
            if gt_masks == []:
                gt_masks = [np.zeros((width,height))]
        else:
            raise NotImplementedError
        
        human = {'from': 'human', 'value': question}
        conversations = [system, human, answers]

        ret = {
            'image': {'path': img_path, 'width': width, 'height': height},
            'target': {'masks': gt_masks},
            'conversations': conversations
        }
        ret['map_placeholders'] = self.map_placeholders
        return ret



