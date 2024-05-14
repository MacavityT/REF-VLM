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
from xtuner.registry import DATASETS
from xtuner.utils.constants import (
    QUESTION_PLACEHOLDER,
    IMAGE_PLACEHOLDER,
    BOXES_PLACEHOLDER,
    OBJS_PLACEHOLDER,
    MASKS_PLACEHOLDER)
from .mixin import MInstrDataset
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils


#osprey_724k
DETAILED_QUESTIONS =  [
    'Can you provide me with a detailed description of the region in the picture marked by <region>?',
    "I'm curious about the region represented by <region> in the picture. Could you describe it in detail?",
    'What can you tell me about the region indicated by <region> in the image?',
    "I'd like to know more about the area in the photo labeled <region>. Can you give me a detailed description?",
    'Could you describe the region shown as <region> in the picture in great detail?',
    'What details can you give me about the region outlined by <region> in the photo?',
    'Please provide me with a comprehensive description of the region marked with <region> in the image.',
    'Can you give me a detailed account of the region labeled as <region> in the picture?',
    "I'm interested in learning more about the region represented by <region> in the photo. Can you describe it in detail?",
    'What is the region outlined by <region> in the picture like? Could you give me a detailed description?',
    'Can you provide me with a detailed description of the region in the picture marked by <region>, please?',
    "I'm curious about the region represented by <region> in the picture. Could you describe it in detail, please?",
    'What can you tell me about the region indicated by <region> in the image, exactly?',
    "I'd like to know more about the area in the photo labeled <region>, please. Can you give me a detailed description?",
    'Could you describe the region shown as <region> in the picture in great detail, please?',
    'What details can you give me about the region outlined by <region> in the photo, please?',
    'Please provide me with a comprehensive description of the region marked with <region> in the image, please.',
    'Can you give me a detailed account of the region labeled as <region> in the picture, please?',
    "I'm interested in learning more about the region represented by <region> in the photo. Can you describe it in detail, please?",
    'What is the region outlined by <region> in the picture like, please? Could you give me a detailed description?',
    'Please describe the region <region> in the image in detail.',
    'Can you offer a thorough analysis of the region <region> in the image?',
    'Could you elaborate on the region highlighted by <region> in the picture provided?',
    'Please share more information about the zone emphasized with <region> in the photo.',
    'What insights can you give ablout the area denoted by <region> in the image presented?',
    'Can you share a comprehensive rundown of the region denoted by <region> in the presented image?',
    "I'd like to know more about the region highlighted by <region> in the picture provided.",
    'Work through the important details of the area <region> in the image.',
    'Illustrate the area represtented by <region> through a descriptive explanation.',
    'Examine the region <region> closely and share its details.'
]

START_SENTENCE = [
    "This provides an overview of the picture.",
    "This gives a complete view of the scene.",
    "This offers a full depiction of the image.",
    "This presents a comprehensive overview of the photograph.",
    "This delivers an all-encompassing view of the picture.",
    "This depicts the entire scope of the image.",
    "This shows a broad perspective of the photograph.",
    "This outlines the total context of the image.",
    "This conveys a general impression of the scene.",
    "This illustrates the full extent of the picture.",
    "This reveals the overall composition of the image.",
    "This provides a panoramic view of the photograph.",
    "This gives a bird's-eye view of the picture.",
    "This presents the whole framework of the image.",
    "This sketches out the complete layout of the photograph.",
    "This offers a detailed representation of the picture.",
    "This displays the entire panorama of the image.",
    "This captures the full breadth of the scene.",
    "This shows the comprehensive layout of the photograph.",
    "This renders the complete perspective of the image.",
    "This exhibits the whole view of the picture.",
    "This portrays the total overview of the photograph.",
    "This provides a holistic view of the scene.",
    "This gives a full snapshot of the image.",
    "This presents an all-inclusive look at the picture.",
    "This conveys the entire vista of the photograph.",
    "This offers a summary view of the image.",
    "This outlines the broad contours of the picture.",
    "This displays the overall framework of the photograph.",
    "This shows the total landscape of the image.",
    "This illustrates the comprehensive scene of the picture.",
    "This provides a sweeping overview of the photograph.",
    "This gives a complete outline of the image.",
    "This presents the full scale of the scene.",
    "This reveals the entire layout of the picture.",
    "This portrays the wide-angle view of the photograph.",
    "This sketches the full range of the image.",
    "This depicts the broad spectrum of the picture.",
    "This exhibits the entire configuration of the photograph.",
]


#osprey_724k part
# base dataset
class ConversationDataset(MInstrDataset):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.begin_str = f"<image>\n{random.choice(START_SENTENCE)}\n"
        self.data_infos = self.load_annotations(self.text_path)

        
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
        return len(self.data_infos)
    
    def load_annotations(self, ann_file):
        data_infos = []
        ann_list = json.load(open(ann_file))

        for ann in ann_list:
            if len(ann['conversations'])//2 ==0:
                continue
            masks = []
            qa_s = []
            # filename = ann['file_name'].split('_')[-1]
            filename = ann['file_name']
            region_num = len(ann['annotation'])
            h, w = ann['height'], ann['width']
            str_region = ""
            for i in range(region_num):
                mask = ann['annotation'][i]['segmentation']
                masks.append(mask)
                if i > 0:
                    str_region += ','
                str_region += "region" + str(i+1) + MASKS_PLACEHOLDER

            for i in range(len(ann['conversations'])//2):
                    
                if i == 0:
                    if region_num == 1:
                        mid_str = "There are 1 part region in the picture: "+str_region+'. '
                    else:
                        mid_str = "There are {} part regions in the picture: ".format(str(region_num)) + str_region + '. '

                    question = ann['conversations'][i*2]['value']
                    question = question.replace('<','').replace('>','')
                    question = self.begin_str + mid_str + question
                    qa_s.append({'from': 'human', 'value': question + self.limit,'masks_seq':[[i] for i in range(region_num)]}) 
                else:
                    question = ann['conversations'][i*2]['value']
                    question = question.replace('<','').replace('>','')
                    qa_s.append({'from': 'human', 'value': question + self.limit})         

                
                answer = ann['conversations'][i*2+1]['value']
                answer = answer.replace('<','').replace('>','')
                qa_s.append({'from': 'gpt', 'value': answer})

            data_infos.append(dict(
                img_path = filename,
                masks = masks,
                height = h,
                width = w,
                qas = qa_s
            ))
        return data_infos

    def __getitem__(self, index):
        data_info = self.data_infos[index]
        img_path = data_info['img_path']
        height = data_info['height']
        width = data_info['width']
        masks_raw = data_info['masks']
        masks = []
        for mask_r in masks_raw:
            mask = self.annToMask(mask_r, height, width)
            masks.append(mask)
            
        masks = np.array(masks)
        qas = data_info['qas']
        
        # image = self.read_process_image(img_path)
        image = self.get_image(img_path)
        assert len(qas) % 2 == 0, "invalid quesion & answer pairs!"
        system = {
            'from': 'system',
            'value': [{'task':{'task_name':'vqa','element':['sentence'],'use_unit':False}} for _ in range(len(qas)//2)]
        }
        qas.insert(0,system)
        ret = {
            'image': image,
            'target': {'masks': masks},
            'conversations': qas
        }
        ret['map_placeholders'] = self.map_placeholders

        return ret
    
@DATASETS.register_module()
class OspreyPartLevel(ConversationDataset):
    def __init__(self, *args, **kwargs):
        self.limit = ' Answer the question using a single word or phrase.'
        super().__init__(*args, **kwargs)
        
@DATASETS.register_module()
class OspreyLVISPosNeg(ConversationDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, placeholders=(IMAGE_PLACEHOLDER, ), **kwargs) 

    def load_annotations(self, ann_file): #no limit
        data_infos = []
        ann_list = json.load(open(ann_file))

        for ann in ann_list:
            if len(ann['conversations']) // 2 ==0:
                continue
            masks = []
            qa_s = []
            filename = ann['file_name']
            # img_path = os.path.join(self.img_prefix, filename)
            region_num = len(ann['annotation'])
            h, w = ann['height'], ann['width']

            for i in range(region_num):
                mask = ann['annotation'][i]['segmentation']
                masks.append(mask)
        
            for i in range(len(ann['conversations']) // 2):
                    
                question = ann['conversations'][i*2]['value']
                match = re.search(r'<region(\d+)>',question)
                mask_seq = int(match.group(1)) - 1
                question = re.sub(r'<region\d+>', MASKS_PLACEHOLDER, question)

                if i==0:
                    question = self.begin_str + question
                qa_s.append({'from': 'human', 'value': question,'masks_seq':[[mask_seq]]})         
             
                answer = ann['conversations'][i*2+1]['value']
                qa_s.append({'from': 'gpt', 'value': answer})

            data_infos.append(dict(
                img_path = filename,
                masks = masks,
                height = h,
                width = w,
                qas = qa_s
            ))

        return data_infos

@DATASETS.register_module()
class OspreyConversations(ConversationDataset):
    def __init__(self, *args, **kwargs):
        self.limit = ""
        super().__init__(*args, **kwargs)
        
@DATASETS.register_module()
class OspreyShortForm(ConversationDataset):
     def __init__(self, *args, **kwargs):
        self.limit = ' Answer the question using a single word or phrase.'
        super().__init__(*args, **kwargs) 

@DATASETS.register_module()
class OspreyDetailedDescription(ConversationDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, placeholders=(IMAGE_PLACEHOLDER, ), **kwargs) 

    def load_annotations(self, ann_file):
        data_infos = []
        ann_list = json.load(open(ann_file))

        for ann in ann_list:
            masks = []
            qa_s = []
            filename = ann['file_name']
            region_num = len(ann['annotation'])
            h, w = ann['height'], ann['width']
            for i in range(region_num):
                mask = ann['annotation'][i]['segmentation']
                masks.append(mask)

                question = random.choice(DETAILED_QUESTIONS)
                question = question.replace('<region>', MASKS_PLACEHOLDER)
                if i==0:
                    qa_s.append({'from': 'human', 'value': self.begin_str + question,'masks_seq':[[i]]})         
                else:
                    qa_s.append({'from': 'human', 'value': question,'masks_seq':[[i]]})     
            
                answer = re.findall(r"<.*>:\ (.*)", ann['description'][i])[0]
           
                qa_s.append({'from': 'gpt', 'value': answer})

            data_infos.append(dict(
                img_path = filename,
                masks = masks,
                height = h,
                width = w,
                qas = qa_s
            ))
        return data_infos




