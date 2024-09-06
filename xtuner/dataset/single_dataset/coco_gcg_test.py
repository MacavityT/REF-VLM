import os
import re
import xml.etree.ElementTree as ET
from typing import Dict, List
from tqdm import tqdm
import json
from collections import defaultdict
from pycocotools.mask import decode
from torch.utils.data import Dataset
from xtuner.registry import DATASETS
from xtuner.utils.constants import (
    MASKS_PLACEHOLDER, 
    IMAGE_PLACEHOLDER,
    PHRASE_ST_PLACEHOLDER,
    PHRASE_ED_PLACEHOLDER,
    PHRASE_ST_PLACEHOLDER_STAGE2,
    PHRASE_ED_PLACEHOLDER_STAGE2
    )
from .mixin import MInstrDataset

def insert_phrases(input_str, indices, place_holder):
    # (0,"start",-1),(0,"end",[box_seq])
    phrase_map = {
        "start":PHRASE_ST_PLACEHOLDER_STAGE2,
        "end":PHRASE_ED_PLACEHOLDER_STAGE2,
    }
    new_seq = []
    for index, phrase, seq in sorted(indices, reverse=True):
        if phrase == 'end':
            output = phrase_map[phrase] + place_holder * len(seq)
            new_seq.append(seq)
        else:
            output = phrase_map[phrase]
        input_str = input_str[:index] + output + input_str[index:]
    
    new_seq = new_seq[::-1]
    return input_str, new_seq

def find_token_indices(caption, labels):
    all_indices = []
    # Create a dictionary to count the occurrences of each label
    occurrence_count = {label: 0 for label in labels}
    
    for i,label in enumerate(labels):
        if label == '':
            start_idx = 0
            end_idx = len(caption) - 1
            all_indices.append([start_idx,'start',-1])
            all_indices.append([end_idx,'end',[i]])
            continue
        # Update the occurrence count for the current label
        occurrence_count[label] += 1
        
        # Use regex to find the correct match based on the occurrence count
        matches = list(re.finditer(re.escape(label.lower()), caption.lower()))
        if matches == []:
            matches = list(re.finditer(re.escape(label[-1].lower()), caption.lower()))
        if len(matches) >= occurrence_count[label]:
            # Get the nth occurrence (1-based index) of the label
            match = matches[occurrence_count[label] - 1]
            start_idx = match.start()
            end_idx = match.end()
            all_indices.append([start_idx,'start',-1])
            all_indices.append([end_idx,'end',[i]])
        else:
            label_words = label.split()
            for label_word in label_words:
                matches = list(re.finditer(re.escape(label_word.lower()), caption.lower()))
                if matches != []:
                    match = matches[0]
                    start_idx = match.start()
                    end_idx = match.end()
                else:
                    continue
            if matches != []:
                all_indices.append([start_idx,'start',-1])
                all_indices.append([end_idx,'end',[i]])
            else:
                raise "labels not in the caption!"

    return all_indices

@DATASETS.register_module()
class COCOGCG(MInstrDataset):

    def __init__(self, *args, mask_path, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset = self.read_json(self.text_path)['annotations']
        self.mask_dataset = self.read_json(mask_path)
        self.createIndex()

    def read_json(self,path):
        with open(path) as f:
            img_json = json.loads(f.read())
        return img_json
    
    def createIndex(self):
        self.anns = defaultdict(list)
        if 'annotations' in self.mask_dataset:
            for ann in self.mask_dataset['annotations']:
                self.anns[ann['image_id']].append(ann)
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = self.dataset[index]
        caption = item['caption']
        labels = item['labels']
        image_id = item['image_id']
        masks = self.anns[image_id]
        masks = [decode(mask['segmentation']) for mask in masks]
        img_path = image_id + '.jpg'
        image = self.get_image(img_path)
        assert len(labels) == len(masks)
        if len(masks) == 0:
            question = 'Please describe the image in details.'
            value = [{'task':{'task_name':'vqa','element':['sentence'],'use_unit':False}}]
            ret = {
                'image': image,
                'conversations': [
                    {
                        'from': 'system',
                        'value':value,
                    },
                    {
                        'from': 'human',
                        'value': question,
                    },
                    {
                        'from': 'gpt',
                        'value': caption,
                    }
                ]
            }            

        else:
            # Use regex to replace the labels with the specified format
            all_indices = find_token_indices(caption,labels)
            caption, masks_seq = insert_phrases(caption,all_indices,MASKS_PLACEHOLDER)
            
            question = self.get_template()
            
            value = [{'task':{'task_name':'gcg_segmentation','element':['phrase','sentence'],'use_unit':True},'unit':['mask']}]
            ret = {
                'image': image,
                'target': {'masks': masks},  # 'seg' /
                'conversations': [
                    {
                        'from': 'system',
                        'value':value,
                    },
                    {
                        'from': 'human',
                        'value': question,
                    },
                    {
                        'from': 'gpt',
                        'value': caption,
                        'masks_seq': masks_seq,
                    }
                ]
            }

        ret['map_placeholders'] = self.map_placeholders

        return ret

