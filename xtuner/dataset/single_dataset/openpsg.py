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




@DATASETS.register_module()
class OpenPSGDataset(MInstrDataset):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset = self.read_json(self.text_path)

    def read_json(self,path):
        with open(path) as f:
            img_json = json.loads(f.read())
        return img_json
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = self.dataset[index]
        file_name = item['file_name'].split('/')[1]
        caption = item['caption']
        grounding_dict = item['groundings']
        image = self.get_image(file_name)

        all_indices = []
        all_masks = []
        count = 0
        for i, phrase in enumerate(grounding_dict.keys()):
            grounding = grounding_dict[phrase]
            token_indices = grounding['token_positives']
            cur_seq = []
            for rle_mask in grounding['rle_masks']:
                decode_mask = decode(rle_mask)
                all_masks.append(decode_mask)
                cur_seq.append(count)
                count += 1

            all_indices.append([token_indices[0],'start',-1])
            all_indices.append([token_indices[1],'end',cur_seq])
        
        new_caption, new_seq = insert_phrases(caption,all_indices,MASKS_PLACEHOLDER)

        question = self.get_template()
            
        value = [{'task':{'task_name':'gcg_segmentation','element':['phrase','sentence'],'use_unit':True},'unit':['mask']}]
        ret = {
            'image': image,
            'target': {'masks': all_masks},  # 'seg' /
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
                    'value': new_caption,
                    'masks_seq': new_seq,
                }
            ]
        }

        ret['map_placeholders'] = self.map_placeholders

        return ret