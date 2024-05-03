import os
import json
from typing import Dict
import numpy as np
import torch
from torch.utils.data import Dataset
from pycocotools.mask import decode
from .mixin import MInstrDataset

from xtuner.registry import DATASETS
from xtuner.utils.constants import (
    IMAGE_PLACEHOLDER,
    BOXES_PLACEHOLDER,
    PHRASE_ST_PLACEHOLDER_STAGE2,
    PHRASE_ED_PLACEHOLDER_STAGE2,
    EXPR_PLACEHOLDER,
    CLASS_PLACEHOLDER
)

class GritDataset(Dataset):

    def __init__(self, folder, root):
        super().__init__()
        assert folder is not None
        self.folder = folder
        self.frit = os.listdir(folder)
        self.img = os.listdir(root)
        
    def read_json(self, path):
        with open(path, 'r') as f:
            datas = json.loads(f.read())
        return datas

    def __getitem__(self, index):
        index = str(index).zfill(9)
        path_ads = os.path.join(self.folder, f"{index}.json")
        annotations = self.read_json(path_ads)
        id = index+'.jpg'
        noun_chunks = annotations['noun_chunks']
        caption =  annotations['caption']
        ref_exps = annotations["ref_exps"]
        clip_similarity_vitb32 = annotations["clip_similarity_vitb32"]
        clip_similarity_vitl14 = annotations["clip_similarity_vitl14"]
        width = annotations["width"]
        height = annotations["height"]
        original_width = annotations["original_width"]
        original_height = annotations["original_height"]
        item = {
            "img_id": id,
            "caption":caption,
            "noun_chunks":noun_chunks,
            "ref_exps":ref_exps,
            "clip_similarity_vitb32":clip_similarity_vitb32,
            "clip_similarity_vitl14":clip_similarity_vitl14,
            "width":width,
            "height":height,
            "original_width":original_width,
            "original_height":original_height
        }

        return item

    def __len__(self):
        return len(self.grit)

@DATASETS.register_module()
class GRITDataset(MInstrDataset):
    def __init__(self, *args,version, **kwargs):
        super().__init__(*args, **kwargs)
        self.version = version
        assert os.path.isdir(self.text_path), "GRIT dataset is composed of list of json files, not a single json!"
        self.text_path_file = os.listdir(self.text_path)

    def get_file_data(self, path):
        with open(path, 'r') as f:
            file_data = json.loads(f.read())
        return file_data
    
    def get_template_from_dict(self,template_name):
        assert self.templates is isinstance(self.templates,Dict)
        import random
        template = self.templates[template_name]
        return random.choice(template)

    
    def __getitem__(self, index):
        text_file = self.text_path_file[index]
        annotations = self.get_file_data(os.path.join(self.text_path,text_file))
        img_path = annotations['key'] + '.jpg'
        image = self.get_image(img_path)
        caption = annotations['caption']

        if self.version == 'c':  # caption task
            # question = self.get_template('caption')
            question = self.get_template()
            ret = {
                'image': image,
                'conversations': [
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
        
        elif self.version == 'd':  # detection task
            # question = self.get_template('DET')
            question = self.get_template()
            noun_chunks = annotations['noun_chunks']
            boxes = []
            cls_names = []
            box_seq = []
            for i,noun_chunk in enumerate(noun_chunks):
                cls_name = annotations['caption'][int(noun_chunk[0]):int(noun_chunk[1])]
                box = noun_chunk[2:-1]
                assert len(box) == 4
                boxes.append(box)
                
                if cls_name in cls_names:  # some class may have two or more boxes
                    previous_seq = cls_names.index(cls_name)
                    box_seq[previous_seq].append(i)
                else:
                    seq = [i]
                    cls_names.append(cls_name)
                    box_seq.append(seq)
            

            ret = {
                'image': image,
                'target': {'boxes':boxes,'class_names':cls_names},
                'conversations': [
                    {
                        'from': 'human',
                        'value': question,
                    },
                    {
                        'from': 'gpt',
                        'value': BOXES_PLACEHOLDER * len(boxes),
                        'box_seq': box_seq
                    }
                ]
            }

        elif self.version == 'cond_d':  # conditional detection task
            '''multi-turn conversation'''
            noun_chunks = annotations['noun_chunks']
            boxes = []
            cls_names = []
            conversations = []
            
            for i,noun_chunk in enumerate(noun_chunks):
                # question = self.get_template('Cond_DET')
                question = self.get_template()
                cls_name = annotations['caption'][int(noun_chunk[0]):int(noun_chunk[1])]
                box = noun_chunk[2:-1]
                assert len(box) == 4
                boxes.append(box)

                if cls_name in cls_names:
                    previous_seq = cls_names.index(cls_name)
                    conversations[previous_seq*2+1]['value'] += BOXES_PLACEHOLDER  # add one <boxes> in the conversation
                    conversations[previous_seq*2+1]['box_seq'][0].append(i)

                else:
                    question = question.replace(CLASS_PLACEHOLDER,cls_name)
                    box_seq = [i]
                    conversation_human = {'from': 'human','value': question}
                    conversation_gpt = {'from': 'gpt', 'value': BOXES_PLACEHOLDER,'box_seq': [box_seq]}

                    cls_names.append(cls_name)
                    conversations.append(conversation_human)
                    conversations.append(conversation_gpt)
            ret = {
                'image': image,
                'target': {'boxes':boxes,'class_names':cls_names},
                'conversations': conversations
            }

        elif self.version == 'r':  # referring expression task
            '''multi-turn conversation'''
            ref_exps = annotations['ref_exps']
            boxes = []
            expr_names = []
            conversations = []
            for i,ref_exp in enumerate(ref_exps):
                # question = self.get_template('Cond_DET')
                question = self.get_template()
                expr_name = annotations['caption'][int(ref_exp[0]):int(ref_exp[1])]
                box = ref_exp[2:-1]
                assert len(box) == 4
                boxes.append(box)

                if expr_name in expr_names:
                    previous_seq = expr_names.index(expr_name)
                    conversations[previous_seq*2+1]['value'] += BOXES_PLACEHOLDER  # add one <boxes> in the conversation
                    conversations[previous_seq*2+1]['box_seq'][0].append(i)

                else:
                    question = question.replace(EXPR_PLACEHOLDER,expr_name)
                    box_seq = [i]
                    conversation_human = {'from': 'human','value': question}
                    conversation_gpt = {'from': 'gpt', 'value': BOXES_PLACEHOLDER,'box_seq': [box_seq]}

                    expr_names.append(expr_name)
                    conversations.append(conversation_human)
                    conversations.append(conversation_gpt)

            ret = {
                'image': image,
                'target': {'boxes':boxes,'expr_names':expr_names},
                'conversations': conversations
            }

        elif self.version == 'c_d': # caption + detection task
            # question = self.get_template('flickr30k')
            question = self.get_template()
            noun_chunks = annotations['noun_chunks']
            caption = annotations['caption']
            boxes = []
            cls_names = []
            box_seq = []
            for i,noun_chunk in enumerate(noun_chunks):
                cls_name = annotations['caption'][int(noun_chunk[0]):int(noun_chunk[1])]
                box = noun_chunk[2:-1]
                assert len(box) == 4
                boxes.append(box)
                if cls_name in cls_names:  # some class may have two or more boxes
                    previous_seq = cls_names.index(cls_name)
                    box_seq[previous_seq].append(i)
                else:
                    caption = caption.replace(cls_name,PHRASE_ST_PLACEHOLDER_STAGE2 + cls_name + PHRASE_ED_PLACEHOLDER_STAGE2 + BOXES_PLACEHOLDER)
                    seq = [i]
                    box_seq.append(seq)
                    cls_names.append(cls_name)
            
            ret = {
                'image': image,
                'target': {'boxes': boxes},  # 'seg' /
                'conversations': [
                    {
                        'from': 'human',
                        'value': question,
                    },
                    {
                        'from': 'gpt',
                        'value': caption,
                        'boxes_seq': box_seq,
                    }
                ]
            }

        return ret









if __name__ == '__main__':

    annotations = '/data/Aaronzhu/DatasetStage2and3/GRIT/annotations'
    img = '/data/Aaronzhu/DatasetStage2and3/GRIT/img'

    Grit = GritDataset(folder = annotations,root = img)
    grit = Grit[0]

    print(grit)