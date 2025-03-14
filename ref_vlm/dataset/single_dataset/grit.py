import os
import json
from typing import Dict, List
import numpy as np
import torch
import random
from torch.utils.data import Dataset
from pycocotools.mask import decode
from .mixin import MInstrDataset
from ..utils import de_norm_box_xyxy
import re
from ref_vlm.registry import DATASETS
from ref_vlm.utils.constants import (
    IMAGE_PLACEHOLDER,
    BOXES_PLACEHOLDER,
    OBJS_PLACEHOLDER,
    PHRASE_ST_PLACEHOLDER_STAGE2,
    PHRASE_ED_PLACEHOLDER_STAGE2,
    EXPR_PLACEHOLDER,
    CLASS_PLACEHOLDER
)


def flatten(element,all_concat=False):
    list_output = []
    for i,conversation in enumerate(element):
        if type(conversation) is list:
            for idx,j in enumerate(conversation):
                list_output.append(j)
                if all_concat:
                    if not (i == 0 and idx == 0):   # remove <image> for the rest of conversation
                        if j['from'] == 'human':
                            j['value'] = j['value'].replace(IMAGE_PLACEHOLDER,'')
        else:
            raise "Multi-conversations must be Lists !"
    return list_output


def insert_phrases(input_str, indices, place_holder):
    # (0,"start",-1),(0,"end",[box_seq])
    phrase_map = {
        "start":PHRASE_ST_PLACEHOLDER_STAGE2,
        "end":PHRASE_ED_PLACEHOLDER_STAGE2,
    }
    for index, phrase, seq in sorted(indices, reverse=True):
        if phrase == 'end':
            output = phrase_map[phrase] + place_holder * len(seq)
        else:
            output = phrase_map[phrase]
        input_str = input_str[:index] + output + input_str[index:]
    return input_str

@DATASETS.register_module()
class GRITDataset(MInstrDataset):
    def __init__(self, *args,version,max_conv_length=None,**kwargs):
        super().__init__(*args, **kwargs)
        self.version = version
        self.length = max_conv_length
        assert os.path.isdir(self.text_path), "GRIT dataset is composed of list of json files, not a single json!"
        self.text_path_file = os.listdir(self.text_path)

    def get_file_data(self, path):
        with open(path, 'r') as f:
            file_data = json.loads(f.read())
        return file_data
    
    def get_template_from_dict(self,template_name):
        assert isinstance(self.templates,Dict)
        import random
        template = self.templates[template_name]
        return random.choice(template)
    
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
    
    def check_conversations(self,conversations,placeholders=[BOXES_PLACEHOLDER]):
        SEQ_MAP = {
            BOXES_PLACEHOLDER: 'boxes_seq',
        }
        for conversation in conversations:
            for placeholder in placeholders:
                if SEQ_MAP[placeholder] in conversation.keys():
                    pattern = re.compile(placeholder)
                    all_find = pattern.findall(conversation['value'])
                    all_seq = flatten(conversation[SEQ_MAP[placeholder]])
                    assert len(all_find) == len(all_seq),\
                            f"placeholder {placeholder} not match. sentence: {conversation['value']}. num targets:{len(all_seq)}"
                    
    def get_caption(self,ret,caption):
        try:
            question = self.get_template()
        except:
            question = self.get_template_from_dict('image_cap')

        conversations =  [
                    {
                        'from': 'human',
                        'value': question,
                    },
                    {
                        'from': 'gpt',
                        'value': caption,
                    }
                ]
        ret['conversations'] = conversations
        ret['values'] = [{'task':{'task_name':'vqa','element':['sentence'],'use_unit':False}}]
        return ret

    def get_detection(self,ret,noun_chunks,caption):
        try:
            question = self.get_template()
        except:
            question = self.get_template_from_dict('DET')
        boxes = []
        cls_names = []
        box_seq = []
        caption_new = ''
        orig_caption = caption
        for i,noun_chunk in enumerate(noun_chunks):
            cls_name = orig_caption[int(noun_chunk[0]):int(noun_chunk[1])]
            box = noun_chunk[2:-1]
            assert len(box) == 4
            boxes.append(box)
            
            if cls_name in cls_names:  # some class may have two or more boxes
                previous_seq = cls_names.index(cls_name)
                box_seq[previous_seq].append(i)
                caption_new = caption_new.replace(f"{cls_name}{PHRASE_ED_PLACEHOLDER_STAGE2}",f"{cls_name}{PHRASE_ED_PLACEHOLDER_STAGE2}{BOXES_PLACEHOLDER}")
            else:
                seq = [i]
                cls_names.append(cls_name)
                box_seq.append(seq)
                caption_new = caption_new + PHRASE_ST_PLACEHOLDER_STAGE2 + cls_name + PHRASE_ED_PLACEHOLDER_STAGE2 + BOXES_PLACEHOLDER + ', '

        conversations = [
                    {
                        'from': 'human',
                        'value': question,
                    },
                    {
                        'from': 'gpt',
                        'value': caption_new,
                        'boxes_seq': box_seq
                    }
                ]
        ret['values'] = [{'task':{'task_name':'detection','element':['phrase'],'use_unit':True},'unit':['box']}]
        ret['conversations'] = conversations
        return ret
    
    def get_cond_detection(self,ret,noun_chunks,caption,random_select=False,length=None):
        boxes = []
        cls_names = []
        conversations = []
        
        for i,noun_chunk in enumerate(noun_chunks):
            
            cls_name = caption[int(noun_chunk[0]):int(noun_chunk[1])]
            box = noun_chunk[2:-1]
            assert len(box) == 4
            boxes.append(box)

            if cls_name in cls_names:
                previous_seq = cls_names.index(cls_name)
                conversations[previous_seq][1]['value'] += BOXES_PLACEHOLDER  # add one <boxes> in the conversation
                conversations[previous_seq][1]['boxes_seq'][0].append(i)

            else:
                try:
                    question = self.get_template()
                except:
                    question = self.get_template_from_dict('Cond_DET')
                question = question.replace(CLASS_PLACEHOLDER,cls_name)
                box_seq = [i]
                conversation_human = {'from': 'human','value': question}
                # value = PHRASE_ST_PLACEHOLDER_STAGE2 + 'target' + PHRASE_ED_PLACEHOLDER_STAGE2 + BOXES_PLACEHOLDER
                value = PHRASE_ST_PLACEHOLDER_STAGE2 + cls_name + PHRASE_ED_PLACEHOLDER_STAGE2 + BOXES_PLACEHOLDER
                conversation_gpt = {'from': 'gpt', 'value': value,'boxes_seq': [box_seq]}

                single_conversation = [conversation_human,conversation_gpt]
                cls_names.append(cls_name)
                conversations.append(single_conversation)

        if random_select:
            conversations = self.random_select(conversations,length)

        ret['values'] = [{'task':{'task_name':'grounding_detection','element':['phrase'],'use_unit':True},'unit':['box']} for _ in range(len(conversations))]
        conversations = flatten(conversations)
        ret['conversations'] = conversations
        return ret 

    def get_rec(self,ret,ref_exps,noun_chunks,caption,random_select=False,length=None):


        cls_boxes = []
        cls_box_dict = {}
        for i,noun_chunk in enumerate(noun_chunks):
            
            cls_name = caption[int(noun_chunk[0]):int(noun_chunk[1])]
            box = noun_chunk[2:-1]
            assert len(box) == 4
            cls_boxes.append(box)

            if cls_name in cls_box_dict.keys():
                cls_box_dict[cls_name].append(i)            
            else:
                seq = [i]
                cls_box_dict[cls_name] = seq


        expr_names = []
        conversations = []

        for j,ref_exp in enumerate(ref_exps):

            expr_name = caption[int(ref_exp[0]):int(ref_exp[1])]
            if expr_name in expr_names:
                previous_seq = expr_names.index(expr_name)
                conversations[previous_seq][1]['value'] += BOXES_PLACEHOLDER  # add one <boxes> in the conversation

            else:
                try:
                    question = self.get_template()
                except:
                    question = self.get_template_from_dict('REC')
                question = question.replace(EXPR_PLACEHOLDER,expr_name)
                conversation_human = {'from': 'human','value': question}
                # value = PHRASE_ST_PLACEHOLDER_STAGE2 + 'target' + PHRASE_ED_PLACEHOLDER_STAGE2 + BOXES_PLACEHOLDER
                value = PHRASE_ST_PLACEHOLDER_STAGE2 + expr_name + PHRASE_ED_PLACEHOLDER_STAGE2 + BOXES_PLACEHOLDER
                conversation_gpt = {'from': 'gpt', 'value': value}

                # find box_seq for expr in class_names
                for cls_name in cls_box_dict.keys():
                    if cls_name in expr_name:
                        conversation_gpt['boxes_seq'] = [cls_box_dict[cls_name]]
                        break

                expr_names.append(expr_name)
                single_conversation = [conversation_human,conversation_gpt]
                conversations.append(single_conversation)

        if random_select:
            conversations = self.random_select(conversations,length)
        ret['values'] = [{'task':{'task_name':'grounding_detection','element':['phrase'],'use_unit':True},'unit':['box']} for _ in range(len(conversations))]
        conversations = flatten(conversations)
        ret['conversations'] = conversations
        return ret
    
    def get_reg(self,ret,ref_exps,noun_chunks,caption,random_select=False,length=None):

        cls_boxes = []
        cls_box_dict = {}
        for i,noun_chunk in enumerate(noun_chunks):
            
            cls_name = caption[int(noun_chunk[0]):int(noun_chunk[1])]
            box = noun_chunk[2:-1]
            assert len(box) == 4
            cls_boxes.append(box)

            if cls_name in cls_box_dict.keys():
                cls_box_dict[cls_name].append(i)            
            else:
                seq = [i]
                cls_box_dict[cls_name] = seq

        conversations = []
        previous_ref_exp = []
        for j,ref_exp in enumerate(ref_exps):
            question = self.get_template_from_dict('REG')
            expr_name = caption[int(ref_exp[0]):int(ref_exp[1])]
            if expr_name in previous_ref_exp:
                continue

            question = question.replace(OBJS_PLACEHOLDER,BOXES_PLACEHOLDER)

            # find box_seq for expr in class_names
            for cls_name in cls_box_dict.keys():
                if cls_name in expr_name:
                    box_seq_exp = cls_box_dict[cls_name]
                    break

            if len(box_seq_exp) > 1:
                for id in box_seq_exp:
                    conversation_human = {'from': 'human','value': question,'boxes_seq':[[id]]}
                    conversation_gpt = {'from': 'gpt', 'value': expr_name}
                    conversations.append([conversation_human,conversation_gpt])

            else:
                conversation_human = {'from': 'human','value': question,'boxes_seq':[box_seq_exp]}
                conversation_gpt = {'from': 'gpt', 'value': expr_name}
                conversations.append([conversation_human,conversation_gpt])

            previous_ref_exp.append(expr_name)

        if random_select:
            conversations = self.random_select(conversations,length)
        ret['values'] = [{'task':{'task_name':'vqa','element':['sentence'],'use_unit':False}} for _ in range(len(conversations))]
        # ret['values'] = [{'task':{'task_name':'referring vqa','element':['sentence'],'use_unit':False}} for _ in range(len(conversations))]
        conversations = flatten(conversations)
        ret['conversations'] = conversations

        return ret

    
    def get_caption_detection(self,ret,noun_chunks,caption):
        try:
            question = self.get_template()
        except:
            question = self.get_template_from_dict('flickr30k')

        boxes = []
        cls_names = []
        box_seq = []
        orig_caption = caption
        cls_seq_map = {}
        for i,noun_chunk in enumerate(noun_chunks):
            cls_name = orig_caption[int(noun_chunk[0]):int(noun_chunk[1])]
            box = noun_chunk[2:-1]
            assert len(box) == 4
            boxes.append(box)
            if cls_name in cls_names:  # some class may have two or more boxes
                previous_seq = cls_names.index(cls_name)
                box_seq[previous_seq].append(i)
                cls_seq_map[cls_name]['seq'] = box_seq[previous_seq]
            else:
                seq = [i]
                box_seq.append(seq)
                cls_names.append(cls_name)
                cls_seq_map[cls_name] = {'token_positive':[int(noun_chunk[0]),int(noun_chunk[1])],'seq':seq}
        
        all_indices = []
        for cls_name in cls_seq_map.keys():
            item = cls_seq_map[cls_name]
            all_indices.append((item['token_positive'][0],"start",-1))
            all_indices.append((item['token_positive'][1],"end",item['seq']))
        
        caption = insert_phrases(caption,all_indices,BOXES_PLACEHOLDER)

        ret['values'] = [{'task':{'task_name':'gcg_detection','element':['phrase','sentence'],'use_unit':True},'unit':['box']}]
        ret['conversations'] = [
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
        return ret
        
    def make_conversations(self,name,ret,noun_chunks,ref_exps,caption,length):
        
        if name == 'image_cap':
            ret = self.get_caption(ret,caption)
        elif name == 'DET':
            ret = self.get_detection(ret,noun_chunks,caption)
        elif name == 'Cond_DET':
            ret = self.get_cond_detection(ret,noun_chunks,caption,random_select=True,length=length)
        elif name == 'REC':
            ret = self.get_rec(ret,ref_exps,noun_chunks,caption,random_select=True,length=length)
        elif name == 'REG':
            ret = self.get_reg(ret,ref_exps,noun_chunks,caption,random_select=True,length=length)
        elif name == 'flickr30k':
            ret = self.get_caption_detection(ret,noun_chunks,caption)

        return ret

    def __len__(self):
        if (self.offline_processed_text_folder is not None) and \
            os.path.exists(self.offline_processed_text_folder):
            return len(self.text_data)
        else:
            return len(self.text_path_file)
    
    def __getitem__(self, index):
        offline_item = super().__getitem__(index)
        if offline_item is not None:
            return offline_item
        text_file = self.text_path_file[index]
        annotations = self.get_file_data(os.path.join(self.text_path,text_file))
        img_path = annotations['key'] + '.jpg'
        image_path_abs = os.path.join(self.image_folder,img_path)
        noun_chunks = sorted(annotations['noun_chunks'])
        ref_exps = sorted(annotations['ref_exps'])
        caption = annotations['caption']

        ret = {}
        ret['image'] = {'path': image_path_abs,'width':annotations['width'],'height':annotations['height']}
        ret['map_placeholders'] = self.map_placeholders

        if self.version == 'c':  # caption task
            ret = self.get_caption(ret,caption)
            ret['conversations'].insert(0,{'from':'system','value':ret['values']})
            del ret['values']

        elif self.version == 'd':  # detection task
            boxes = []
            for i,noun_chunk in enumerate(noun_chunks):
                cls_name = annotations['caption'][int(noun_chunk[0]):int(noun_chunk[1])]
                box = list(de_norm_box_xyxy(noun_chunk[2:-1],w=ret['image']['width'],h=ret['image']['height']))
                assert len(box) == 4
                boxes.append(box)
            ret['target'] = {'boxes':boxes}
            ret = self.get_detection(ret,noun_chunks,caption)
            ret['conversations'].insert(0,{'from':'system','value':ret['values']})
            del ret['values']

        elif self.version == 'cond_d':  # conditional detection task
            '''multi-turn conversation'''
            boxes = []
            for i,noun_chunk in enumerate(noun_chunks):
                cls_name = annotations['caption'][int(noun_chunk[0]):int(noun_chunk[1])]
                box = list(de_norm_box_xyxy(noun_chunk[2:-1],w=ret['image']['width'],h=ret['image']['height']))
                assert len(box) == 4
                boxes.append(box)
            ret['target'] = {'boxes':boxes}
            ret = self.get_cond_detection(ret,noun_chunks,caption,random_select=True,length=self.length)
            for i,conversation in enumerate(ret['conversations']):
                if i != 0 and conversation['from'] == 'human':
                    conversation['value'] = conversation['value'].replace(IMAGE_PLACEHOLDER,'')
            ret['conversations'].insert(0,{'from':'system','value':ret['values']})
            del ret['values']

        elif self.version == 'r':  # referring expression task
            '''multi-turn conversation'''
            ref_exps = annotations['ref_exps']
            boxes = []
            expr_names = []
            conversations = []
            for i,ref_exp in enumerate(ref_exps):
                question = self.get_template()
                if i != 0:
                    question = question.replace(IMAGE_PLACEHOLDER,'')
                expr_name = annotations['caption'][int(ref_exp[0]):int(ref_exp[1])]
                box = list(de_norm_box_xyxy(ref_exp[2:-1],w=ret['image']['width'],h=ret['image']['height']))
                assert len(box) == 4
                boxes.append(box)

                if expr_name in expr_names:
                    previous_seq = expr_names.index(expr_name)
                    conversations[previous_seq*2+1]['value'] += BOXES_PLACEHOLDER  # add one <boxes> in the conversation
                    conversations[previous_seq*2+1]['boxes_seq'][0].append(i)

                else:
                    question = question.replace(EXPR_PLACEHOLDER,expr_name)
                    conversation_human = {'from': 'human','value': question}
                    value = PHRASE_ST_PLACEHOLDER_STAGE2 + 'target' + PHRASE_ED_PLACEHOLDER_STAGE2 + BOXES_PLACEHOLDER
                    boxes_seq = [[i]]
                    conversation_gpt = {'from': 'gpt', 'value': value,'boxes_seq':boxes_seq}

                    expr_names.append(expr_name)
                    conversations.append(conversation_human)
                    conversations.append(conversation_gpt)
            value = [{'task':{'task_name':'grounding_detection','element':['phrase'],'use_unit':True},'unit':['box']} for _ in range(len(conversations)//2)]

            ret['target'] = {'boxes':boxes}
            ret['conversations'] = conversations
            ret['conversations'].insert(0,{'from':'system','value': value})
        
        elif self.version == 'g':
            '''multi-turn conversation'''
            ref_exps = annotations['ref_exps']
            boxes = []
            expr_names = []
            conversations = []
            for i,ref_exp in enumerate(ref_exps):
                question = self.get_template()
                if i != 0:
                    question = question.replace(IMAGE_PLACEHOLDER,'')
                expr_name = annotations['caption'][int(ref_exp[0]):int(ref_exp[1])]
                box = ref_exp[2:-1]
                assert len(box) == 4
                box = list(de_norm_box_xyxy(ref_exp[2:-1],w=ret['image']['width'],h=ret['image']['height']))
                boxes.append(box)

                question = question.replace(OBJS_PLACEHOLDER,BOXES_PLACEHOLDER)
                conversation_human = {'from': 'human','value': question,'boxes_seq':[[i]]}
                conversation_gpt = {'from': 'gpt', 'value': expr_name}

                conversations.append(conversation_human)
                conversations.append(conversation_gpt)

            value = [{'task':{'task_name':'vqa','element':['sentence'],'use_unit':False}} for _ in range(len(conversations)//2)]
            # value = [{'task':{'task_name':'referring vqa','element':['sentence'],'use_unit':False}} for _ in range(len(conversations)//2)]
            ret['target'] = {'boxes':boxes}
            ret['conversations'] = conversations
            ret['conversations'].insert(0,{'from':'system','value': value})

                

        elif self.version == 'c_d': # caption + detection task
            boxes = []
            for i,noun_chunk in enumerate(noun_chunks):
                cls_name = annotations['caption'][int(noun_chunk[0]):int(noun_chunk[1])]
                box = list(de_norm_box_xyxy(noun_chunk[2:-1],w=ret['image']['width'],h=ret['image']['height']))
                assert len(box) == 4
                boxes.append(box)
            ret['target'] = {'boxes':boxes}
            ret = self.get_caption_detection(ret,noun_chunks,caption)
            ret['conversations'].insert(0,{'from':'system','value':ret['values']})
            del ret['values']


        elif self.version == 'combine':   # combine all previous tasks
            """multi-turn conversations"""
            assert isinstance(self.template_name,List)

            boxes = []
            for i,noun_chunk in enumerate(noun_chunks):
                cls_name = annotations['caption'][int(noun_chunk[0]):int(noun_chunk[1])]
                box = list(de_norm_box_xyxy(noun_chunk[2:-1],w=ret['image']['width'],h=ret['image']['height']))
                assert len(box) == 4
                boxes.append(box)
            ret['target'] = {'boxes':boxes}
            
            all_conversations = []
            all_system_values = []
            for template_name in self.template_name:
                ret_for_single_task = self.make_conversations(template_name,ret,noun_chunks,ref_exps,caption,length=self.length)
                conversations = ret_for_single_task['conversations']
                all_conversations.append(conversations)
                all_system_values.append(ret_for_single_task['values'])

            # self.check_conversations(flatten(all_conversations))
            all_conversations,all_system_values = self.random_select(all_conversations,system_value=all_system_values)
            
            ret['conversations'] = [{'from':'system','value':flatten(all_system_values)}]
            ret['conversations'] += flatten(all_conversations,all_concat=True)

            if 'values' in ret.keys():
                 del ret['values']

        return ret


@DATASETS.register_module()
class GRITOfflineDataset(MInstrDataset):
    def __init__(self, *args, map_placeholders, **kwargs):
        super().__init__(*args, **kwargs)
        assert os.path.isfile(self.text_path), "GRIT post process dataset is a single json file!"
        self.map_placeholders = map_placeholders
    
    def __getitem__(self, index):
        offline_item = super().__getitem__(index)
        if offline_item is not None:
            return offline_item
        
        item = self.get_raw_item(index)
        
        return item