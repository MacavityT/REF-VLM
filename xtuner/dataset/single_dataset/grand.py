import os
import json
from typing import Dict, List
import numpy as np
import torch
import random
from torch.utils.data import Dataset
from pycocotools.mask import decode
import time
from PIL import Image
import re
from .mixin import MInstrDataset
from xtuner.dataset.utils import norm_box_xyxy,de_norm_box_xyxy
from xtuner.registry import DATASETS
from xtuner.utils.constants import (
    IMAGE_PLACEHOLDER,
    OBJS_PLACEHOLDER,
    BOXES_PLACEHOLDER,
    MASKS_PLACEHOLDER,
    PHRASE_ST_PLACEHOLDER_STAGE2,
    PHRASE_ED_PLACEHOLDER_STAGE2,
    EXPR_PLACEHOLDER,
    CLASS_PLACEHOLDER
)

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

def sort_objects(objects,filter=True,filter_num=10):
    for object in objects:
        object['segmentation'] = decode(object['segmentation'])
        area = object['segmentation'].sum().item()
        object['area'] = area
    objects = sorted(objects, key=lambda x: x['area'],reverse=True)

    if filter:
        for i,item in enumerate(objects):
            if i > filter_num-1:
                item['segmentation'] = None
        objects = sorted(objects, key=lambda x: x['id'],reverse=False)

    return objects


def delete_objects(objects,filter_num):
    for object in objects:
        object['segmentation'] = decode(object['segmentation'])
        area = object['segmentation'].sum().item()
        object['area'] = area
    objects = sorted(objects, key=lambda x: x['area'],reverse=True)

    for i,item in enumerate(objects):
        if i > filter_num-1:
            del objects[i]
    objects = sorted(objects, key=lambda x: x['id'],reverse=False)

    return objects

def resize_mask(mask,width,height,ratio=0.3):
    if mask is None:
        return None
    mask = Image.fromarray(mask)
    mask = mask.resize((int(width*ratio),int(height*ratio)), Image.ANTIALIAS)
    mask = np.array(mask)
    mask[mask!=0] = 1
    return mask.astype(np.uint8)

def resize_box(box,width,height,ratio=0.3):
    box = norm_box_xyxy(box,width,height)
    box = de_norm_box_xyxy(box,int(width*ratio),int(height*ratio))
    return box


@DATASETS.register_module()
class GranDDataset(MInstrDataset):

    def __init__(self, *args, version,use_floating_objects=True,max_conv_length=None,ratio=0.3,**kwargs):
        super().__init__(*args, **kwargs)
        self.ratio = ratio
        self.version = version
        self.use_floating_objects = use_floating_objects
        self.length = max_conv_length
        assert os.path.isdir(self.text_path), "GRIT dataset is composed of list of json files, not a single json!"
        self.text_path_file = os.listdir(self.text_path)
        self.detailed_template = [
            'Please describe it in details.',
            'Please describe it thoroughly.',
            'Could you provide a detailed description?',
            'I would appreciate a comprehensive explanation.',
            'Can you elaborate on that?',
            'Please give me a detailed rundown.',
            'Could you break it down for me?',
            'I need a more elaborate explanation, please.',
            'Please provide more specifics.',
            'I am looking for a detailed account.',
            'Can you go into more depth about it?',
            'I require a more detailed description, if possible.',
            'Please expand on what you have mentioned.',
            'Could you offer a more detailed explanation?',
            'I would like to hear a more thorough description.',
            'Please delve deeper into the topic.',
            'I am seeking a more comprehensive understanding.',
            'Could you provide further details?',
            'Please elaborate further on the subject.',
            'I need a more detailed breakdown, please.', 
            'Can you provide additional information?', 
        ]

    def get_box_segm_by_id(self,search_id,objects,floating_objects):
        for obj in objects:
            if obj['id'] == search_id:
                return obj['bbox'], obj['segmentation']
        # Search in floating object if they have segmentation
        for obj in floating_objects:
            if obj['id'] == search_id:
                return obj['bbox'], obj['segmentation']
        return None
    
    def get_template_from_dict(self,template_name):
        assert isinstance(self.templates,Dict)
        template = self.templates[template_name]
        return random.choice(template)
    
    def get_file_data(self, path):
        with open(path, 'r') as f:
            file_data = json.loads(f.read())
        return file_data
    
    def dense_question(self,question):
        question += ' '
        question += random.choice(self.detailed_template)
        return question   

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
    
    def select_target(self,item):

        def remove_ones_and_next(lst):
            i = 0
            while i < len(lst):
                if lst[i] == 1:
                    del lst[i:i+2]
                else:
                    i += 1
            return lst

        target = item['target']
        if 'boxes' in target.keys():
            boxes = target['boxes']
        if 'masks' in target.keys():
            masks = target['masks']
        conversations = item['conversations']
        select_conversations = []

        selected_masks = []
        selected_boxes = []
        remove_idx = []
        human_flag = False
        new_conversation = None
        for i, conversation in enumerate(conversations):
            if i % 2 == 0:
                assert conversation['from'] == 'human'
                # {'from': 'human', 'value': xxx}
                human_flag = True
                new_conversation = {'from':'human','value':conversation['value']}
                if 'masks_seq' in conversation.keys():
                    seq_list = conversation['masks_seq']
                    assert len(seq_list[0]) == 1
                    seq = seq_list[0][0]
                    mask = masks[seq]
                    if mask is None:
                        remove_idx.append(i)
                        human_flag = False
                        continue
                    else:
                        selected_masks.append(mask)
                        new_masks_seq = [[len(selected_masks)-1]]
                        new_conversation['masks_seq'] = new_masks_seq
                
                elif 'boxes_seq' in conversation.keys():
                    seq_list = conversation['boxes_seq']
                    assert len(seq_list[0]) == 1
                    seq = seq_list[0][0]
                    box = boxes[seq]
                    selected_boxes.append(box)
                    new_boxes_seq = [[len(selected_boxes)-1]]
                    new_conversation['boxes_seq'] = new_boxes_seq

                assert new_conversation is not None
                select_conversations.append(new_conversation)
                
            else:
                assert conversation['from'] == 'gpt'
                new_conversation = {'from':'gpt','value':conversation['value']}
                # {'from': 'gpt', 'value': xxx}
                if 'masks_seq' in conversation.keys():
                    seq_list = conversation['masks_seq']
                    seq_list_selected = []
                    for seqs in seq_list:
                        seqs_selected = []
                        for seq in seqs:
                            mask = masks[seq]
                            selected_masks.append(mask)
                            selected_seq = len(selected_masks) - 1
                            seqs_selected.append(selected_seq)
                        seq_list_selected.append(seqs_selected)
                    
                    new_conversation['masks_seq'] = seq_list_selected

                elif 'boxes_seq' in conversation.keys():
                    seq_list = conversation['boxes_seq']
                    seq_list_selected = []
                    for seqs in seq_list:
                        seqs_selected = []
                        for seq in seqs:
                            box = boxes[seq]
                            selected_boxes.append(box)
                            selected_seq = len(selected_boxes) - 1
                            seqs_selected.append(selected_seq)
                        seq_list_selected.append(seqs_selected)
                    
                    new_conversation['boxes_seq'] = seq_list_selected
                if human_flag:
                    assert new_conversation is not None
                    select_conversations.append(new_conversation)
                else:
                    continue
        if len(conversations) > 1:
            assert conversations[0]['from'] == 'human'
            if conversations[0]['value'].count(IMAGE_PLACEHOLDER) == 0:
                conversations[0]['value'] = IMAGE_PLACEHOLDER + conversations[0]['value']
        if 'boxes' in target.keys():
            item['target']['boxes'] = selected_boxes
        if 'masks' in target.keys():
            item['target']['masks'] = selected_masks 
        item['conversations'] = select_conversations

        return remove_idx
    
    def check_conversations(self,conversations,placeholders):
        SEQ_MAP = {
            BOXES_PLACEHOLDER: 'boxes_seq',
            MASKS_PLACEHOLDER: 'masks_seq',
        }
        for conversation in conversations:
            for placeholder in placeholders:
                if SEQ_MAP[placeholder] in conversation.keys():
                    pattern = re.compile(placeholder)
                    all_find = pattern.findall(conversation['value'])
                    all_seq = self.concat_conversations(conversation[SEQ_MAP[placeholder]])
                    assert len(all_find) == len(all_seq),\
                            f"placeholder {placeholder} not match. sentence: {conversation['value']}. num targets:{len(all_seq)}"
    
    def caption(self,ret,captions,template_name=None,random_select=False,length=None):
        
        conversations = []
        for i,caption in enumerate(captions):
            caption_expr = caption['caption']

            if not isinstance(self.template_name,List):
                question = self.get_template()
            else:
                question = self.get_template_from_dict(template_name)
            if i != 0:
                question = question.replace(IMAGE_PLACEHOLDER,'')
            if caption['is_dense']:
                question += ' '
                question += random.choice(self.detailed_template)

            single_conversation = [{'from': 'human','value': question},{'from': 'gpt','value': caption_expr}]
            conversations.append(single_conversation)

        if random_select:
            conversations = self.random_select(conversations,length)
        all_conversations = []
        all_conversations.append({'from':'system','value':[{'task':{'task_name':'vqa',
                                                                    'element':['sentence'],
                                                                    'use_unit':False}} \
                                                            for _ in range(len(conversations))]})
        all_conversations.extend(self.concat_conversations(conversations))
        ret['conversations'] = all_conversations

        return ret
    
    def detection_segmentation(self,ret,objects,floating_objects,template_name=None):
        if not isinstance(self.template_name,List):
            question = self.get_template()
        else:
            assert template_name is not None
            question = self.get_template_from_dict(template_name)
        
        if self.use_floating_objects:
            objects = objects + floating_objects
        
        boxes_or_masks = []
        cls_names = []
        box_mask_seq = []
        caption_new = ''
        for i,object in enumerate(objects):
            if template_name == 'DET':
                boxes_or_masks.append(object['bbox'])
                type = 'boxes'
                task = {'task_name':'detection','element':['phrase'],'use_unit':True}
                unit = ['box']
                seq_name = 'boxes_seq'
                place_holder = BOXES_PLACEHOLDER
            elif template_name == 'SEG':
                boxes_or_masks.append(object['segmentation'])
                type = 'masks'
                task = {'task_name':'segmentation','element':['phrase'],'use_unit':True}
                unit = ['mask']
                seq_name = 'masks_seq'
                place_holder = MASKS_PLACEHOLDER
            else:
                raise "Please select valid template: DET or SEG!"

            cls_name = ', '.join(object['labels'])
            cls_name = cls_name.replace('_',' ')
            if cls_name in cls_names:  # some class may have two or more boxes
                previous_seq = cls_names.index(cls_name)
                box_mask_seq[previous_seq].append(i)
                caption_new = caption_new.replace(f"{cls_name}{PHRASE_ED_PLACEHOLDER_STAGE2}",f"{cls_name}{PHRASE_ED_PLACEHOLDER_STAGE2}{place_holder}")
            else:
                seq = [i]
                cls_names.append(cls_name)
                box_mask_seq.append(seq)
                caption_new = caption_new + PHRASE_ST_PLACEHOLDER_STAGE2 + cls_name + PHRASE_ED_PLACEHOLDER_STAGE2 + place_holder + ', '

        conversations = [
                    {
                        'from':'system',
                        'value':[{'task':task,'unit':unit}]
                    },
                    {
                        'from': 'human',
                        'value': question,
                    },
                    {
                        'from': 'gpt',
                        'value': caption_new,
                        seq_name: box_mask_seq
                    }
                ]
        
        ret['target'] = {type:boxes_or_masks}
        ret['conversations'] = conversations

        return ret

    def grounding_detection_segmentation(self,task,ret,objects,ratio,template_name=None,random_select=False,length=None):

        if task == 'detection':
            type = 'boxes'
            unit_task = {'task_name':'grounding_detection','element':['phrase'],'use_unit':True}
            unit= ['box']
            seq_name = 'boxes_seq'
            place_holder = BOXES_PLACEHOLDER
        elif task == 'segmentation':
            unit_task = {'task_name':'grounding_segmentation','element':['phrase'],'use_unit':True}
            unit= ['mask']
            type = 'masks'
            seq_name = 'masks_seq'
            place_holder = MASKS_PLACEHOLDER
        else:
            raise "Please select valid template: DET or SEG!"

        valid_objects = []
        for i,object in enumerate(objects):
            cls_name = object['attributes']
            if cls_name is None or cls_name == []:
                continue
            else:
                valid_objects.append(object)

        if random_select:
            if length > len(valid_objects):
                length = len(valid_objects)
            select_numbers = random.sample(range(len(valid_objects)),length)
        else:
            select_numbers = random.sample(range(len(valid_objects)),len(valid_objects))
        boxes_or_masks = []
        cls_names = []
        conversations = []
        for i,num in enumerate(select_numbers):
            select_object = valid_objects[num]
            cls_name = select_object['attributes'][0]
            if task == 'detection':
                # box = resize_box(select_object['bbox'],width=ret['image']['width'],
                #              height=ret['image']['height'],ratio=ratio)
                boxes_or_masks.append(select_object['bbox'])
            elif task == 'segmentation':
                mask = resize_mask(decode(select_object['segmentation']),width=ret['image']['width'],
                             height=ret['image']['height'],ratio=ratio)
                boxes_or_masks.append(mask)
            if cls_name in cls_names:
                previous_seq = cls_names.index(cls_name)
                conversations[previous_seq][1]['value'] += place_holder  # add one <boxes> in the conversation
                conversations[previous_seq][1][seq_name][0].append(i)

            else:
                if not isinstance(self.template_name,List):
                    question = self.get_template()
                else:
                    assert template_name is not None
                    question = self.get_template_from_dict(template_name)
                question = question.replace(CLASS_PLACEHOLDER,cls_name)
                box_mask_seq = [i]
                if i != 0:
                    question = question.replace(IMAGE_PLACEHOLDER,'')
                value = PHRASE_ST_PLACEHOLDER_STAGE2 + 'target' + PHRASE_ED_PLACEHOLDER_STAGE2 + place_holder
                conversation_human = {'from': 'human','value': question}
                conversation_gpt = {'from': 'gpt', 'value': value, seq_name: [box_mask_seq]}

                cls_names.append(cls_name)
                single_conversation = [conversation_human,conversation_gpt]

                conversations.append(single_conversation)

        # ret['image']['width'] = int(ret['image']['width']*ratio)
        # ret['image']['height'] = int(ret['image']['height']*ratio)    
        all_conversations = []
        if task == 'detection':
            all_conversations.append({'from':'system','value':[{'task':{'task_name':'grounding_detection',
                                                                        'element':['phrase'],'use_unit':True},'unit':['box']} 
                                                               for _ in range(len(conversations))]})
        elif task == 'segmentation':
            all_conversations.append({'from':'system','value':[{'task':{'task_name':'grounding_segmentation',
                                                                        'element':['phrase'],'use_unit':True},'unit':['mask']} 
                                                               for _ in range(len(conversations))]})
        all_conversations.extend(self.concat_conversations(conversations))
        ret['target'] = {type:boxes_or_masks}
        ret['conversations'] = all_conversations

        return ret

    def rec(self,task,ret,objects,floating_objects,captions,template_name=None,random_select=False,length=None):

        if self.use_floating_objects:
            objects = objects + floating_objects
        if task == 'detection':
            type = 'boxes'
            unit_task = {'task_name':'grounding_detection','element':['phrase'],'use_unit':True}
            unit= ['box']
            seq_name = 'boxes_seq'
            place_holder = BOXES_PLACEHOLDER
        elif task == 'segmentation':
            unit_task = {'task_name':'grounding_segmentation','element':['phrase'],'use_unit':True}
            unit= ['mask']
            type = 'masks'
            seq_name = 'masks_seq'
            place_holder = MASKS_PLACEHOLDER
        else:
            raise "Please select valid template: REC or RES!"
        
        boxes_or_masks = []
        for i,object in enumerate(objects):
            if task == 'detection':
                boxes_or_masks.append(object['bbox'])
            elif task == 'segmentation':
                boxes_or_masks.append(decode(object['segmentation']))
            else:
                raise "Please select valid template: REC or RES!"
        
        conversations = []
        for j,caption in enumerate(captions):
            for detail in caption['details']:
                phrase = detail['phrase']
                if 'ids' in detail.keys():
                    seq = detail['ids']
                    value = PHRASE_ST_PLACEHOLDER_STAGE2 + 'target' + PHRASE_ED_PLACEHOLDER_STAGE2 + place_holder * len(seq)
                    conversation_gpt = {'from': 'gpt', 'value': value, seq_name: [seq]}
                else:
                    seq = [detail['id']]
                    value = PHRASE_ST_PLACEHOLDER_STAGE2 + 'target' + PHRASE_ED_PLACEHOLDER_STAGE2 + place_holder
                    conversation_gpt = {'from': 'gpt', 'value': value, seq_name: [seq]}

                if not isinstance(self.template_name,List):
                    question = self.get_template()
                else:
                    assert template_name is not None
                    question = self.get_template_from_dict(template_name)

                question = question.replace(EXPR_PLACEHOLDER,phrase)
                if j != 0:
                    question = question.replace(IMAGE_PLACEHOLDER,'')
                conversation_human = {'from': 'human','value': question}

                single_conversation = [conversation_human,conversation_gpt]
                conversations.append(single_conversation)

        if random_select:
            conversations = self.random_select(conversations,length)

        all_conversations = []
        all_conversations.append({'from':'system','value':[{'task':unit_task,'unit':unit} for _ in range(len(conversations))]})
        all_conversations.extend(self.concat_conversations(conversations))
        ret['target'] = {type:boxes_or_masks}
        ret['conversations'] = all_conversations

        return ret
    

    def reg(self,task,ret,objects,ratio,template_name=None,length=None):
        objects = delete_objects(objects,filter_num=10)
        if task == 'detection':
            unit_task = {'task_name':'vqa','element':['sentence'],'use_unit':False}
            # unit_task = {'task_name':'referring vqa','element':['sentence'],'use_unit':False}
            type = 'boxes'
            seq_name = 'boxes_seq'
            place_holder = BOXES_PLACEHOLDER
        elif task == 'segmentation':
            unit_task = {'task_name':'vqa','element':['sentence'],'use_unit':False}
            # unit_task = {'task_name':'referring vqa','element':['sentence'],'use_unit':False}
            type = 'masks'
            seq_name = 'masks_seq'
            place_holder = MASKS_PLACEHOLDER
        else:
            raise "Please select valid template: REG or REG_SEG!"
        if length == None:
            length = 6
        if len(objects) > length:
            objects = random.sample(objects,length)

        boxes_or_masks = []
        all_conversations = []
        for i,object in enumerate(objects):
            attributes = object['attributes']
            if attributes is None or attributes == []:
                continue

            if task == 'detection':
                boxes_or_masks.append(object['bbox'])
            elif task == 'segmentation':
                mask = resize_mask(object['segmentation'],width=ret['image']['width'],
                             height=ret['image']['height'],ratio=ratio)
                boxes_or_masks.append(mask)
            else:
                raise "Please select valid template: REC or RES!"            

            # construct conversations
            if not isinstance(self.template_name,List):
                question = self.get_template()
            else:
                assert template_name is not None
                question = self.get_template_from_dict(template_name)
            if task == 'detection':
                question = question.replace(OBJS_PLACEHOLDER,BOXES_PLACEHOLDER)
            elif task == 'segmentation':
                question = question
            if i != 0:
                question = question.replace(IMAGE_PLACEHOLDER,'')
            seq_id = len(boxes_or_masks) - 1
            all_conversations.append({'from': 'human','value': question,seq_name:[[seq_id]]})
            all_conversations.append({'from': 'gpt', 'value': attributes[0]})        

        system = {'from':'system','value':[{'task':unit_task} for _ in range(len(all_conversations)//2)]}
        all_conversations.insert(0,system)

        ret['target'] = {type:boxes_or_masks}
        ret['conversations'] = all_conversations


        return ret
    
    def caption_detection_segmentation(self,task,ret,objects,floating_objects,captions,ratio,
                                       template_name=None,random_select=False,length=None):
        if self.use_floating_objects:
            objects = objects + floating_objects
        if task == 'detection':
            unit_task = {'task_name':'gcg_detection','element':['phrase','sentence'],'use_unit':True}
            unit= ['box']
            type = 'boxes'
            seq_name = 'boxes_seq'
            place_holder = BOXES_PLACEHOLDER
        elif task == 'segmentation':
            unit_task = {'task_name':'gcg_segmentation','element':['phrase','sentence'],'use_unit':True}
            unit= ['mask']
            type = 'masks'
            seq_name = 'masks_seq'
            place_holder = MASKS_PLACEHOLDER
        else:
            raise "Please select valid template: flickr30k or flickr30k_SEG!"
        
        object_dict_map = {}
        id_map = {}
        for i,object in enumerate(objects):
            id_map[f"id_{object['id']}"] = i
            object_dict_map[i] = object

            
        conversations = []
        boxes_or_masks = []
        count = 0
        for j,caption in enumerate(captions):
            # generate caption + detection conversation
            caption_expr = caption['caption']
            if not isinstance(self.template_name,List):
                question_gcg = self.get_template()
            else:
                assert template_name is not None
                question_gcg = self.get_template_from_dict(template_name)            

            seq_cap_det_seg = []
            if caption['is_dense']:
                question_gcg = self.dense_question(question_gcg)

            if j != 0:
                question_gcg = question_gcg.replace(IMAGE_PLACEHOLDER,'')

            all_indices = []
            for detail in caption['details']:
                phrase = detail['phrase']
                token_positive = detail['tokens_positive']
                if token_positive is None:
                    continue
                if 'ids' in detail.keys():
                    reform_ids = []
                    for id in detail['ids']:
                        reform_ids.append(id_map[f"id_{id}"])
                    detail['ids'] = reform_ids
                    reg_seq = detail['ids']
                    new_seq = []
                    for new_id in reg_seq:
                        current_obj = object_dict_map[new_id]
                        if task == 'detection':
                            boxes_or_masks.append(current_obj['bbox'])
                        elif task == 'segmentation':
                            boxes_or_masks.append(decode(current_obj['segmentation']))
                        else:
                            raise "Please select valid template: flickr30k or flickr30k_SEG!"
                        new_seq.append(count)
                        count += 1

                else:
                    detail['id'] = id_map[f"id_{detail['id']}"]
                    reg_seq = detail['id']
                    current_obj = object_dict_map[reg_seq]
                    if task == 'detection':
                        boxes_or_masks.append(current_obj['bbox'])
                    elif task == 'segmentation':
                        mask = resize_mask(decode(current_obj['segmentation']),width=ret['image']['width'],
                             height=ret['image']['height'],ratio=ratio)
                        boxes_or_masks.append(mask)
                    else:
                        raise "Please select valid template: flickr30k or flickr30k_SEG!"
                    new_seq = count
                    count += 1

                # generate caption + detection conversation
                if isinstance(new_seq,List):
                    seq_cap_det_seg.append(new_seq)
                else:
                    new_seq = [new_seq]
                    seq_cap_det_seg.append(new_seq)

                all_indices.append((token_positive[0],"start",-1))
                all_indices.append((token_positive[1],"end",new_seq))

            caption_expr, result_seq = insert_phrases(caption_expr,all_indices,place_holder)

            conversations.append([{'from': 'human','value': question_gcg},
                                  {'from': 'gpt', 'value': caption_expr, seq_name: result_seq}])
            


        all_conversations = self.concat_conversations(conversations)
        ret['target'] = {type:boxes_or_masks}
        ret['conversations'] = all_conversations

        if task == 'detection':
            ret['conversations'].insert(0,{'from':'system','value':[{'task':{'task_name':'gcg_detection',
                                                                        'element':['phrase','sentence'],'use_unit':True},'unit':['box']} 
                                                               for _ in range(len(ret['conversations'])//2)]})
        elif task == 'segmentation':
            ret['conversations'].insert(0,{'from':'system','value':[{'task':{'task_name':'gcg_segmentation',
                                                                        'element':['phrase','sentence'],'use_unit':True},'unit':['mask']} 
                                                               for _ in range(len(ret['conversations'])//2)]})        

        return ret
    
    def mix(self,ret,objects,floating_objects,captions,ratio,random_select=False,length=None):
        '''mix all tasks'''
        if self.use_floating_objects:
            objects = objects + floating_objects

        objects = sort_objects(objects,filter=False)
        
        # define box and mask basic variables
        det_dict = {
            'type': 'boxes',
            'seq_name':'boxes_seq',
            'place_holder': BOXES_PLACEHOLDER,
            'bboxes': [],
            # 'box_caption':'',
            'conversations':{
                # 'conversations_det':None,
                'ground_conversations':[],
                'rec_conversations':[],
                'reg_conversations':[],
                'cap_det_conversations':[],
            }

        }
        seg_dict = {
            'type': 'masks',
            'seq_name':'masks_seq',
            'place_holder': MASKS_PLACEHOLDER,
            'masks': [],
            # 'mask_caption':'',
            'conversations':{
                # 'conversations_seg':None,
                'ground_conversations':[],
                'rec_conversations':[],
                'reg_conversations':[],
                'cap_seg_conversations':[],
            }

        }


        cls_names = []
        box_mask_seq = []
        id_map = {}

        for i,object in enumerate(objects):
            box = resize_box(object['bbox'],width=ret['image']['width'],
                             height=ret['image']['height'],ratio=ratio)
            mask = resize_mask(object['segmentation'],width=ret['image']['width'],
                             height=ret['image']['height'],ratio=ratio)
            det_dict['bboxes'].append(box)
            seg_dict['masks'].append(mask)

            id_map[f"id_{object['id']}"] = i

            if 'label' in object.keys():
                cls_name = object['label']
            else:
                cls_name = object['labels'][0]
            cls_name = cls_name.replace("_"," ")


            if cls_name in cls_names:  # some class may have two or more boxes
                previous_seq = cls_names.index(cls_name)
                box_mask_seq[previous_seq].append(i)

                # generate grounding detection & segmentation captions
                det_dict['conversations']['ground_conversations'][previous_seq][1]['value'] += det_dict['place_holder']  # add one <boxes> in the conversation
                det_dict['conversations']['ground_conversations'][previous_seq][1][det_dict['seq_name']][0].append(i)
                seg_dict['conversations']['ground_conversations'][previous_seq][1]['value'] += seg_dict['place_holder']  # add one <masks> in the conversation
                seg_dict['conversations']['ground_conversations'][previous_seq][1][seg_dict['seq_name']][0].append(i)

                # generate detection & segmentation captions
                # cls_captions_boxes[previous_seq] = cls_captions_boxes[previous_seq].replace(f"{cls_name}{PHRASE_ED_PLACEHOLDER_STAGE2}",
                #                                                           f"{cls_name}{PHRASE_ED_PLACEHOLDER_STAGE2}{det_dict['place_holder']}")
                # cls_captions_masks[previous_seq] = cls_captions_masks[previous_seq].replace(f"{cls_name}{PHRASE_ED_PLACEHOLDER_STAGE2}",
                #                                                           f"{cls_name}{PHRASE_ED_PLACEHOLDER_STAGE2}{seg_dict['place_holder']}")

            else:
                seq = [i]
                cls_names.append(cls_name)
                box_mask_seq.append(seq)

                # generate grounding detection & segmentation captions
                seq_cond_det = [i]
                seq_cond_seg = [i]
                question_cond_det = self.get_template_from_dict('Cond_DET')
                cls_name_cond = cls_name
                if object['attributes'] is not None and object['attributes'] != []:
                    attributes = object['attributes'][0]

                    # generate reg detection & segmentation conversations
                    question_reg_det = self.get_template_from_dict('REG')
                    question_reg_det = question_reg_det.replace(OBJS_PLACEHOLDER,BOXES_PLACEHOLDER)
                    question_reg_seg = self.get_template_from_dict('REG_SEG')
                    single_conversation_det = [{'from': 'human','value': question_reg_det,det_dict['seq_name']:[[i]]},
                                                     {'from': 'gpt', 'value': attributes}]
                    single_conversation_seg = [{'from': 'human','value': question_reg_seg,seg_dict['seq_name']:[[i]]},
                                                     {'from': 'gpt', 'value': attributes}]

                    det_dict['conversations']['reg_conversations'].append(single_conversation_det)
                    seg_dict['conversations']['reg_conversations'].append(single_conversation_seg)

                    rand_prob = random.uniform(0,1)
                    if rand_prob >= 0.8:
                        cls_name_cond = attributes
                question_cond_det = question_cond_det.replace(CLASS_PLACEHOLDER,cls_name_cond)
                conversation_human_cond_det = {'from': 'human','value': question_cond_det}
                value_ground_det = PHRASE_ST_PLACEHOLDER_STAGE2 + 'target' + PHRASE_ED_PLACEHOLDER_STAGE2 + det_dict['place_holder']
                conversation_gpt_cond_det = {'from': 'gpt', 'value': value_ground_det, det_dict['seq_name']: [seq_cond_det]}
                single_conversation_cond_det = [conversation_human_cond_det,conversation_gpt_cond_det]
                det_dict['conversations']['ground_conversations'].append(single_conversation_cond_det)
                question_cond_seg = self.get_template_from_dict('Cond_SEG')
                question_cond_seg = question_cond_seg.replace(CLASS_PLACEHOLDER,cls_name_cond)
                conversation_human_cond_seg = {'from': 'human','value': question_cond_seg}
                value_ground_seg = PHRASE_ST_PLACEHOLDER_STAGE2 + 'target' + PHRASE_ED_PLACEHOLDER_STAGE2 + seg_dict['place_holder']
                conversation_gpt_cond_seg = {'from': 'gpt', 'value': value_ground_seg, seg_dict['seq_name']: [seq_cond_seg]}
                single_conversation_cond_seg = [conversation_human_cond_seg,conversation_gpt_cond_seg]
                seg_dict['conversations']['ground_conversations'].append(single_conversation_cond_seg)

                # generate detection & segmentation captions
        #         box_caption = PHRASE_ST_PLACEHOLDER_STAGE2 + cls_name + PHRASE_ED_PLACEHOLDER_STAGE2 + det_dict['place_holder']
        #         mask_caption = PHRASE_ST_PLACEHOLDER_STAGE2 + cls_name + PHRASE_ED_PLACEHOLDER_STAGE2 + seg_dict['place_holder']
        #         cls_captions_boxes.append(box_caption)
        #         cls_captions_masks.append(mask_caption)


        # det_dict['box_caption'] = ', '.join(cls_captions_boxes)
        # seg_dict['mask_caption'] = ','.join(cls_captions_masks)
        ret['image']['width'] = int(ret['image']['width']*ratio)
        ret['image']['height'] = int(ret['image']['height']*ratio)

        cap_conversations = []
        for j,caption in enumerate(captions):

            # generate caption conversation
            question_cap = self.get_template_from_dict('image_cap')
            cation_cap = caption['caption']

            # generate caption + detection conversation
            caption_expr_det = caption['caption']
            caption_expr_seg = caption['caption']
            question_cap_det = self.get_template_from_dict('flickr30k')
            question_cap_seg = self.get_template_from_dict('flickr30k_SEG')
            seq_cap_det_seg = []
            if caption['is_dense']:
                question_cap_det = self.dense_question(question_cap_det)
                question_cap_seg = self.dense_question(question_cap_seg)
                question_cap = self.dense_question(question_cap)

            all_indices = []
            # place_holders_det = ''
            # place_holders_seg = ''
            for detail in caption['details']:
                phrase = detail['phrase']
                token_positive = detail['tokens_positive']
                if token_positive is None:
                    continue
                if 'ids' in detail.keys():
                    reform_ids = []
                    for id in detail['ids']:
                        reform_ids.append(id_map[f"id_{id}"])
                    detail['ids'] = reform_ids
                    rec_seq = detail['ids']
                    reg_seq = detail['ids']
                    # generate rec detection & segmentation answers
                    value_rec_det = PHRASE_ST_PLACEHOLDER_STAGE2 + 'target' + PHRASE_ED_PLACEHOLDER_STAGE2 + det_dict['place_holder'] * len(rec_seq)
                    value_rec_seg = PHRASE_ST_PLACEHOLDER_STAGE2 + 'target' + PHRASE_ED_PLACEHOLDER_STAGE2 + seg_dict['place_holder'] * len(rec_seq)
                    conversation_gpt_rec_det = {'from': 'gpt', 'value': value_rec_det, 
                                                det_dict['seq_name']: [rec_seq]}
                    conversation_gpt_rec_seg = {'from': 'gpt', 'value': value_rec_seg, 
                                                seg_dict['seq_name']: [rec_seq]}
                else:
                    detail['id'] = id_map[f"id_{detail['id']}"]
                    rec_seq = [detail['id']]
                    reg_seq = detail['id']
                    # generate rec detection & segmentation answers
                    value_rec_det = PHRASE_ST_PLACEHOLDER_STAGE2 + 'target' + PHRASE_ED_PLACEHOLDER_STAGE2 + det_dict['place_holder']
                    value_rec_seg = PHRASE_ST_PLACEHOLDER_STAGE2 + 'target' + PHRASE_ED_PLACEHOLDER_STAGE2 + seg_dict['place_holder']
                    conversation_gpt_rec_det = {'from': 'gpt', 'value': value_rec_det, 
                                                det_dict['seq_name']: [rec_seq]}
                    conversation_gpt_rec_seg = {'from': 'gpt', 'value': value_rec_seg, 
                                                seg_dict['seq_name']: [rec_seq]}

                # generate rec detection & segmentation conversations
                question_rec_det = self.get_template_from_dict('REC')
                question_rec_seg = self.get_template_from_dict('RES')
                question_rec_det = question_rec_det.replace(EXPR_PLACEHOLDER,phrase)
                question_rec_seg = question_rec_seg.replace(EXPR_PLACEHOLDER,phrase)
                conversation_human_rec_det = {'from': 'human','value': question_rec_det}
                conversation_human_rec_seg = {'from': 'human','value': question_rec_seg}
                single_conversation_rec_det = [conversation_human_rec_det,conversation_gpt_rec_det]
                single_conversation_rec_seg = [conversation_human_rec_seg,conversation_gpt_rec_seg]
                det_dict['conversations']['rec_conversations'].append(single_conversation_rec_det)
                seg_dict['conversations']['rec_conversations'].append(single_conversation_rec_seg)


                # generate caption + detection conversation
                if isinstance(reg_seq,List):
                    seq_cap_det_seg.append(reg_seq)
                else:
                    reg_seq = [reg_seq]
                    seq_cap_det_seg.append(reg_seq)

                all_indices.append((token_positive[0],"start",-1))
                all_indices.append((token_positive[1],"end",reg_seq))

            caption_expr_det = insert_phrases(caption_expr_det,all_indices,det_dict['place_holder'])
            caption_expr_seg = insert_phrases(caption_expr_seg,all_indices,seg_dict['place_holder'])

            det_dict['conversations']['cap_det_conversations'].append([{'from': 'human','value': question_cap_det},
                                                                       {'from': 'gpt', 'value': caption_expr_det,det_dict['seq_name']: seq_cap_det_seg}])
            seg_dict['conversations']['cap_seg_conversations'].append([{'from': 'human','value': question_cap_seg},
                                                                       {'from': 'gpt', 'value': caption_expr_seg,seg_dict['seq_name']: seq_cap_det_seg}])

            # generate caption conversation
            single_conversation_cap = [{'from': 'human','value': question_cap},{'from': 'gpt','value': cation_cap}]
            cap_conversations.append(single_conversation_cap)



        # construct multi-turn conversations and random selection
        if random_select:
            det_dict['conversations']['ground_conversations'] = self.random_select(det_dict['conversations']['ground_conversations'],length=1)
            seg_dict['conversations']['ground_conversations'] = self.random_select(seg_dict['conversations']['ground_conversations'],length=1)
            det_dict['conversations']['rec_conversations'] = self.random_select(det_dict['conversations']['rec_conversations'],length=1)
            seg_dict['conversations']['rec_conversations'] = self.random_select(seg_dict['conversations']['rec_conversations'],length=1)
            det_dict['conversations']['reg_conversations'] = self.random_select(det_dict['conversations']['reg_conversations'],length=length)
            seg_dict['conversations']['reg_conversations'] = self.random_select(seg_dict['conversations']['reg_conversations'],length=length)
            det_dict['conversations']['cap_det_conversations'] = self.random_select(det_dict['conversations']['cap_det_conversations'],length=length)
            seg_dict['conversations']['cap_seg_conversations'] = self.random_select(seg_dict['conversations']['cap_seg_conversations'],length=length)
            cap_conversations = self.random_select(cap_conversations)
        
        det_dict['conversations']['ground_conversations'] = self.concat_conversations(det_dict['conversations']['ground_conversations'])
        seg_dict['conversations']['ground_conversations'] = self.concat_conversations(seg_dict['conversations']['ground_conversations'])
        det_dict['conversations']['rec_conversations'] = self.concat_conversations(det_dict['conversations']['rec_conversations'])
        seg_dict['conversations']['rec_conversations'] = self.concat_conversations(seg_dict['conversations']['rec_conversations'])
        det_dict['conversations']['reg_conversations'] = self.concat_conversations(det_dict['conversations']['reg_conversations'])
        seg_dict['conversations']['reg_conversations'] = self.concat_conversations(seg_dict['conversations']['reg_conversations'])
        det_dict['conversations']['cap_det_conversations'] = self.concat_conversations(det_dict['conversations']['cap_det_conversations'])
        seg_dict['conversations']['cap_seg_conversations'] = self.concat_conversations(seg_dict['conversations']['cap_seg_conversations'])
        cap_conversations = self.concat_conversations(cap_conversations)

        # generate system values
        system_value_det = {key:None for key in list(det_dict['conversations'].keys())}
        system_value_seg = {key:None for key in list(seg_dict['conversations'].keys())}
        # system_value_det['conversations_det'] = [{'task':{'task_name':'detection','element':['phrase'],'use_unit':True},'unit':['box']}]
        system_value_det['ground_conversations'] = [{'task':{'task_name':'grounding_detection','element':['phrase'],'use_unit':True},'unit':['box']} for _ in range(len(det_dict['conversations']['ground_conversations'])//2)]
        system_value_det['rec_conversations'] = [{'task':{'task_name':'grounding_detection','element':['phrase'],'use_unit':True},'unit':['box']} for _ in range(len(det_dict['conversations']['rec_conversations'])//2)]
        # system_value_det['reg_conversations'] = [{'task':{'task_name':'vqa','element':['sentence'],'use_unit':False}} for _ in range(len(det_dict['conversations']['reg_conversations'])//2)]
        system_value_det['reg_conversations'] = [{'task':{'task_name':'referring vqa','element':['sentence'],'use_unit':False}} for _ in range(len(det_dict['conversations']['reg_conversations'])//2)]
        system_value_det['cap_det_conversations'] = [{'task':{'task_name':'gcg_detection','element':['phrase','sentence'],'use_unit':True},'unit':['box']} for _ in range(len(det_dict['conversations']['cap_det_conversations'])//2)]

        # system_value_seg['conversations_seg'] = [{'task':{'task_name':'segmentation','element':['phrase'],'use_unit':True},'unit':['mask']}]
        system_value_seg['ground_conversations'] = [{'task':{'task_name':'grounding_segmentation','element':['phrase'],'use_unit':True},'unit':['mask']} for _ in range(len(seg_dict['conversations']['ground_conversations'])//2)]
        system_value_seg['rec_conversations'] = [{'task':{'task_name':'grounding_segmentation','element':['phrase'],'use_unit':True},'unit':['mask']} for _ in range(len(seg_dict['conversations']['rec_conversations'])//2)]
        # system_value_seg['reg_conversations'] = [{'task':{'task_name':'vqa','element':['sentence'],'use_unit':False}} for _ in range(len(seg_dict['conversations']['reg_conversations'])//2)]
        system_value_seg['reg_conversations'] = [{'task':{'task_name':'referring vqa','element':['sentence'],'use_unit':False}} for _ in range(len(seg_dict['conversations']['reg_conversations'])//2)]
        system_value_seg['cap_seg_conversations'] = [{'task':{'task_name':'gcg_segmentation','element':['phrase','sentence'],'use_unit':True},'unit':['mask']} for _ in range(len(seg_dict['conversations']['cap_seg_conversations'])//2)]

        system_value_cap = [{'task':{'task_name':'vqa','element':['sentence'],'use_unit':False}} for _ in range(len(cap_conversations)//2)]


        # concat all conversations
        all_conversations = []
        all_system_values = []
        for conversation_name in det_dict['conversations'].keys():
            all_conversations.append(det_dict['conversations'][conversation_name])
            all_system_values.append(system_value_det[conversation_name])

            assert len(det_dict['conversations'][conversation_name]) // 2 == len(system_value_det[conversation_name])
            
        for conversation_name in seg_dict['conversations'].keys():
            all_conversations.append(seg_dict['conversations'][conversation_name])
            all_system_values.append(system_value_seg[conversation_name])

            assert len(seg_dict['conversations'][conversation_name]) // 2 == len(system_value_seg[conversation_name])
            
        all_conversations.append(cap_conversations)
        all_system_values.append(system_value_cap)

       

        if random_select:
            all_conversations,all_system_values = self.random_select(all_conversations,system_value=all_system_values)
        all_system_values = self.concat_conversations(all_system_values)
        all_conversations = self.concat_conversations(all_conversations,concat_all=True)

        assert len(all_conversations) // 2 == len(all_system_values)

        ret['target'] = {det_dict['type']:det_dict['bboxes'],seg_dict['type']:seg_dict['masks']}
        ret['conversations'] = all_conversations
        remove_idx = self.select_target(ret)
        for idx in sorted(remove_idx,reverse=True):
            assert idx % 2 == 0
            idx = idx // 2
            del all_system_values[idx]

        ret['conversations'].insert(0,{'from':'system','value':all_system_values})
        

        return ret
    

    def seg_det(self,ret,objects,floating_objects,ratio):
        if self.use_floating_objects:
            objects = objects + floating_objects

        det_dict = {
            'type': 'boxes',
            'seq_name':'boxes_seq',
            'place_holder': BOXES_PLACEHOLDER,
            'bboxes': [],
            'box_caption': '',
            'conversations':{
                'conversations_det':None,
            }
        }
        seg_dict = {
            'type': 'masks',
            'seq_name':'masks_seq',
            'place_holder': MASKS_PLACEHOLDER,
            'masks': [],
            'mask_caption': '',
            'conversations':{
                'conversations_seg':None,
            }
        }

        objects = delete_objects(objects,20)
        box_mask_seq = []
        cls_captions_boxes = []
        cls_captions_masks = []
        cls_names = []
        for i,object in enumerate(objects):
            box = resize_box(object['bbox'],width=ret['image']['width'],
                             height=ret['image']['height'],ratio=ratio)
            mask = resize_mask(object['segmentation'],width=ret['image']['width'],
                             height=ret['image']['height'],ratio=ratio)
            det_dict['bboxes'].append(box)
            seg_dict['masks'].append(mask)
            if 'label' in object.keys():
                cls_name = object['label']
            else:
                cls_name = object['labels'][0]
            cls_name = cls_name.replace("_"," ")

            if cls_name in cls_names:  # some class may have two or more boxes
                previous_seq = cls_names.index(cls_name)
                box_mask_seq[previous_seq].append(i)
                # generate detection & segmentation captions
                cls_captions_boxes[previous_seq] = cls_captions_boxes[previous_seq].replace(f"{cls_name}{PHRASE_ED_PLACEHOLDER_STAGE2}",
                                                                          f"{cls_name}{PHRASE_ED_PLACEHOLDER_STAGE2}{det_dict['place_holder']}")
                cls_captions_masks[previous_seq] = cls_captions_masks[previous_seq].replace(f"{cls_name}{PHRASE_ED_PLACEHOLDER_STAGE2}",
                                                                          f"{cls_name}{PHRASE_ED_PLACEHOLDER_STAGE2}{seg_dict['place_holder']}")

            else:
                seq = [i]
                cls_names.append(cls_name)
                box_mask_seq.append(seq)
                # generate detection & segmentation captions
                box_caption = PHRASE_ST_PLACEHOLDER_STAGE2 + cls_name + PHRASE_ED_PLACEHOLDER_STAGE2 + det_dict['place_holder']
                mask_caption = PHRASE_ST_PLACEHOLDER_STAGE2 + cls_name + PHRASE_ED_PLACEHOLDER_STAGE2 + seg_dict['place_holder']
                cls_captions_boxes.append(box_caption)
                cls_captions_masks.append(mask_caption)
            

        # construct detection_segmentation template and conversations
        question_det = self.get_template_from_dict('DET')
        question_seg = self.get_template_from_dict('SEG')
        det_dict['box_caption'] = ', '.join(cls_captions_boxes)
        seg_dict['mask_caption'] = ','.join(cls_captions_masks)
        ret['image']['width'] = int(ret['image']['width']*ratio)
        ret['image']['height'] = int(ret['image']['height']*ratio)
        det_dict['conversations']['conversations_det'] = [
                             {'from': 'system', 'value': [{'task':{'task_name':'detection','element':['phrase'],'use_unit':True},'unit':['box']}]},
                             {'from': 'human','value': question_det},
                             {'from': 'gpt','value': det_dict['box_caption'], det_dict['seq_name']: box_mask_seq}]
        seg_dict['conversations']['conversations_seg'] = [
                             {'from': 'system', 'value': [{'task':{'task_name':'segmentation','element':['phrase'],'use_unit':True},'unit':['mask']}]},
                             {'from': 'human','value': question_seg},
                             {'from': 'gpt','value': seg_dict['mask_caption'], seg_dict['seq_name']: box_mask_seq}]
        rand_num = random.choice([0,1])
        if rand_num == 0:
            ret['target'] = {det_dict['type']:det_dict['bboxes']}
            ret['conversations'] = det_dict['conversations']['conversations_det'] 
        elif rand_num == 1:
            ret['target'] = {seg_dict['type']:seg_dict['masks']}
            ret['conversations'] = seg_dict['conversations']['conversations_seg']

        return ret
    

    def make_conversations(self,ret,annotations,ratio):
        objects = annotations['objects']
        floating_objects = annotations['floating_objects']
        short_captions = annotations['short_captions']
        dense_caption = annotations['dense_caption']

        dense_caption['is_dense'] = True
        for i,caption in enumerate(short_captions):
            short_captions[i]['is_dense'] = False
        short_captions.append(dense_caption)
        captions = short_captions

        if self.version == 'c':
            ret = self.caption(ret,captions,random_select=False)
            return ret

        elif self.version == 'd':
            task = 'detection'
            ret = self.detection_segmentation(task,ret,objects,floating_objects)
            return ret
        
        elif self.version == 's':
            task = 'segmentation'
            ret = self.detection_segmentation(task,ret,objects,floating_objects)
            return ret            
        
        elif self.version == 'cond_d':
            task = 'detection'
            ret = self.grounding_detection_segmentation(task,ret,objects,ratio,random_select=True,length=self.length)
            return ret  

        elif self.version == 'cond_s':
            task = 'segmentation'
            ret = self.grounding_detection_segmentation(task,ret,objects,ratio,random_select=True,length=self.length)
            return ret  
        
        elif self.version == 'r_det':
            task = 'detection'
            ret = self.rec(task,ret,objects,floating_objects,captions,random_select=False)
            return ret  

        elif self.version == 'r_seg':
            task = 'segmentation'
            ret = self.rec(task,ret,objects,floating_objects,captions,random_select=False)
            return ret
        
        elif self.version == 're_det':
            task = 'detection'
            ret = self.reg(task,ret,objects,ratio,length=self.length)
            return ret  

        elif self.version == 're_seg':
            task = 'segmentation'
            ret = self.reg(task,ret,objects,ratio,length=self.length)
            return ret
        
        elif self.version == 'c_d':
            task = 'detection'
            ret = self.caption_detection_segmentation(task,ret,objects,floating_objects,captions,ratio,random_select=False)
            return ret  
        
        elif self.version == 'c_s':
            task = 'segmentation'
            ret = self.caption_detection_segmentation(task,ret,objects,floating_objects,captions,ratio,random_select=False)
            return ret  
        
        elif self.version == 'd_s':
            ret = self.seg_det(ret,objects,floating_objects,ratio)
            return ret
        
        elif self.version == 'mix':
            ret = self.mix(ret,objects,floating_objects,captions,random_select=True,length=self.length,ratio=ratio)
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
            if len(offline_item['conversations']) > 1:
                if IMAGE_PLACEHOLDER not in offline_item['conversations'][1]['value']:
                    offline_item['conversations'][1]['value'] = IMAGE_PLACEHOLDER + offline_item['conversations'][1]['value']
            return offline_item
        text_file = self.text_path_file[index]
        annotations_json = self.get_file_data(os.path.join(self.text_path,text_file))
        img_path = text_file[:-5] + '.jpg'
        annotations = annotations_json[img_path]
        try:
            shape = annotations['objects'][0]['segmentation']['size']
        except:
            shape = annotations['floating_objects'][0]['segmentation']['size']
        image_path_abs = os.path.join(self.image_folder,img_path)
        ret = {}
        
        ret['image'] = {'path': image_path_abs,'width':int(shape[1]),'height':int(shape[0])}

        ret = self.make_conversations(ret,annotations,ratio=self.ratio)
        ret['map_placeholders'] = self.map_placeholders

        return ret
