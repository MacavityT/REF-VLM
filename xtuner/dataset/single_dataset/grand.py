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
        "end":PHRASE_ED_PLACEHOLDER_STAGE2 + place_holder,
    }
    for index, phrase, seq in sorted(indices, reverse=True):
        input_str = input_str[:index] + phrase_map[phrase] + input_str[index:]
    return input_str

def sort_objects(objects):
    for object in objects:
        object['segmentation'] = decode(object['segmentation'])
        area = object['segmentation'].sum().item()
        object['area'] = area
    objects = sorted(objects, key=lambda x: x['area'],reverse=True)
    for i,item in enumerate(objects):
        if i > 9:
            item['segmentation'] = None
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

    def __init__(self, *args,version,use_floating_objects=True,max_conv_length=None,**kwargs):
        super().__init__(*args, **kwargs)
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
        boxes = target['boxes']
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

        item['target']['boxes'] = selected_boxes
        item['target']['masks'] = selected_masks 
        item['conversations'] = select_conversations

        return remove_idx
    
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
        all_conversations.append(self.concat_conversations(conversations))
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

    def grounding_detection_segmentation(self,task,ret,objects,floating_objects,template_name=None,random_select=False,length=None):

        if self.use_floating_objects:
            objects = objects + floating_objects
        
        boxes_or_masks = []
        cls_names = []
        conversations = []
        
        for i,object in enumerate(objects):
            cls_name = ', '.join(object['labels'])
            cls_name = cls_name.replace('_',' ')
            if task == 'detection':
                boxes_or_masks.append(object['bbox'])
                type = 'boxes'
                unit_task = {'task_name':'grounding_detection','element':['phrase'],'use_unit':True}
                unit= ['box']
                seq_name = 'boxes_seq'
                place_holder = BOXES_PLACEHOLDER
            elif task == 'segmentation':
                boxes_or_masks.append(decode(object['segmentation']))
                unit_task = {'task_name':'grounding_segmentation','element':['phrase'],'use_unit':True}
                unit= ['mask']
                type = 'masks'
                seq_name = 'masks_seq'
                place_holder = MASKS_PLACEHOLDER
            else:
                raise "Please select valid template: DET or SEG!"
            
            if object['attributes'] is not None:
                attributes = ', '.join(object['attributes'])
                rand_prob = random.uniform(0,1)
                if rand_prob >= 0.5:
                    cls_name = attributes


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

        if random_select:
            conversations = self.random_select(conversations,length)
        
        all_conversations = []
        all_conversations.append({'from':'system','value':[{'task':unit_task,'unit':unit} for _ in range(len(conversations))]})
        all_conversations.append(self.concat_conversations(conversations))
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
        all_conversations.append(self.concat_conversations(conversations))
        ret['target'] = {type:boxes_or_masks}
        ret['conversations'] = all_conversations

        return ret
    

    def reg(self,task,ret,objects,floating_objects,captions,template_name=None,random_select=False,length=None):

        if self.use_floating_objects:
            objects = objects + floating_objects
        if task == 'detection':
            unit_task = {'task_name':'vqa','element':['sentence'],'use_unit':False}
            type = 'boxes'
            seq_name = 'boxes_seq'
            place_holder = BOXES_PLACEHOLDER
        elif task == 'segmentation':
            unit_task = {'task_name':'vqa','element':['sentence'],'use_unit':False}
            type = 'masks'
            seq_name = 'masks_seq'
            place_holder = MASKS_PLACEHOLDER
        else:
            raise "Please select valid template: REG or REG_SEG!"
        
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
                else:
                    seq = detail['id']
                if not isinstance(self.template_name,List):
                    question = self.get_template()
                else:
                    assert template_name is not None
                    question = self.get_template_from_dict(template_name)

                if task == 'detection':
                    question = question.replace(OBJS_PLACEHOLDER,BOXES_PLACEHOLDER)
                elif task == 'segmentation':
                    question = question
                if j != 0:
                    question = question.replace(IMAGE_PLACEHOLDER,'')
                single_conversation_dense = []
                single_conversation_short = []
                if isinstance(seq,List):
                    for id in seq:
                        conversation_human = {'from': 'human','value': question,seq_name:[id]}
                        conversation_gpt = {'from': 'gpt', 'value': phrase}
                        single_conversation_dense += [conversation_human,conversation_gpt]
                else:
                    conversation_human = {'from': 'human','value': question,seq_name:[seq]}
                    conversation_gpt = {'from': 'gpt', 'value': phrase}
                    single_conversation_short = [conversation_human,conversation_gpt]
                
                single_conversation = single_conversation_dense + single_conversation_short
                conversations.append(single_conversation)                

        if random_select:
            conversations = self.random_select(conversations,length)

        all_conversations = []
        all_conversations.append({'from':'system','value':[{'task':unit_task} for _ in range(len(conversations))]})
        all_conversations.append(self.concat_conversations(conversations))
        ret['target'] = {type:boxes_or_masks}
        ret['conversations'] = all_conversations

        return ret
    
    def caption_detection_segmentation(self,task,ret,objects,floating_objects,captions,
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
        
        boxes_or_masks = []
        for i,object in enumerate(objects):
            if task == 'detection':
                boxes_or_masks.append(object['bbox'])
            elif task == 'segmentation':
                boxes_or_masks.append(decode(object['segmentation']))
            else:
                raise "Please select valid template: flickr30k or flickr30k_SEG!"
            
        conversations = []
        for j,caption in enumerate(captions):
            caption_expr = caption['caption']
            seqs = []
            if not isinstance(self.template_name,List):
                question = self.get_template()
            else:
                assert template_name is not None
                question = self.get_template_from_dict(template_name)
            if caption['is_dense']:
                question += ' '
                question += random.choice(self.detailed_template)   

            for detail in caption['details']:
                phrase = detail['phrase']
                if 'ids' in detail.keys():
                    seq = detail['ids']
                else:
                    seq = detail['id']

       
                if j != 0:
                    question = question.replace(IMAGE_PLACEHOLDER,'')

                if isinstance(seq,List):
                    place_holders = len(seq) * place_holder
                else:
                    place_holders = place_holder
                caption_expr = caption_expr.replace(phrase,PHRASE_ST_PLACEHOLDER_STAGE2 + phrase + PHRASE_ED_PLACEHOLDER_STAGE2 + place_holders)  
                seqs.append(seq)

            
            conversation_human = {'from': 'human','value': question}
            conversation_gpt = {'from': 'gpt', 'value': caption_expr, seq_name: seqs}

            single_conversation = [conversation_human,conversation_gpt]
            conversations.append(single_conversation)

        if random_select:
            conversations = self.random_select(conversations,length)

        all_conversations = []
        all_conversations.append({'from':'system','value':[{'task':unit_task,'unit':unit} for _ in range(len(conversations))]})
        all_conversations.append(self.concat_conversations(conversations))
        ret['target'] = {type:boxes_or_masks}
        ret['conversations'] = all_conversations

        return ret
    
    def mix(self,ret,objects,floating_objects,captions,ratio,random_select=False,length=None):
        '''mix all tasks'''
        if self.use_floating_objects:
            objects = objects + floating_objects

        objects = sort_objects(objects)
        
        # define box and mask basic variables
        det_dict = {
            'type': 'boxes',
            'seq_name':'boxes_seq',
            'place_holder': BOXES_PLACEHOLDER,
            'bboxes': [],
            'box_caption':'',
            'conversations':{
                'conversations_det':None,
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
            'mask_caption':'',
            'conversations':{
                'conversations_seg':None,
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

            cls_name = ', '.join(object['labels'])
            cls_name = cls_name.replace('_',' ')


            if cls_name in cls_names:  # some class may have two or more boxes
                previous_seq = cls_names.index(cls_name)
                box_mask_seq[previous_seq].append(i)

                # generate grounding detection & segmentation captions
                det_dict['conversations']['ground_conversations'][previous_seq][1]['value'] += det_dict['place_holder']  # add one <boxes> in the conversation
                det_dict['conversations']['ground_conversations'][previous_seq][1][det_dict['seq_name']][0].append(i)
                seg_dict['conversations']['ground_conversations'][previous_seq][1]['value'] += seg_dict['place_holder']  # add one <masks> in the conversation
                seg_dict['conversations']['ground_conversations'][previous_seq][1][seg_dict['seq_name']][0].append(i)

                # generate detection & segmentation captions
                det_dict['box_caption'] = det_dict['box_caption'].replace(f"{cls_name}{PHRASE_ED_PLACEHOLDER_STAGE2}",
                                                                          f"{cls_name}{PHRASE_ED_PLACEHOLDER_STAGE2}{det_dict['place_holder']}")
                seg_dict['mask_caption'] = seg_dict['mask_caption'].replace(f"{cls_name}{PHRASE_ED_PLACEHOLDER_STAGE2}",
                                                                          f"{cls_name}{PHRASE_ED_PLACEHOLDER_STAGE2}{seg_dict['place_holder']}")
            else:
                seq = [i]
                cls_names.append(cls_name)
                box_mask_seq.append(seq)

                # generate grounding detection & segmentation captions
                seq_cond_det = [i]
                seq_cond_seg = [i]
                question_cond_det = self.get_template_from_dict('Cond_DET')
                cls_name_cond = cls_name
                if object['attributes'] is not None:
                    attributes = ', '.join(object['attributes'])
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
                det_dict['box_caption'] = det_dict['box_caption'] + PHRASE_ST_PLACEHOLDER_STAGE2 + cls_name + PHRASE_ED_PLACEHOLDER_STAGE2 + det_dict['place_holder'] + ', '
                seg_dict['mask_caption'] = seg_dict['mask_caption'] + PHRASE_ST_PLACEHOLDER_STAGE2 + cls_name + PHRASE_ED_PLACEHOLDER_STAGE2 + seg_dict['place_holder'] + ', '

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
                cation_cap = self.dense_question(question_cap)

            all_indices = []
            place_holders_det = ''
            place_holders_seg = ''
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
                    value_rec_det = PHRASE_ST_PLACEHOLDER_STAGE2 + 'target' + PHRASE_ED_PLACEHOLDER_STAGE2 + det_dict['place_holder'] * len(seq)
                    value_rec_seg = PHRASE_ST_PLACEHOLDER_STAGE2 + 'target' + PHRASE_ED_PLACEHOLDER_STAGE2 + seg_dict['place_holder'] * len(seq)
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


                # generate reg detection & segmentation conversations
                question_reg_det = self.get_template_from_dict('REG')
                question_reg_det = question_reg_det.replace(OBJS_PLACEHOLDER,BOXES_PLACEHOLDER)
                question_reg_seg = self.get_template_from_dict('REG_SEG')
                question_reg_seg = question_reg_seg
                single_conversation_dense_det = []
                single_conversation_dense_seg = []
                single_conversation_short_det = []
                single_conversation_short_seg = []
                if isinstance(reg_seq,List):
                    for id in reg_seq:
                        single_conversation_dense_det += [{'from': 'human','value': question_reg_det,det_dict['seq_name']:[[id]]},
                                                          {'from': 'gpt', 'value': phrase}]
                        single_conversation_dense_seg += [{'from': 'human','value': question_reg_seg,seg_dict['seq_name']:[[id]]},
                                                          {'from': 'gpt', 'value': phrase}]
                else:
                    single_conversation_short_det = [{'from': 'human','value': question_reg_det,det_dict['seq_name']:[[reg_seq]]},
                                                     {'from': 'gpt', 'value': phrase}]
                    single_conversation_short_seg = [{'from': 'human','value': question_reg_seg,seg_dict['seq_name']:[[reg_seq]]},
                                                     {'from': 'gpt', 'value': phrase}]
                
                det_dict['conversations']['reg_conversations'].append(single_conversation_dense_det + single_conversation_short_det)
                seg_dict['conversations']['reg_conversations'].append(single_conversation_dense_seg + single_conversation_short_seg)

                # generate caption + detection conversation
                if isinstance(reg_seq,List):
                    place_holders_det = len(reg_seq) * det_dict['place_holder']
                    place_holders_seg = len(reg_seq) * seg_dict['place_holder']
                    seq_cap_det_seg.append(reg_seq)
                else:
                    place_holders_det = det_dict['place_holder']
                    place_holders_seg = seg_dict['place_holder']
                    reg_seq = [reg_seq]
                    seq_cap_det_seg.append(reg_seq)

                
                
                # TODO： 这里需要改
                # caption_expr_det = caption_expr_det.lower().replace(phrase,PHRASE_ST_PLACEHOLDER_STAGE2 + phrase + PHRASE_ED_PLACEHOLDER_STAGE2 + place_holders_det)  
                # caption_expr_seg = caption_expr_seg.lower().replace(phrase,PHRASE_ST_PLACEHOLDER_STAGE2 + phrase + PHRASE_ED_PLACEHOLDER_STAGE2 + place_holders_seg)
                # caption_expr_det = caption_expr_det.replace("<phrase>",PHRASE_ST_PLACEHOLDER_STAGE2)
                # caption_expr_det = caption_expr_det.replace("</phrase>",PHRASE_ED_PLACEHOLDER_STAGE2)
                # caption_expr_seg = caption_expr_seg.replace("<phrase>",PHRASE_ST_PLACEHOLDER_STAGE2)
                # caption_expr_seg = caption_expr_seg.replace("</phrase>",PHRASE_ED_PLACEHOLDER_STAGE2)

                all_indices.append((token_positive[0],"start",-1))
                all_indices.append((token_positive[1],"end",reg_seq))

            caption_expr_det = insert_phrases(caption_expr_det,all_indices,place_holders_det)
            caption_expr_seg = insert_phrases(caption_expr_seg,all_indices,place_holders_seg)

            det_dict['conversations']['cap_det_conversations'].append([{'from': 'human','value': question_cap_det},
                                                                       {'from': 'gpt', 'value': caption_expr_det,det_dict['seq_name']: seq_cap_det_seg}])
            seg_dict['conversations']['cap_seg_conversations'].append([{'from': 'human','value': question_cap_seg},
                                                                       {'from': 'gpt', 'value': caption_expr_seg,seg_dict['seq_name']: seq_cap_det_seg}])

            # generate caption conversation
            single_conversation_cap = [{'from': 'human','value': question_cap},{'from': 'gpt','value': cation_cap}]
            cap_conversations.append(single_conversation_cap)

        # construct detection_segmentation template and conversations
        question_det = self.get_template_from_dict('DET')
        question_seg = self.get_template_from_dict('SEG')
        det_dict['conversations']['conversations_det'] = [{'from': 'human','value': question_det},
                             {'from': 'gpt','value': det_dict['box_caption'], det_dict['seq_name']: box_mask_seq}]
        seg_dict['conversations']['conversations_seg'] = [{'from': 'human','value': question_seg},
                             {'from': 'gpt','value': seg_dict['mask_caption'], seg_dict['seq_name']: box_mask_seq}]

        # construct multi-turn conversations and random selection
        if random_select:
            det_dict['conversations']['ground_conversations'] = self.random_select(det_dict['conversations']['ground_conversations'],length=length)
            seg_dict['conversations']['ground_conversations'] = self.random_select(seg_dict['conversations']['ground_conversations'],length=length)
            det_dict['conversations']['rec_conversations'] = self.random_select(det_dict['conversations']['rec_conversations'],length=length)
            seg_dict['conversations']['rec_conversations'] = self.random_select(seg_dict['conversations']['rec_conversations'],length=length)
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
        system_value_det['conversations_det'] = [{'task':{'task_name':'detection','element':['phrase'],'use_unit':True},'unit':['box']}]
        system_value_det['ground_conversations'] = [{'task':{'task_name':'grounding_detection','element':['phrase'],'use_unit':True},'unit':['box']} for _ in range(len(det_dict['conversations']['ground_conversations'])//2)]
        system_value_det['rec_conversations'] = [{'task':{'task_name':'grounding_detection','element':['phrase'],'use_unit':True},'unit':['box']} for _ in range(len(det_dict['conversations']['rec_conversations'])//2)]
        system_value_det['reg_conversations'] = [{'task':{'task_name':'vqa','element':['sentence'],'use_unit':False}} for _ in range(len(det_dict['conversations']['reg_conversations'])//2)]
        system_value_det['cap_det_conversations'] = [{'task':{'task_name':'gcg_detection','element':['phrase','sentence'],'use_unit':True},'unit':['box']} for _ in range(len(det_dict['conversations']['cap_det_conversations'])//2)]

        system_value_seg['conversations_seg'] = [{'task':{'task_name':'segmentation','element':['phrase'],'use_unit':True},'unit':['mask']}]
        system_value_seg['ground_conversations'] = [{'task':{'task_name':'grounding_segmentation','element':['phrase'],'use_unit':True},'unit':['mask']} for _ in range(len(seg_dict['conversations']['ground_conversations'])//2)]
        system_value_seg['rec_conversations'] = [{'task':{'task_name':'grounding_segmentation','element':['phrase'],'use_unit':True},'unit':['mask']} for _ in range(len(seg_dict['conversations']['rec_conversations'])//2)]
        system_value_seg['reg_conversations'] = [{'task':{'task_name':'vqa','element':['sentence'],'use_unit':False}} for _ in range(len(seg_dict['conversations']['reg_conversations'])//2)]
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
            ret = self.grounding_detection_segmentation(task,ret,objects,floating_objects,random_select=False)
            return ret  

        elif self.version == 'cond_s':
            task = 'segmentation'
            ret = self.grounding_detection_segmentation(task,ret,objects,floating_objects,random_select=False)
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
            ret = self.reg(task,ret,objects,floating_objects,captions,random_select=False)
            return ret  

        elif self.version == 're_seg':
            task = 'segmentation'
            ret = self.reg(task,ret,objects,floating_objects,captions,random_select=False)
            return ret
        
        elif self.version == 'c_d':
            task = 'detection'
            ret = self.caption_detection_segmentation(task,ret,objects,floating_objects,captions,random_select=False)
            return ret  
        
        elif self.version == 'c_s':
            task = 'segmentation'
            ret = self.caption_detection_segmentation(task,ret,objects,floating_objects,captions,random_select=False)
            return ret  
        
        elif self.version == 'mix':
            ret = self.mix(ret,objects,floating_objects,captions,random_select=True,length=self.length,ratio=ratio)
            return ret
        
    def __len__(self):
        return len(self.text_path_file)

    def __getitem__(self, index):
        offline_item = super().__getitem__(index)
        if offline_item is not None:
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
        ratio = 0.3
        ret['image'] = {'path': image_path_abs,'width':int(shape[1]),'height':int(shape[0])}

        ret = self.make_conversations(ret,annotations,ratio=ratio)
        ret['map_placeholders'] = self.map_placeholders

        return ret
