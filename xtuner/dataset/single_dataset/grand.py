import os
import json
from typing import Dict, List
import numpy as np
import torch
import random
from torch.utils.data import Dataset
from pycocotools.mask import decode
from .mixin import MInstrDataset

from xtuner.registry import DATASETS
from xtuner.utils.constants import (
    IMAGE_PLACEHOLDER,
    OBJS_PLACEHOLDER,
    BOXES_PLACEHOLDER,
    MASK_PLACEHOLDER,
    MASKS_PLACEHOLDER,
    PHRASE_ST_PLACEHOLDER_STAGE2,
    PHRASE_ED_PLACEHOLDER_STAGE2,
    EXPR_PLACEHOLDER,
    CLASS_PLACEHOLDER
)




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
                boxes_or_masks.append(decode(object['segmentation']))
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
                unit_task = {'task_name':'grounding_detection','element':[],'use_unit':True}
                unit= ['box']
                seq_name = 'boxes_seq'
                place_holder = BOXES_PLACEHOLDER
            elif task == 'segmentation':
                boxes_or_masks.append(decode(object['segmentation']))
                unit_task = {'task_name':'grounding_segmentation','element':[],'use_unit':True}
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
                conversation_human = {'from': 'human','value': question}
                conversation_gpt = {'from': 'gpt', 'value': place_holder, seq_name: [box_mask_seq]}

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
            unit_task = {'task_name':'grounding_detection','element':[],'use_unit':True}
            unit= ['box']
            seq_name = 'boxes_seq'
            place_holder = BOXES_PLACEHOLDER
        elif task == 'segmentation':
            unit_task = {'task_name':'grounding_segmentation','element':[],'use_unit':True}
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
                    conversation_gpt = {'from': 'gpt', 'value': place_holder*len(seq), seq_name: [seq]}
                else:
                    seq = [detail['id']]
                    conversation_gpt = {'from': 'gpt', 'value': place_holder, seq_name: [seq]}

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
                    question = question.replace(MASK_PLACEHOLDER,MASKS_PLACEHOLDER)
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
    
    def mix(self,ret,objects,floating_objects,captions,random_select=False,length=None):
        '''mix all tasks'''
        if self.use_floating_objects:
            objects = objects + floating_objects
        
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
 
        for i,object in enumerate(objects):
            
            det_dict['bboxes'].append(object['bbox'])
            seg_dict['masks'].append(decode(object['segmentation']))

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
                conversation_gpt_cond_det = {'from': 'gpt', 'value': det_dict['place_holder'], det_dict['seq_name']: [seq_cond_det]}
                single_conversation_cond_det = [conversation_human_cond_det,conversation_gpt_cond_det]
                det_dict['conversations']['ground_conversations'].append(single_conversation_cond_det)
                question_cond_seg = self.get_template_from_dict('Cond_SEG')
                question_cond_seg = question_cond_seg.replace(CLASS_PLACEHOLDER,cls_name_cond)
                conversation_human_cond_seg = {'from': 'human','value': question_cond_seg}
                conversation_gpt_cond_seg = {'from': 'gpt', 'value': seg_dict['place_holder'], seg_dict['seq_name']: [seq_cond_seg]}
                single_conversation_cond_seg = [conversation_human_cond_seg,conversation_gpt_cond_seg]
                seg_dict['conversations']['ground_conversations'].append(single_conversation_cond_seg)

                # generate detection & segmentation captions
                det_dict['box_caption'] = det_dict['box_caption'] + PHRASE_ST_PLACEHOLDER_STAGE2 + cls_name + PHRASE_ED_PLACEHOLDER_STAGE2 + det_dict['place_holder'] + ', '
                seg_dict['mask_caption'] = seg_dict['mask_caption'] + PHRASE_ST_PLACEHOLDER_STAGE2 + cls_name + PHRASE_ED_PLACEHOLDER_STAGE2 + seg_dict['place_holder'] + ', '
        
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

            for detail in caption['details']:
                phrase = detail['phrase']
                if 'ids' in detail.keys():
                    rec_seq = detail['ids']
                    reg_seq = detail['ids']
                    # generate rec detection & segmentation answers
                    conversation_gpt_rec_det = {'from': 'gpt', 'value': det_dict['place_holder']*len(seq), 
                                                det_dict['seq_name']: [rec_seq]}
                    conversation_gpt_rec_seg = {'from': 'gpt', 'value': seg_dict['place_holder']*len(seq), 
                                                seg_dict['seq_name']: [rec_seq]}
                else:
                    rec_seq = [detail['id']]
                    reg_seq = detail['id']
                    # generate rec detection & segmentation answers
                    conversation_gpt_rec_det = {'from': 'gpt', 'value': det_dict['place_holder'], 
                                                det_dict['seq_name']: [rec_seq]}
                    conversation_gpt_rec_seg = {'from': 'gpt', 'value': seg_dict['place_holder']*len(seq), 
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
                question_reg_seg = question_reg_seg.replace(MASK_PLACEHOLDER,MASKS_PLACEHOLDER)
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
                    seq_cap_det_seg.append([reg_seq])

                caption_expr_det = caption_expr_det.lower().replace(phrase,PHRASE_ST_PLACEHOLDER_STAGE2 + phrase + PHRASE_ED_PLACEHOLDER_STAGE2 + place_holders_det)  
                caption_expr_seg = caption_expr_seg.lower().replace(phrase,PHRASE_ST_PLACEHOLDER_STAGE2 + phrase + PHRASE_ED_PLACEHOLDER_STAGE2 + place_holders_seg)
                

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
        
        # generate system values
        system_value_det = {key:None for key in list(det_dict['conversations'].keys())}
        system_value_seg = {key:None for key in list(seg_dict['conversations'].keys())}
        system_value_det['conversations_det'] = [{'task':{'task_name':'detection','element':['phrase'],'use_unit':True},'unit':['box']}]
        system_value_det['ground_conversations'] = [{'task':{'task_name':'grounding_detection','element':[],'use_unit':True},'unit':['box']} for _ in range(len(det_dict['conversations']['ground_conversations']))]
        system_value_det['rec_conversations'] = [{'task':{'task_name':'grounding_detection','element':[],'use_unit':True},'unit':['box']} for _ in range(len(det_dict['conversations']['rec_conversations']))]
        system_value_det['reg_conversations'] = [{'task':{'task_name':'vqa','element':['sentence'],'use_unit':False}} for _ in range(len(det_dict['conversations']['reg_conversations']))]
        system_value_det['cap_det_conversations'] = [{'task':{'task_name':'gcg_detection','element':['phrase','sentence'],'use_unit':True},'unit':['box']} for _ in range(len(det_dict['conversations']['cap_det_conversations']))]

        system_value_seg['conversations_seg'] = [{'task':{'task_name':'segmentation','element':['phrase'],'use_unit':True},'unit':['mask']}]
        system_value_seg['ground_conversations'] = [{'task':{'task_name':'grounding_segmentation','element':[],'use_unit':True},'unit':['mask']} for _ in range(len(seg_dict['conversations']['ground_conversations']))]
        system_value_seg['rec_conversations'] = [{'task':{'task_name':'grounding_segmentation','element':[],'use_unit':True},'unit':['mask']} for _ in range(len(seg_dict['conversations']['rec_conversations']))]
        system_value_seg['reg_conversations'] = [{'task':{'task_name':'vqa','element':['sentence'],'use_unit':False}} for _ in range(len(seg_dict['conversations']['reg_conversations']))]
        system_value_seg['cap_seg_conversations'] = [{'task':{'task_name':'gcg_segmentation','element':['phrase','sentence'],'use_unit':True},'unit':['mask']} for _ in range(len(seg_dict['conversations']['cap_seg_conversations']))]

        system_value_cap = [{'task':{'task_name':'vqa','element':['sentence'],'use_unit':False}} for _ in range(len(cap_conversations))]

        det_dict['conversations']['ground_conversations'] = self.concat_conversations(det_dict['conversations']['ground_conversations'])
        seg_dict['conversations']['ground_conversations'] = self.concat_conversations(seg_dict['conversations']['ground_conversations'])
        det_dict['conversations']['rec_conversations'] = self.concat_conversations(det_dict['conversations']['rec_conversations'])
        seg_dict['conversations']['rec_conversations'] = self.concat_conversations(seg_dict['conversations']['rec_conversations'])
        det_dict['conversations']['reg_conversations'] = self.concat_conversations(det_dict['conversations']['reg_conversations'])
        seg_dict['conversations']['reg_conversations'] = self.concat_conversations(seg_dict['conversations']['reg_conversations'])
        det_dict['conversations']['cap_det_conversations'] = self.concat_conversations(det_dict['conversations']['cap_det_conversations'])
        seg_dict['conversations']['cap_seg_conversations'] = self.concat_conversations(seg_dict['conversations']['cap_seg_conversations'])
        cap_conversations = self.concat_conversations(cap_conversations)


        # concat all conversations
        all_conversations = []
        all_system_values = []
        for conversation_name in det_dict['conversations'].keys():
            all_conversations.append(det_dict['conversations'][conversation_name])
            all_system_values.append(system_value_det[conversation_name])
        for conversation_name in seg_dict['conversations'].keys():
            all_conversations.append(seg_dict['conversations'][conversation_name])
            all_system_values.append(system_value_seg[conversation_name])
        all_conversations.append(cap_conversations)
        all_system_values.append(system_value_cap)

        if random_select:
            all_conversations,all_system_values = self.random_select(all_conversations,system_value=all_system_values)
        all_system_values = self.concat_conversations(all_system_values)
        all_conversations = self.concat_conversations(all_conversations,concat_all=True)

        ret['target'] = {det_dict['type']:det_dict['bboxes'],seg_dict['type']:seg_dict['masks']}
        ret['conversations'] = [{'from':'system','value':all_system_values}]
        ret['conversations'] += all_conversations

        return ret
    

    def make_conversations(self,ret,annotations):
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
            ret = self.mix(ret,objects,floating_objects,captions,random_select=True,length=self.length)
            return ret
        
    def __len__(self):
        return len(self.text_path_file)

    def __getitem__(self, index):
        text_file = self.text_path_file[index]
        annotations_json = self.get_file_data(os.path.join(self.text_path,text_file))
        img_path = text_file[:-5] + '.jpg'
        annotations = annotations_json[img_path]
        shape = annotations['objects'][0]['segmentation']['size']
        image_path_abs = os.path.join(self.image_folder,img_path)
        ret = {}
        ret['image'] = {'path': image_path_abs,'width':shape[0],'height':shape[1]}

        ret = self.make_conversations(ret,annotations)

        return ret
