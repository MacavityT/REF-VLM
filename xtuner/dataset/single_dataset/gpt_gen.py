from xtuner.registry import DATASETS
from xtuner.utils.constants import (
    QUESTION_PLACEHOLDER,
    IMAGE_PLACEHOLDER,
    BOXES_PLACEHOLDER,
    PHRASE_ST_PLACEHOLDER, 
    PHRASE_ED_PLACEHOLDER,
    PHRASE_ST_PLACEHOLDER_STAGE2,
    PHRASE_ED_PLACEHOLDER_STAGE2)

from .mixin import MInstrDataset

@DATASETS.register_module()
class GPT4Gen(MInstrDataset):
    def __init__(self, *args, version, **kwargs):
        super().__init__(*args, **kwargs, placeholders=(IMAGE_PLACEHOLDER, QUESTION_PLACEHOLDER))
        self.version = version
        assert version in ['a', 'c', 'bc']

    def __getitem__(self, index):
        offline_item = super().__getitem__(index)
        if offline_item is not None:
            return offline_item
                
        raw = self.get_raw_item(index)
        #
        image = self.get_image(raw['img_path'])
        #
        boxes = raw['boxes']
        #
        question = raw['question']
        if self.stage == 1:
            question = question.replace(PHRASE_ST_PLACEHOLDER, '').replace(PHRASE_ED_PLACEHOLDER, BOXES_PLACEHOLDER)
        if self.stage == 2:
            question = question.replace(PHRASE_ED_PLACEHOLDER,f"{PHRASE_ED_PLACEHOLDER}{BOXES_PLACEHOLDER}")
            question = question.replace(PHRASE_ST_PLACEHOLDER,PHRASE_ST_PLACEHOLDER_STAGE2).replace(PHRASE_ED_PLACEHOLDER,PHRASE_ED_PLACEHOLDER_STAGE2)
            boxes_list = [BOXES_PLACEHOLDER * len(box_seq) for box_seq in raw['question_boxes_seq']]
            question = question.replace(BOXES_PLACEHOLDER,'{}').format(*boxes_list)
        final_question = self.get_template().replace(QUESTION_PLACEHOLDER, question)
        query_boxes_seq = raw['question_boxes_seq']

        if self.version == 'a':
            final_answer = raw['answer']
            answer_boxes_seq = None
            values = [{'task':{'task_name':'vqa','element':['sentence'],'use_unit':False}}]
        elif self.version == 'c':
            final_answer = raw['cot_with_ans'].replace(PHRASE_ST_PLACEHOLDER, '').replace(PHRASE_ED_PLACEHOLDER, '')
            answer_boxes_seq = None
            values = [{'task':{'task_name':'vqa','element':['sentence'],'use_unit':False}}]
        elif self.version == 'bc':
            if self.stage == 1:
                final_answer = raw['cot_with_ans'].replace(PHRASE_ST_PLACEHOLDER, '').replace(PHRASE_ED_PLACEHOLDER, BOXES_PLACEHOLDER)
            if self.stage == 2:
                final_answer = raw['cot_with_ans'].replace(PHRASE_ED_PLACEHOLDER,f"{PHRASE_ED_PLACEHOLDER}{BOXES_PLACEHOLDER}")    
                final_answer = final_answer.replace(PHRASE_ST_PLACEHOLDER,PHRASE_ST_PLACEHOLDER_STAGE2).replace(PHRASE_ED_PLACEHOLDER,PHRASE_ED_PLACEHOLDER_STAGE2)
                boxes_list = [BOXES_PLACEHOLDER * len(box_seq) for box_seq in raw['answer_boxes_seq']]
                final_answer = final_answer.replace(BOXES_PLACEHOLDER,'{}').format(*boxes_list)
            answer_boxes_seq = raw['answer_boxes_seq']
            if answer_boxes_seq == []:
                values = [{'task':{'task_name':'vqa','element':['sentence'],'use_unit':False}}]
            else:
                values = [{'task':{'task_name':'gcg_detection','element':['phrase','sentence'],'use_unit':True},'unit':['box']}]
           
        else:
            assert False

        if self.stage == 1:
            ret = {
                'image': image,
                'target': {'boxes': boxes},
                'conversations': [
                    {
                        'from': 'human',
                        'value': final_question,
                        'boxes_seq': query_boxes_seq,
                    },
                    {
                        'from': 'gpt',
                        'value': final_answer,
                        'boxes_seq': answer_boxes_seq,
                    }
                ]
            }
        if self.stage == 2:
            ret = {
                'image': image,
                'target': {'boxes': boxes},
                'conversations': [
                    {
                        'from':'system',
                        'value':values,
                    },
                    {
                        'from': 'human',
                        'value': final_question,
                        'boxes_seq': query_boxes_seq,
                    },
                    {
                        'from': 'gpt',
                        'value': final_answer,
                        'boxes_seq': answer_boxes_seq,
                    }
                ]
            }
            if ret['conversations'][1]['boxes_seq'] == [] or ret['conversations'][1]['boxes_seq'] is None:
                del ret['conversations'][1]['boxes_seq']
            if ret['conversations'][2]['boxes_seq'] == [] or ret['conversations'][2]['boxes_seq'] is None:
                del ret['conversations'][2]['boxes_seq']

            ret['map_placeholders'] = self.map_placeholders
        return ret
