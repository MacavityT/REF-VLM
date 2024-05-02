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
            caption = caption.replace(PHRASE_ST_PLACEHOLDER,PHRASE_ST_PLACEHOLDER_STAGE2).replace(PHRASE_ED_PLACEHOLDER,PHRASE_ED_PLACEHOLDER_STAGE2)
        final_question = self.get_template().replace(QUESTION_PLACEHOLDER, question)
        query_boxes_seq = raw['question_boxes_seq']

        if self.version == 'a':
            final_answer = raw['answer']
            answer_boxes_seq = None
        elif self.version == 'c':
            final_answer = raw['cot_with_ans'].replace(PHRASE_ST_PLACEHOLDER, '').replace(PHRASE_ED_PLACEHOLDER, '')
            answer_boxes_seq = None
        elif self.version == 'bc':
            if self.stage == 1:
                final_answer = raw['cot_with_ans'].replace(PHRASE_ST_PLACEHOLDER, '').replace(PHRASE_ED_PLACEHOLDER, BOXES_PLACEHOLDER)
            if self.stage == 2:
                final_answer = raw['cot_with_ans'].replace(PHRASE_ED_PLACEHOLDER,f"{PHRASE_ED_PLACEHOLDER}{BOXES_PLACEHOLDER}")
                caption = caption.replace(PHRASE_ST_PLACEHOLDER,PHRASE_ST_PLACEHOLDER_STAGE2).replace(PHRASE_ED_PLACEHOLDER,PHRASE_ED_PLACEHOLDER_STAGE2)
            answer_boxes_seq = raw['answer_boxes_seq']
        else:
            assert False

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
        return ret
