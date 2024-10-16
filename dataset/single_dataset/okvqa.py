from registry import DATASETS
import json
from utils.constants import (
    QUESTION_PLACEHOLDER,
    IMAGE_PLACEHOLDER)

from .mixin import MInstrDataset

@DATASETS.register_module()
class OKVQADataset(MInstrDataset):
    def __init__(self, *args, has_annotation=True, **kwargs):
        super().__init__(*args, **kwargs, placeholders=(IMAGE_PLACEHOLDER, QUESTION_PLACEHOLDER))
        self.has_annotation = has_annotation

    def __getitem__(self, index):
        offline_item = super().__getitem__(index)
        if offline_item is not None:
            return offline_item
        
        item = self.get_raw_item(index)
        image = self.get_image(image_path=item['image_name'])

        question = item['question']
        final_question = self.get_template().replace(QUESTION_PLACEHOLDER, question)

        if self.has_annotation:
            final_answer = item['answers'][0]
            ret = {
                'image': image,
                'conversations': [
                    {
                        'from': 'human',
                        'value': final_question,
                    },
                    {
                        'from': 'gpt',
                        'value': f"The answer is {final_answer}.",
                    },
                ]
            }
        else:
            final_answer = item['answers']
            ret = {
                'image': image,
                'conversations': [
                    {
                        'from': 'human',
                        'value': final_question,
                    },
                    {
                        'from': 'gpt',
                        'value': f"{final_answer}",
                    },
                ]
            }

        if self.stage == 2:
            system = {
                        'from':'system',
                        'value': [{'task':{'task_name':'vqa','element':['sentence'],'use_unit':False}}],
                    }
            ret['conversations'].insert(0, system)        
            ret['map_placeholders'] = self.map_placeholders
        return ret
