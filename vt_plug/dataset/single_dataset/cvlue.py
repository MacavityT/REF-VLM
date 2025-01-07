import json
from vt_plug.utils.constants import (
    QUESTION_PLACEHOLDER,
    IMAGE_PLACEHOLDER)
from vt_plug.utils.constants import (
    PHRASE_ST_PLACEHOLDER_STAGE2,
    PHRASE_ED_PLACEHOLDER_STAGE2,
    IMAGE_PLACEHOLDER,
    BOXES_PLACEHOLDER,
    EXPR_PLACEHOLDER,
)
from vt_plug.registry import DATASETS
from base64 import b64decode
from io import BytesIO
from PIL import Image
import numpy as np
from vt_plug.dataset.utils import convert_bbox
import cv2
from .mixin import MInstrDataset


@DATASETS.register_module()
class CVLUEDialogueDataset(MInstrDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, placeholders=(IMAGE_PLACEHOLDER,))
        self.question = self.text_data['id2question']
        self.answer = self.text_data['id2answer']
        self.data = self.text_data['data']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        offline_item = super().__getitem__(index)
        if offline_item is not None:
            return offline_item
        
        item = self.data[index]
        image = item['image']
        image = self.get_image(image)

        question = self.question[str(item['question'])]
        answer = self.answer[str(item['answer'])]

        ret = {
            'image': image,
            'conversations': [
                {
                    'from': 'human',
                    'value': question,
                },
                {
                    'from': 'gpt',
                    'value': answer,
                }
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



@DATASETS.register_module()
class CVLUEVQADataset(MInstrDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, placeholders=(IMAGE_PLACEHOLDER, QUESTION_PLACEHOLDER))

    def __getitem__(self, index):
        offline_item = super().__getitem__(index)
        if offline_item is not None:
            return offline_item
        
        item = self.text_data[index]
        image = item['image']
        image = self.get_image(image)

        
        question = item['question']
        final_question = self.get_template().replace(QUESTION_PLACEHOLDER, question)
        answer = item['answer']

        ret = {
            'image': image,
            'conversations': [
                {
                    'from': 'human',
                    'value': final_question,
                },
                {
                    'from': 'gpt',
                    'value': answer,
                }
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





@DATASETS.register_module()
class CVLUECaptionDataset(MInstrDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, placeholders=(IMAGE_PLACEHOLDER))

    def __getitem__(self, index):
        offline_item = super().__getitem__(index)
        if offline_item is not None:
            return offline_item
        
        item = self.get_raw_item(index)
        image = item['image']
        image = self.get_image(image)

        question = self.get_template()
        answer = item['caption']

        ret = {
            'image': image,
            'conversations': [
                {
                    'from': 'human',
                    'value': question,
                },
                {
                    'from': 'gpt',
                    'value': answer,
                }
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


@DATASETS.register_module()
class CVLUERECDataset(MInstrDataset):
    def __init__(self, *args, target=False, **kwargs):
        super().__init__(*args, **kwargs, placeholders=(IMAGE_PLACEHOLDER, EXPR_PLACEHOLDER))
        self.map_placeholders = {'output':[BOXES_PLACEHOLDER]}
        self.target = target

    def __getitem__(self, index):
        offline_item = super().__getitem__(index)
        if offline_item is not None:
            return offline_item
        
        item = self.text_data[index]
        img_path = item['image']
        expr = item['text']
        bbox = convert_bbox(item['bbox'])

        image = self.get_image(img_path)
        question = self.get_template().replace(EXPR_PLACEHOLDER, expr)

        if self.stage == 1:
            ret = {
                'image': image,
                'target': {
                    'boxes': [bbox],
                },
                'conversations': [
                    {
                        'from': 'human',
                        'value': question,
                    },
                    {
                        'from': 'gpt',
                        'value': f'Answer: {BOXES_PLACEHOLDER}.',
                        'boxes_seq': [[0]],
                    }
                ]
            }


        if self.stage == 2:
            if self.target:
                value = PHRASE_ST_PLACEHOLDER_STAGE2 + expr + PHRASE_ED_PLACEHOLDER_STAGE2 + BOXES_PLACEHOLDER
            else:
                value = PHRASE_ST_PLACEHOLDER_STAGE2 + 'target' + PHRASE_ED_PLACEHOLDER_STAGE2 + BOXES_PLACEHOLDER
            ret = {
                'image': image,
                'target': {
                    'boxes': [bbox],
                },
                'conversations': [
                    {
                        'from':'system',
                        'value':[{'task':{'task_name':'grounding_detection','element':['phrase'],'use_unit':True},'unit':['box']}],
                    },
                    {
                        'from': 'human',
                        'value': question,
                    },
                    {
                        'from': 'gpt',
                        'value': f'{value}.',
                        'boxes_seq': [[0]],
                    }
                ]
            }
            ret['map_placeholders'] = self.map_placeholders
        return ret

