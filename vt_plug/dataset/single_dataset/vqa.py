from vt_plug.registry import DATASETS
from base64 import b64decode
from io import BytesIO
from PIL import Image
import numpy as np
from vt_plug.utils.constants import (
    QUESTION_PLACEHOLDER,
    IMAGE_PLACEHOLDER)

from .mixin import MInstrDataset

@DATASETS.register_module()
class VQADataset(MInstrDataset):
    def __init__(self, *args, has_annotation=True, **kwargs):
        super().__init__(*args, **kwargs, placeholders=(IMAGE_PLACEHOLDER, QUESTION_PLACEHOLDER))
        self.has_annotation = has_annotation

    def __getitem__(self, index):
        offline_item = super().__getitem__(index)
        if offline_item is not None:
            return offline_item
        
        item = self.get_raw_item(index)
        if 'base64' in item.keys():
            image = item['base64']
            image = Image.open(BytesIO(b64decode(image)))
            image = {'value': np.array(image)}
        else:
            img_path = item['img_path']
            image = self.get_image(img_path)

        question = item['question']
        final_question = self.get_template().replace(QUESTION_PLACEHOLDER, question)

        final_answer = item['answer']


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
