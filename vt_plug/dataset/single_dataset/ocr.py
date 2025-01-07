import json
from vt_plug.utils.constants import IMAGE_PLACEHOLDER
from vt_plug.registry import DATASETS
from base64 import b64decode
from io import BytesIO
from PIL import Image
import numpy as np
import cv2
from .mixin import MInstrDataset


@DATASETS.register_module()
class OCRCNDataset(MInstrDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, placeholders=(IMAGE_PLACEHOLDER,))

    def __getitem__(self, index):
        offline_item = super().__getitem__(index)
        if offline_item is not None:
            return offline_item
        
        item = self.text_data.loc[index]
        image_bytes = item['image']['bytes']
        image = np.asarray(bytearray(image_bytes),dtype='uint8')
        image = cv2.imdecode(image,cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        image = {'value': np.array(image)}

        if 'text' in item.keys():
            caption = item['text']
        elif 'ground_truth' in item.keys():
            json_string = item['ground_truth']
            parsed_data = json.loads(json_string)
            caption = parsed_data['gt_parse']['text_sequence']
            
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

        if self.stage == 2:
            system = {
                        'from':'system',
                        'value': [{'task':{'task_name':'vqa','element':['sentence'],'use_unit':False}}],
                    }
            ret['conversations'].insert(0, system)
            ret['map_placeholders'] = self.map_placeholders
        return ret




        