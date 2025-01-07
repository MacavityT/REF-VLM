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
class CaptionDataset(MInstrDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, placeholders=(IMAGE_PLACEHOLDER,))

    def __getitem__(self, index):
        offline_item = super().__getitem__(index)
        if offline_item is not None:
            return offline_item
        
        item = self.get_raw_item(index)

        if 'image_str' in item.keys() or 'image_base64' in item.keys():
            try:
                image = item['image_str']
            except:
                image = item['image_base64']
            image = Image.open(BytesIO(b64decode(image)))
            image = {'value': np.array(image)}


        else:
            img_path = item['img_path']
            image = self.get_image(img_path)


        caption = item['caption']
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




        