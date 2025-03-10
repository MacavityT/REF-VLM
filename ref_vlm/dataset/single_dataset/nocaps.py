import json
from ref_vlm.utils.constants import IMAGE_PLACEHOLDER
from ref_vlm.registry import DATASETS
from .mixin import MInstrDataset


@DATASETS.register_module()
class NoCapsDataset(MInstrDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, placeholders=(IMAGE_PLACEHOLDER,))
        self.annotations = self.read_json()
        self.imgs = {}
        if 'images' in self.annotations:
            for img in self.annotations['images']:
                self.imgs[img['id']] = img

    def read_json(self):
        with open(self.text_path) as f:
            img_json = json.loads(f.read())
        return img_json
    
    def __len__(self):
        return len(self.annotations['annotations'])

    def __getitem__(self, index):
        
        item = self.annotations['annotations'][index]
        image_info = self.imgs[item['image_id']]
        img_path = image_info['file_name']
        image_id = image_info['id']
        caption = item['caption']

        image = self.get_image(img_path)
        question = self.get_template()

        ret = {
            'image': image,
            'image_id': image_id,
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
    



