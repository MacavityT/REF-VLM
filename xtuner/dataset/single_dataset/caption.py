import json
from xtuner.registry import DATASETS
from xtuner.utils.constants import IMAGE_PLACEHOLDER
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
        img_path = item['img_path']
        caption = item['caption']

        image = self.get_image(img_path)
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
        return ret
    



