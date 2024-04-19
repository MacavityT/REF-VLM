from xtuner.registry import DATASETS, METRICS
from xtuner.utils.constants import (
    IMAGE_PLACEHOLDER,
    BOXES_PLACEHOLDER,
    OBJS_PLACEHOLDER
)
from xtuner.evaluation.metrics import BaseComputeMetrics
from .mixin import MInstrDataset



@DATASETS.register_module()
class REGDataset(MInstrDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, placeholders=(IMAGE_PLACEHOLDER, OBJS_PLACEHOLDER), **kwargs)

    def __getitem__(self, index):
        offline_item = super().__getitem__(index)
        if offline_item is not None:
            return offline_item
        
        item = self.get_raw_item(index)
        img_path = item['img_path']
        expr = item['expression']
        bbox = item['bbox']

        image = self.get_image(img_path)
        question = self.get_template().replace(OBJS_PLACEHOLDER, BOXES_PLACEHOLDER)
        caption = expr

        ret = {
            'image': image,
            'target': {
                'boxes': [bbox],
            },
            'conversations': [
                {
                    'from': 'human',
                    'value': question,
                    'boxes_seq': [[0]],
                },
                {
                    'from': 'gpt',
                    'value': f'{caption}',
                }
            ]
        }
        return ret


@DATASETS.register_module()
class GCDataset(REGDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
