from pycocotools.mask import decode
import pycocotools.mask as mask_utils
from registry import DATASETS, METRICS
from utils.constants import (
    IMAGE_PLACEHOLDER,
    BOXES_PLACEHOLDER,
    OBJS_PLACEHOLDER
)
from evaluation.metrics import BaseComputeMetrics
from .mixin import MInstrDataset



@DATASETS.register_module()
class REGDataset(MInstrDataset):
    def __init__(self, *args, version='box', **kwargs):
        super().__init__(*args, **kwargs)
        self.version = version

    def __getitem__(self, index):
        offline_item = super().__getitem__(index)
        if offline_item is not None:
            return offline_item
        
        item = self.get_raw_item(index)
        img_path = item['img_path']
        expr = item['expression']
        image = self.get_image(img_path)
        caption = expr

        if self.version == 'box':
            question = self.get_template().replace(OBJS_PLACEHOLDER, BOXES_PLACEHOLDER)
            bbox = item['bbox']
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
        
        elif self.version == 'mask':
            question = self.get_template()
            mask_rle = item['mask']
            rleObjs = mask_utils.frPyObjects(mask_rle, item["height"], item["width"])
            mask_decode = decode(rleObjs)
            if len(mask_decode.shape) == 3:
                mask = [mask_decode[:,:,0]]
            elif len(mask_decode.shape) == 2:
                mask = [mask_decode]
            else:
                raise NotImplementedError
            ret = {
                'image': image,
                'target': {
                    'masks': mask,
                },
                'conversations': [
                    {
                        'from': 'human',
                        'value': question,
                        'masks_seq': [[0]],
                    },
                    {
                        'from': 'gpt',
                        'value': f'{caption}',
                    }
                ]
            }            

        if self.stage == 2:
            system = {
                        'from':'system',
                        'value': [{'task':{'task_name':'vqa','element':['sentence'],'use_unit':False}}],
                        # 'value': [{'task':{'task_name':'referring vqa','element':['sentence'],'use_unit':False}}],
                    }
            ret['conversations'].insert(0, system)
            ret['map_placeholders'] = self.map_placeholders

        return ret


@DATASETS.register_module()
class GCDataset(REGDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
