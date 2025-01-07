from vt_plug.registry import DATASETS
from vt_plug.utils.constants import IMAGE_PLACEHOLDER
import os
from .mixin import MInstrDataset



@DATASETS.register_module()
class WuKongDataset(MInstrDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, placeholders=(IMAGE_PLACEHOLDER,))

    def __getitem__(self, index):
        offline_item = super().__getitem__(index)
        if offline_item is not None:
            return offline_item
        item = self.get_raw_item(index)
        image_name = item['image_name']
        image = self.get_image(image_name)
        
        caption = item['title']
        question = self.get_template()
        conversations = [
            {'from': 'human','value': question},
            {'from': 'gpt','value': caption}
        ]
        assert len(conversations) % 2 == 0, "Conversations are incomplete!"

 
        ret = {
            'image': image,
            'conversations': conversations,
        }

        if self.stage == 2:
            system = {
                        'from':'system',
                        'value': [{'task':{'task_name':'vqa','element':['sentence'],'use_unit':False}} for _ in range(len(conversations)//2)],
                    }
            ret['conversations'].insert(0, system)
            ret['map_placeholders'] = self.map_placeholders
            
        return ret
