from vt_plug.registry import DATASETS
from vt_plug.utils.constants import IMAGE_PLACEHOLDER
import os
from .mixin import MInstrDataset



@DATASETS.register_module()
class Product1MDataset(MInstrDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, placeholders=(IMAGE_PLACEHOLDER,))
        self.all_images = os.listdir(self.image_folder)

    def __getitem__(self, index):
        offline_item = super().__getitem__(index)
        if offline_item is not None:
            return offline_item
        image_name = self.all_images[index]
        image = self.get_image(image_name)
        image_key = image_name[:-4]
        item = self.text_data[image_key]

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
