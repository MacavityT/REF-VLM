from xtuner.registry import DATASETS
from .mixin import MInstrDataset


@DATASETS.register_module()
class InstructDataset(MInstrDataset):
    def __init__(self, *args, add_coco_prefix=False, **kwargs):
        super().__init__(*args, **kwargs, placeholders=(), template_string='', template_file=None)
        self.add_coco_prefix = add_coco_prefix

    def __getitem__(self, index):
        offline_item = super().__getitem__(index)
        if offline_item is not None:
            return offline_item
        
        item = self.get_raw_item(index)
        if self.add_coco_prefix:
            img_path = f"COCO_train2014_{item['image']}"
        else:
            img_path = item['image']
        conversations = item['conversations']

        image = self.get_image(img_path)
        ret = {
            'image': image,
            'conversations': conversations,
        }
        return ret

@DATASETS.register_module()
class InstructMixDataset(MInstrDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def __getitem__(self, index):
        offline_item = super().__getitem__(index)
        if offline_item is not None:
            return offline_item
        
        item = self.text_data[index]

        conversations = item['conversations']
        assert len(conversations) % 2 == 0, "Conversations are incomplete!"

        if 'image' in item.keys():
            img_path = item['image']
            image = self.get_image(img_path)            
            ret = {
                'image': image,
                'conversations': conversations,
            }
        else:
            ret = {
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


