from ref_vlm.registry import DATASETS
from ref_vlm.utils.constants import (
    QUESTION_PLACEHOLDER,
    IMAGE_PLACEHOLDER,
)
from .mixin import MInstrDataset


@DATASETS.register_module()
class POPEVQADataset(MInstrDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, placeholders=(IMAGE_PLACEHOLDER, QUESTION_PLACEHOLDER))

    def __getitem__(self, index):
        offline_item = super().__getitem__(index)
        if offline_item is not None:
            return offline_item
        
        item = self.get_raw_item(index)
        image = self.get_image(image_path=item['image'])

        question = item['text']
        final_question = self.get_template().replace(QUESTION_PLACEHOLDER, question)

        label = str(item['label']).lower()

        ret = {
            'image': image,
            'conversations': [
                {
                    'from': 'human',
                    'value': final_question,
                },
                {
                    'from': 'gpt',
                    'value': f"The answer is {label} .",
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
