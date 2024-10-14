import os
import random
from PIL import Image
import numpy as np
from xtuner.registry import DATASETS
from xtuner.utils.constants import MASKS_PLACEHOLDER, IMAGE_PLACEHOLDER, PHRASE_ST_PLACEHOLDER_STAGE2, PHRASE_ED_PLACEHOLDER_STAGE2
from .mixin import MInstrDataset

def get_categories_with_prompt_eng(path, ignore_invalid):
    with open(path) as f:
        lines = f.readlines()
    return {
        key: value for line in lines 
        if (key := line.strip().split(':')[0]) and 
           (value := line.strip().split(':')[1]) and 
           (not ignore_invalid or (key != '0' and value != "invalid_class_id"))
    }

def get_gt(idx, path, invalid_value):
    img_path = os.path.join(path, f'{idx}.png')
    if not os.path.isfile(img_path):
        img_path = os.path.join(path, f'{idx}.tif')
    img = np.array(Image.open(img_path))
    mask_types = np.unique(img)
    mask_types = mask_types[~np.isin(mask_types, invalid_value)]
    masks = [(img == mask_type).astype(np.uint8) for mask_type in mask_types]
    return masks, list(range(len(masks))), mask_types

def flatten_annotation(gt_dir, cate_path, index, invalid_value, ignore_invalid):
    masks, masks_seq, types = get_gt(index, gt_dir, invalid_value)
    categories = get_categories_with_prompt_eng(cate_path, ignore_invalid)
    labels = [categories.get(str(t)) for t in types]
    return {'masks': masks, 'labels': labels, 'masks_seq':masks_seq}

def mask_num(gt_dir):
    gt_list = os.listdir(gt_dir)
    mask_list = {}
    idx = 0
    for i in range(len(gt_list)):
        path = os.path.join(gt_dir, gt_list[i])
        unique_values = np.unique(np.array(Image.open(path)))
        for j in range(len(unique_values)):
            mask_list[idx] = gt_list[i].split('.')[0] # +'.'+unique_values[j]
            idx += 1
    return mask_list

@DATASETS.register_module()
class PascalDataset(MInstrDataset):
    def __init__(self, gt_info, categories, target_type='semantic', *args, **kwargs):
        super().__init__(*args, **kwargs, placeholders=(IMAGE_PLACEHOLDER,))
        self.gt_info = gt_info
        self.target_type = target_type
        self.text_path = [fn for fn in os.listdir(categories) if fn.endswith('txt')]
        self.cate_path = os.path.join(categories, self.text_path[0])
        self.mask_list = mask_num(self.gt_info)

    def __getitem__(self, index, invalid_value, ignore_invalid):
        offline_item = super().__getitem__(index)
        if offline_item is not None:
            return offline_item
        
        img_name = self.mask_list[index]
        img_path = os.path.join(self.image_folder, f"{img_name}.jpg")
        image = self.get_image(img_path)

        item = flatten_annotation(
            gt_dir=self.gt_info,
            cate_path=self.cate_path,
            index=img_name,
            invalid_value=invalid_value,
            ignore_invalid=ignore_invalid
        )

        conversations = []
        for mask, label, mask_seq in zip(item['masks'], item['labels'], item['masks_seq']):
            question = f"Can you assist me by segmenting {label} in the image <image>?"
            unit_task = {'task_name':'grounding_segmentation','element':['phrase'],'use_unit':True}
            unit= ['mask']
            system = {'from':'system','value':[{'task':unit_task,'unit':unit}]}
            human = {'from': 'human', 'value': question}
            answer = {'from': 'gpt', 'value':PHRASE_ST_PLACEHOLDER_STAGE2 + label + PHRASE_ED_PLACEHOLDER_STAGE2 + MASKS_PLACEHOLDER, 'masks_seq':[[mask_seq]]}
            conversation = [system, human, answer]
            ret = {'image': image, 'target': {'masks': mask}, 'conversations': conversation}
            conversations.append(ret)
        ret = conversations[index]
        return ret
    
    def __len__(self):
        return len(self.mask_list)
    
@DATASETS.register_module()
class PascalVoc59Dataset(PascalDataset):
    def __init__(self, gt_info, target_type='semantic', *args, **kwargs):
        categories = '/data/Aaronzhu/DatasetStage2and3/pascal_ctx_d2/annotations_ctx59'
        super().__init__(gt_info, categories, target_type, *args, **kwargs)

    def __getitem__(self, index):
        return super().__getitem__(index, invalid_value=[0, 255], ignore_invalid=True)
    
    def __len__(self):
        return super().__len__()
    
@DATASETS.register_module()
class PascalVoc459Dataset(PascalDataset):
    def __init__(self, gt_info, target_type='semantic', *args, **kwargs):
        categories = '/data/Aaronzhu/DatasetStage2and3/pascal_ctx_d2/annotations_ctx459'
        super().__init__(gt_info, categories, target_type, *args, **kwargs)

    def __getitem__(self, index):
        return super().__getitem__(index, invalid_value=[0], ignore_invalid=True)
    
    def __len__(self):
        return super().__len__()
    
@DATASETS.register_module()
class PascalVocDataset(PascalDataset):
    def __init__(self, gt_info, target_type='semantic', *args, **kwargs):
        categories = '/data/Aaronzhu/DatasetStage2and3/pascal_voc_d2/annotations_pascal21'
        super().__init__(gt_info, categories, target_type, *args, **kwargs)

    def __getitem__(self, index):
        return super().__getitem__(index, invalid_value=[255], ignore_invalid=False)
    
    def __len__(self):
        return super().__len__()