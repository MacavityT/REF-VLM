from xtuner.registry import DATASETS
from xtuner.utils.constants import (
    QUESTION_PLACEHOLDER,
    IMAGE_PLACEHOLDER,
    OBJS_PLACEHOLDER)
from .mixin import MInstrDataset
from pycocotools.coco import COCO


# https://github.com/open-mmlab/mmpretrain/blob/17a886cb5825cd8c26df4e65f7112d404b99fe12/mmpretrain/datasets/refcoco.py#L14

@DATASETS.register_module()
class RefCocoDataset(MInstrDataset):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, placeholders=(IMAGE_PLACEHOLDER, OBJS_PLACEHOLDER), **kwargs)
        self.coco = COCO(self.text_path)
        self.ids = list(sorted(self.coco.imgs.keys()))

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)

        #load annotations
        targets = coco.loadAnns(ann_ids) 
        segmentation = []
        area = []
        iscrowd = []
        image_id = []
        bbox = []
        category = []
        id = []
        for target in targets: 
            segmentation.append(target['segmentation'])
            area.append(target['area'])
            iscrowd.append(target['iscrowd'])
            image_id.append(target['image_id'])
            bbox.append(target['bbox'])
            category.append(target['category_id'])
            id.append(target['id']) #box_id

        img_path = coco.loadImgs(img_id)[0]['file_name']
        image = self.get_image(img_path)
        question = self.get_template().replace(OBJS_PLACEHOLDER, OBJS_PLACEHOLDER) 
        
        ret = {
            'image': image,
            'target': {
                # 'segmentation': [[segmentation]],
                # 'area': area,
                # 'iscrowd': iscrowd,
                # 'image_id': image_id,
                'bbox': [bbox],
                # 'category_id': category,
                # 'id': id
            },
            'conversations': [
                {
                    'from': 'human',
                    'value': question,
                    'boxes_seq': id,
                },
                {
                    'from': 'gpt',
                    'value': f"The answer is <boxes>.",
                    'boxes_seq': id,
                }
            ]
        }
        return ret

     

