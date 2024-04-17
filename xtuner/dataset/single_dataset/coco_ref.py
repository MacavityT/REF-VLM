import os
import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from PIL import Image
from xtuner.registry import DATASETS
from .mixin import MInstrDataset
from xtuner.utils.constants import (
    QUESTION_PLACEHOLDER,
    IMAGE_PLACEHOLDER,
    BOXES_PLACEHOLDER,
    POINTS_PLACEHOLDER,
)

'''
item = {
    'image': {
        'path': '/path/to/image', # str
        'width': 512, # int
        'height': 512, # int 
    },
    'target': {
        # xmin, ymin, xmax, ymax
        'boxes': [
            [10, 10, 256, 265],  # dog1
            [24, 18, 378, 768],  # dog2
            [100, 310, 670, 653],  # man
            [278, 320, 809, 673],  # rope
        ],
        'masks':[
        
        ]
    },
    "conversations": [
        {
            'from': 'human',
            'value': 'What is the relation between the two dogs <boxes> and the man <boxes> in the image <image> ?',
            'boxes_seq': [[0, 1], [2], ],
        },
        {
            'from': 'gpt',
            'value': 'a rope <boxes> is connecting the left dog <boxes> with the man <boxes>. '
                        'So the man <boxes> is walking the dog <boxes>.'
                    'And the man <boxes> has no relationship with the right dog <boxes>',
            'boxes_seq': [[3], [0], [2], [2], [0], [2], [1]],
        }
    ]
}

'''

@DATASETS.register_module()
class CoCoRefDataset(MInstrDataset):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, placeholders=(IMAGE_PLACEHOLDER,))

    def __len__(self):
        return len(self.text_data)
    
    def __getitem(self,index):
        item = self.get_raw_item(index)
    
    def get_raw_item(self):
        pass


class CoCoRefDataset(Dataset):


    def __init__(self, root, annotation_file):
        """
        Args:
            root (string): Root directory where images are downloaded to.
            annotation_file (string): Path to the COCO annotation file.
        """
        self.root = root
        self.coco = COCO(annotation_file)
        self.ids = list(sorted(self.coco.imgs.keys()))

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)

        path = coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        return img, target

    def __len__(self):
        return len(self.ids)

if __name__ == '__main__':

    image_path = '/data/Aaronzhu/DatasetStage1/MSCOCO/2014/train'
    annotation_file = '/data/Aaronzhu/DatasetStage1/Refcoco/refcoco/instances.json'


    dataset = CoCoRefDataset(root=image_path, annotation_file=annotation_file)
    image, annotation = dataset[0]
    print(image, annotation)
