import os
import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from PIL import Image

class CoCoDataset(Dataset):
    def __init__(self, root, annotation_file):
        """
        Args:
            root (string): Root directory where images are downloaded to.
            annotation_file (string): Path to the COCO annotation file.
            transform (callable, optional): A function/transform that takes in an image and returns a transformed version.
            target_transform (callable, optional): A function/transform that takes in the target and transforms it.
            use_custom_bboxes (bool): Flag to use custom bounding boxes instead of COCO ones.
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

        # Example
        for target in target:
            area = target['area']
            bbox = target['bbox']
            category_id = target['category_id']
            image_id = target['image_id']
            segmentation = target['segmentation']

        return img, target

    def __len__(self):
        return len(self.ids)

if __name__ == '__main__':

    dataset = CoCoDataset(root='/data/Aaronzhu/DatasetStage1/MSCOCO/2017/train2017',
                         annotation_file='/data/Aaronzhu/DatasetStage1/COCO-ReM/instances_trainrem.json')
    data = dataset[0]
    print(data)
