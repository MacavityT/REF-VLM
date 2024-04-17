import os
import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from PIL import Image

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
    annotation_file = '//data/Aaronzhu/DatasetStage1/Refcoco/refcoco/instances.json'


    dataset = CoCoRefDataset(root=image_path, annotation_file=annotation_file)
    image, annotation = dataset[0]
    print(image, annotation)
