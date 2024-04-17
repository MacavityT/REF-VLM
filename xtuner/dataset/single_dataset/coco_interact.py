import os
import torch
import json
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from PIL import Image

class CoCoInteractDataset(Dataset):
    def __init__(self, root, annotation_file):

        self.root = root
        self.annotation = self.load_data(annotation_file) # self.coco = COCO(annotation_file)无法使用

    def load_data(self, file_path):
        """
        Load annotations from the specified JSON file.
        """
        with open(file_path, 'r') as file:
            data = json.load(file)
        return data

    def __getitem__(self, index):

        img_name = os.path.join(self.root, self.annotation[index]['image'])
        image = Image.open(img_name)
        annotation = self.annotation[index]
        return image, annotation

    def __len__(self):
        return len(self.ids)

if __name__ == '__main__':

    image_path = '/data/Aaronzhu/DatasetStage1/MSCOCO/2017/train2017'
    annotation_file = '/data/Aaronzhu/DatasetStage2and3/COCO_interactive/coco_interactive_train_psalm.json'


    dataset = CoCoInteractDataset(root=image_path, annotation_file=annotation_file)
    image, annotation = dataset[0]
    print(image, annotation)
