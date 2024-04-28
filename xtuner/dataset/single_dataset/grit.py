import os
import json
import torch
from torch.utils.data import Dataset
from pycocotools.mask import decode

class GritDataset(Dataset):

    def __init__(self, folder, root):
        super().__init__()
        assert folder is not None
        self.folder = folder
        self.frit = os.listdir(folder)
        self.img = os.listdir(root)
        
    def read_json(self, path):
        with open(path, 'r') as f:
            datas = json.loads(f.read())
        return datas

    def __getitem__(self, index):
        index = str(index).zfill(9)
        path_ads = os.path.join(self.folder, f"{index}.json")
        annotations = self.read_json(path_ads)
        id = index+'.jpg'
        noun_chunks = annotations['noun_chunks']
        caption =  annotations['caption']
        ref_exps = annotations["ref_exps"]
        clip_similarity_vitb32 = annotations["clip_similarity_vitb32"]
        clip_similarity_vitl14 = annotations["clip_similarity_vitl14"]
        width = annotations["width"]
        height = annotations["height"]
        original_width = annotations["original_width"]
        original_height = annotations["original_height"]
        item = {
            "img_id": id,
            "caption":caption,
            "noun_chunks":noun_chunks,
            "ref_exps":ref_exps,
            "clip_similarity_vitb32":clip_similarity_vitb32,
            "clip_similarity_vitl14":clip_similarity_vitl14,
            "width":width,
            "height":height,
            "original_width":original_width,
            "original_height":original_height
        }

        return item

    def __len__(self):
        return len(self.grit)

if __name__ == '__main__':

    annotations = '/data/Aaronzhu/DatasetStage2and3/GRIT/annotations'
    img = '/data/Aaronzhu/DatasetStage2and3/GRIT/img'

    Grit = GritDataset(folder = annotations,root = img)
    grit = Grit[0]