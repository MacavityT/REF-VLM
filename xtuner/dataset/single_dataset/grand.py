import os
import json
import torch
from torch.utils.data import Dataset
from pycocotools.mask import decode

class GranDDataset(Dataset):

    def __init__(self, folder):
        super().__init__()
        assert folder is not None
        self.folder = folder
        self.granD = os.listdir(folder)
        
    def read_json(self, path):
        with open(path, 'r') as f:
            datas = json.loads(f.read())
        return datas

    def mask_decode(self, l, obj):
        for i in range(l):
            mask = decode(obj[i]['segmentation'])
            mask = torch.tensor(mask, dtype = torch.uint8)
            obj[i]['segmentation'] = mask
        return obj

    def __getitem__(self, index):
        
        self.index = index
        path_ads = os.path.join(self.folder, self.granD[index])
        item = self.read_json(path_ads)
        id = self.granD[index].split('.')[0]+'.jpg'
        obj = item[id]['objects']
        float_obj = item[id]['floating_objects']
        item[id]['objects'] = self.mask_decode(len(obj), obj)
        item[id]['floating_objects'] = self.mask_decode(len(float_obj), float_obj)
        return item[id]
        '''
            annotation's key
            objects = item[id]['objects']
            floating_objects = item[id]['floating_objects']
            floating_attributes = item[id]['floating_attributes']
            landmark = item[id]['landmark']
            affordances = item[id]['affordances']
            id_counter = item[id]['id_counter']
            relationships = item[id]['relationships']
            short_captions = item[id]['short_captions']
            dense_caption = item[id]['dense_caption']
            additional_context = item[id]['additional_context']
        '''


    def __len__(self):
        return len(self.granD)

if __name__ == '__main__':

    annotations = '/data/Aaronzhu/GranD/GLaMM_data/GranD/part_1'
    GranD = GranDDataset(folder = annotations)
    grand = GranD[0]
    print(grand)