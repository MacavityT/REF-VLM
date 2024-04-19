# Copyright (c) OpenMMLab. All rights reserved.
import os
import json
import jsonlines
import numpy as np

from mmengine import print_log
from torch.utils.data import Dataset
from xtuner.registry import DATASETS

@DATASETS.register_module()
class OfflineDataset(Dataset):
    def  __init__(self, folder, format):
        super().__init__()
        assert folder is not None
        assert format in ['json', 'jsonl', 'npy']
        self.folder = folder
        self.format = format
        self.data = os.listdir(folder)

    def __len__(self):
        return len(self.data)

    def read_json(self, json_path):
        with open(json_path, 'r') as f:
            datas = json.loads(f.read())
        return datas

    def read_jsonl(self, jsonl_path):
        with jsonlines.open(jsonl_path, 'r') as f:
            datas = [data for data in f]
        return datas
    
    def read_npy(self, npy_path):
        pass

    def __getitem__(self, index):
        path_abs = os.path.join(self.folder, self.data[index])
        if self.format == 'json':
            item = self.read_json(path_abs)
        elif self.format == 'jsonl':
            item = self.read_jsonl(path_abs)
        elif self.format == 'npy':
            item = self.read_npy(path_abs)
        
        return item