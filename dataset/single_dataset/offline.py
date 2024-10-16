# Copyright (c) OpenMMLab. All rights reserved.
import os
import json
import jsonlines
import numpy as np
import pickle
from mmengine import print_log
from torch.utils.data import Dataset
from registry import DATASETS

@DATASETS.register_module()
class OfflineDataset(Dataset):
    def  __init__(self, folder):
        super().__init__()
        assert folder is not None
        self.folder = folder
        if os.path.isdir(self.folder):
            self.data = os.listdir(folder)
            self.format = self.data[0].split(".")[-1]
        else:
            self.format = self.folder.split(".")[-1]
            if self.format == 'json':
                self.data = self.read_json(self.folder)
            elif self.format == 'jsonl':
                self.data = self.read_jsonl(self.folder)
            elif self.format == 'npy':
                self.data = self.read_npy(self.folder)
            elif self.format == 'pkl':
                self.data = self.read_pkl(self.folder)

        assert self.format in ['json', 'jsonl', 'npy','pkl']

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

    def read_pkl(self, pkl_path):
        with open(pkl_path,"rb") as f:
            try:
                datas = pickle.load(f)
            except:
                print(f"{pkl_path} is wrong!")
                path = "/data/Aaronzhu/DatasetStage2and3/llava-instruct/offline/0.pkl"
                with open(path,"rb") as f1:
                    datas = pickle.load(f1)
                    f1.close()
            f.close()
        return datas

    def __getitem__(self, index):
        if os.path.isdir(self.folder):
            path_abs = os.path.join(self.folder, self.data[index])
            if self.format == 'json':
                item = self.read_json(path_abs)
            elif self.format == 'jsonl':
                item = self.read_jsonl(path_abs)
            elif self.format == 'npy':
                item = self.read_npy(path_abs)
            elif self.format == 'pkl':
                item = self.read_pkl(path_abs)
            item['offline_path'] = path_abs
        else:
            item = self.data[index]    
        
        return item