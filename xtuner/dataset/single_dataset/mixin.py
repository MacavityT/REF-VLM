import json
import jsonlines
import os
import logging
import jsonlines
import numpy as np
from typing import List
from mmengine import print_log

from torch.utils.data import Dataset
from xtuner.registry import BUILDER, DATASETS
from .offline import OfflineDataset
from .dataset_templates import dataset_template_path
from ..utils import imfrombytes


class QuestionTemplateMixin:
    def __init__(
            self,
            *args,
            offline_processed_text_folder=None,
            template_name=None,
            template_file=None,
            max_dynamic_size=None,
            placeholders=None,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.template_file = template_file
        self.template_name = template_name
        self.max_dynamic_size = max_dynamic_size
        self.template_placeholders = placeholders
        self.offline_processed_text_folder = offline_processed_text_folder

        if self.offline_processed_text_folder is not None and os.path.exists(self.offline_processed_text_folder):
            # assert template_name or template_file, ("assign neither template_name nor template_file")
            if template_name is None and template_file is None:
                print_log("Warning: No template, please check whether the dataset has valid questions!")

            if template_name is not None and template_file is not None:
                raise ValueError(f"assign both template_name and template_file:\nstring:{template_name}\nfile:{template_file}")
            
            # Aaron: add template_name / template_file could be inputted as List format
            if template_name is not None:
                if isinstance(template_name,List):
                    self.templates = {}
                    for i,template_name_single in enumerate(template_name):
                        self.template_file = dataset_template_path[template_name_single]
                        self.templates[template_name_single] = json.load(open(self.template_file, 'r', encoding='utf8'))
                        if self.max_dynamic_size is not None:
                            self.templates[template_name_single] = self.templates[template_name_single][: self.max_dynamic_size]

                        # sanity check
                        assert self.template_placeholders is not None
                        # because template name is list, placeholders should be list as well
                        # [(Placeholder1, Placeholder2), (Placeholder3, Placeholder4)]
                        assert isinstance(self.template_placeholders,List)  
                        for template in self.templates[template_name_single]:
                            for placeholder in self.template_placeholders[i]:
                                assert str(template).count(placeholder) == 1, f"template: {template}\nplaceholder:{placeholder}"
                else:
                    self.template_file = dataset_template_path[template_name]
                    self.templates = json.load(open(self.template_file, 'r', encoding='utf8'))
                    if self.max_dynamic_size is not None:
                        self.templates = self.templates[: self.max_dynamic_size]

                    # sanity check
                    assert self.template_placeholders is not None
                    for template in self.templates:
                        for placeholder in self.template_placeholders:
                            assert str(template).count(placeholder) == 1, f"template: {template}\nplaceholder:{placeholder}"

    def get_template(self):
        import random
        return random.choice(self.templates)

    def template_nums(self):
        return len(self.templates)


class MInstrDataset(QuestionTemplateMixin, Dataset):
    _repr_indent = 4

    def __init__(self,
                text_path, 
                image_folder=None,
                image_info_folder=None, 
                stage=1,
                offline_processed_image_folder=None,
                map_placeholders=None,
                enforce_online=False, 
                seed=None,
                **kwargs):
        super().__init__(**kwargs)
        self.text_path = text_path
        self.image_folder = image_folder
        self.image_info_folder = image_info_folder
        self.map_placeholders = map_placeholders
        self.stage = stage
        self.rng = np.random.default_rng(seed)
        self.enforce_online = enforce_online
        self.offline_processed_image_folder = offline_processed_image_folder

        assert offline_processed_image_folder or image_folder
        if offline_processed_image_folder and os.path.exists(offline_processed_image_folder) and image_folder:
            print_log(
                'Both `offline_processed_image_folder` and '
                '`image_folder` are set, and we load dataset from'
                '`offline_processed_image_folder` '
                f'({offline_processed_image_folder})',
                logger='current',
                level=logging.WARNING)

        assert self.offline_processed_text_folder or text_path
        if self.offline_processed_text_folder and os.path.exists(self.offline_processed_text_folder) \
            and text_path and (not enforce_online):
            print_log(
                'Both `offline_processed_text_folder` and '
                '`data_path` are set, and we load dataset from'
                '`offline_processed_text_folder` '
                f'({self.offline_processed_text_folder})',
                logger='current',
                level=logging.WARNING)

        if offline_processed_image_folder is not None and \
            os.path.exists(offline_processed_image_folder):
            self.image_data = self.load_offline_image_data(offline_processed_image_folder)

        self.image_data_info = None
        if (self.offline_processed_text_folder is not None) and \
            os.path.exists(self.offline_processed_text_folder) and (not enforce_online):
            self.text_data = self.load_offline_text_data(self.offline_processed_text_folder)
        else:
            if os.path.isfile(text_path):   # judge whether the input path is a jsonfile or a directory.
                self.text_data = self.get_file_data(text_path)
            else:
                self.text_data = None
            if image_info_folder is not None:
                self.image_data_info = self.get_file_data(image_info_folder)
                if isinstance(self.image_data_info, list):
                    rearrange = dict()
                    for info in self.image_data_info:
                        if isinstance(info, str): 
                            info = json.loads(info)
                        rearrange.update(info)
                    self.image_data_info = rearrange
         
    def load_offline_text_data(self, offline_processed_text_folder):
        
        offline_dataset_args = dict(
            type = 'OfflineDataset',
            folder = offline_processed_text_folder
        )
        return DATASETS.build(offline_dataset_args)

    def load_offline_image_data(self, offline_processed_image_folder, image_path):
        raise NotImplementedError

    def get_raw_item(self, index):
        return json.loads(self.text_data[index])
    
    def get_file_data(self, file_path):
        if file_path.endswith('.json'):
            file_data = json.load(open(file_path))
        elif file_path.endswith('.jsonl'):
            file_data = []
            with open(file_path, 'r', encoding='utf8') as f:
                for line in f:
                    file_data.append(line)
        return file_data

    def get_image(self, image_path):
        if self.image_folder is not None:
            image_path_abs = os.path.join(self.image_folder, image_path)
        else:
            image_path_abs = image_path
        # image = Image.open(image_path).convert('RGB')

        item = {'path': image_path_abs}
        if self.image_info_folder is not None:
            key_example = list(self.image_data_info.keys())[0]
            image_path_abs_list = image_path_abs.split("/")
            if len(key_example.split("/")) == 2:
                image_path = os.path.join(image_path_abs_list[-2],image_path_abs_list[-1])
            elif len(key_example.split("/")) == 3:
                image_path = os.path.join(image_path_abs_list[-3],image_path_abs_list[-2],image_path_abs_list[-1])

            try:
                width = self.image_data_info[image_path]['width']
                height = self.image_data_info[image_path]['height']
                item['width'] = width
                item['height'] = height
            except:
                Warning(f"get height and width info failed, image path: {image_path}")
        return item

    def get_template(self):
        return self.rng.choice(self.templates)

    def __getitem__(self, index):
        '''region
        item = {
            'image': {
                'path': '/path/to/image', # str
                'width': 512, # int
                'height': 512, # int 
                'pixel_values': np.array
            },
            'target': {
                # xmin, ymin, xmax, ymax
                'boxes': [
                    [10, 10, 256, 265],  # dog1
                    [24, 18, 378, 768],  # dog2
                    [100, 310, 670, 653],  # man
                    [278, 320, 809, 673],  # rope
                ],
            },
            "conversations": [
                {
                    'from': 'system',
                    'value': [dict(task=xxx, unit=xxx)],
                
                },
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
        endregion'''
        if isinstance(self.text_data, OfflineDataset):
            item = self.text_data[index]
            item['map_placeholders'] = self.map_placeholders
        else:
            item = None
        return item

    def __len__(self):
        return len(self.text_data)

    def __repr__(self) -> str:
        head = "Dataset " + self.__class__.__name__
        body = [
            f"Number of datapoints: {self.__len__()}",
            f"ann file: {self.text_path}"
        ]
        if self.image_folder is not None:
            body.append(f"image folder: {self.image_folder}")
        body += self.extra_repr().splitlines()
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return "\n".join(lines)

    # noinspection PyMethodMayBeStatic
    def extra_repr(self) -> str:
        return ""
